import argparse
import glob
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


SKIP_THRESHOLD = 0.5
ANCHOR_MIN_TIME = 0.3
QUANTILE_ALPHA = 0.7
MMR_LAMBDA = 0.7
WILSON_Z = 1.96


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--topk", type=int, default=200)
    p.add_argument("--n-estimators", type=int, default=600)
    p.add_argument("--learning-rate", type=float, default=0.04)
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--seed", type=int, default=31312)
    return p.parse_args()


def load_logs(log_dir: Path) -> pd.DataFrame:
    paths = sorted(glob.glob(str(log_dir / "**/data.json*"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"no data.json under {log_dir}")
    return pd.concat([pd.read_json(p, lines=True) for p in paths], ignore_index=True)


def load_candidates(path: Path) -> dict:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[int(row["item_id"])] = [int(x) for x in row["recommendations"]]
    return out


def load_meta(path: Path):
    npz = np.load(path)
    item_ids = npz["item_ids"]
    item_embs = npz["item_embs"]
    idx = {int(t): i for i, t in enumerate(item_ids.tolist())}
    return idx, item_embs


def build_transitions(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["message"].isin(["next", "last"])].copy()
    base = base.sort_values(["user", "timestamp"])
    base["next_track"] = base.groupby("user")["track"].shift(-1)
    base["next_time"] = base.groupby("user")["time"].shift(-1)

    anchors = base[base["message"] == "next"].copy()
    anchors = anchors.dropna(subset=["next_track", "next_time", "recommendation"])
    anchors["next_track"] = anchors["next_track"].astype(int)
    anchors["recommendation"] = anchors["recommendation"].astype(int)
    anchors = anchors[anchors["next_track"] == anchors["recommendation"]]
    anchors = anchors[anchors["time"] >= ANCHOR_MIN_TIME]
    anchors["track"] = anchors["track"].astype(int)
    return anchors[["track", "next_track", "next_time"]].rename(columns={
        "track": "anchor",
        "next_track": "candidate",
        "next_time": "y",
    })


def aggregate_pair_stats(transitions: pd.DataFrame) -> pd.DataFrame:
    transitions = transitions.copy()
    transitions["is_skip"] = (transitions["y"] < SKIP_THRESHOLD).astype(int)
    grouped = transitions.groupby(["anchor", "candidate"]).agg(
        mean_y=("y", "mean"),
        n_trans=("y", "size"),
        n_skip=("is_skip", "sum"),
    ).reset_index()
    return grouped


def stickiness_maps(transitions: pd.DataFrame):
    by_anchor = transitions.groupby("anchor")["y"].mean()
    by_candidate = transitions.groupby("candidate")["y"].mean()
    global_mean = float(transitions["y"].mean())
    return by_anchor.to_dict(), by_candidate.to_dict(), global_mean


def wilson_lower_bound(skips: float, trans: float) -> float:
    if trans <= 0:
        return 0.5
    p = skips / trans
    z = WILSON_Z
    denom = 1.0 + z * z / trans
    centre = p + z * z / (2.0 * trans)
    half = z * math.sqrt(p * (1.0 - p) / trans + z * z / (4.0 * trans * trans))
    return max(0.0, (centre - half) / denom)


def cosine_lookup(embs: np.ndarray, idx_map: dict, anchor: int, candidate: int) -> float:
    a = idx_map.get(anchor)
    c = idx_map.get(candidate)
    if a is None or c is None:
        return 0.0
    va = embs[a]
    vc = embs[c]
    na = np.linalg.norm(va)
    nc = np.linalg.norm(vc)
    if na == 0 or nc == 0:
        return 0.0
    return float(np.dot(va, vc) / (na * nc))


def assemble_features(
    pairs,
    pair_stats: dict,
    anchor_stick: dict,
    cand_stick: dict,
    global_mean: float,
    embs: np.ndarray,
    idx_map: dict,
):
    rows = np.empty((len(pairs), 7), dtype=np.float32)
    for i, (anchor, candidate, rank) in enumerate(pairs):
        stats = pair_stats.get((anchor, candidate))
        if stats is None:
            n_trans, n_skip = 0.0, 0.0
        else:
            n_trans = float(stats["n_trans"])
            n_skip = float(stats["n_skip"])
        rows[i, 0] = cosine_lookup(embs, idx_map, anchor, candidate)
        rows[i, 1] = math.log1p(n_trans)
        rows[i, 2] = math.log1p(n_skip)
        rows[i, 3] = wilson_lower_bound(n_skip, n_trans)
        rows[i, 4] = anchor_stick.get(anchor, global_mean)
        rows[i, 5] = cand_stick.get(candidate, global_mean)
        rows[i, 6] = 1.0 / (rank + 1.0)
    return rows


FEATURE_NAMES = [
    "cos_sim",
    "log_transitions",
    "log_skips",
    "skip_rate_wilson_lb",
    "anchor_stickiness",
    "candidate_stickiness",
    "inv_rank",
]


def build_training_matrix(
    pair_stats_df: pd.DataFrame,
    candidates: dict,
    pair_stats_dict: dict,
    anchor_stick: dict,
    cand_stick: dict,
    global_mean: float,
    embs: np.ndarray,
    idx_map: dict,
):
    rank_lookup = {}
    for anchor, recs in candidates.items():
        for r, cand in enumerate(recs):
            rank_lookup[(anchor, cand)] = r

    keep_rows = []
    for row in pair_stats_df.itertuples(index=False):
        rank = rank_lookup.get((row.anchor, row.candidate))
        if rank is None:
            continue
        keep_rows.append((row.anchor, row.candidate, rank, float(row.mean_y)))

    if not keep_rows:
        raise RuntimeError("no overlap between observed pairs and retrieval candidates")

    pairs = [(a, c, r) for (a, c, r, _) in keep_rows]
    targets = np.array([y for (_, _, _, y) in keep_rows], dtype=np.float32)
    features = assemble_features(pairs, pair_stats_dict, anchor_stick, cand_stick, global_mean, embs, idx_map)
    print(f"  training rows: {len(targets)} y_mean={targets.mean():.3f} y_std={targets.std():.3f}")
    return features, targets


def fit_quantile_ranker(X, y, args):
    train_set = lgb.Dataset(X, label=y, feature_name=FEATURE_NAMES)
    params = {
        "objective": "quantile",
        "alpha": QUANTILE_ALPHA,
        "metric": "quantile",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "seed": args.seed,
        "verbose": -1,
    }
    model = lgb.train(params, train_set, num_boost_round=args.n_estimators)
    importances = dict(zip(FEATURE_NAMES, model.feature_importance(importance_type="gain")))
    print("  feature importances (gain):")
    for k, v in sorted(importances.items(), key=lambda kv: -kv[1]):
        print(f"    {k}: {v:.0f}")
    return model


def mmr_select(scores: np.ndarray, sub_embs: np.ndarray, topk: int) -> list:
    n = len(scores)
    if n == 0:
        return []
    norms = np.linalg.norm(sub_embs, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    unit = sub_embs / safe
    sim = unit @ unit.T

    picked = []
    sim_to_picked = np.full(n, -np.inf, dtype=np.float32)
    available = np.ones(n, dtype=bool)

    for _ in range(min(topk, n)):
        if not picked:
            mmr = scores.copy()
        else:
            mmr = MMR_LAMBDA * scores - (1.0 - MMR_LAMBDA) * sim_to_picked
        mmr = np.where(available, mmr, -np.inf)
        idx = int(np.argmax(mmr))
        picked.append(idx)
        available[idx] = False
        sim_to_picked = np.maximum(sim_to_picked, sim[idx])
    return picked


def rerank_all(
    candidates: dict,
    model,
    pair_stats_dict: dict,
    anchor_stick: dict,
    cand_stick: dict,
    global_mean: float,
    embs: np.ndarray,
    idx_map: dict,
    topk: int,
):
    out = {}
    n_anchors = len(candidates)
    for k, (anchor, cand_list) in enumerate(candidates.items()):
        if k % 2000 == 0:
            print(f"  reranking {k}/{n_anchors}")

        triples = [(anchor, c, r) for r, c in enumerate(cand_list)]
        feats = assemble_features(triples, pair_stats_dict, anchor_stick, cand_stick, global_mean, embs, idx_map)
        scores = model.predict(feats).astype(np.float32)

        sub_idx = np.array([idx_map.get(c, -1) for c in cand_list])
        valid = sub_idx >= 0
        if valid.sum() < len(cand_list):
            fallback = embs.mean(axis=0)
            sub_embs = np.where(valid[:, None], embs[np.where(valid, sub_idx, 0)], fallback).astype(np.float32)
        else:
            sub_embs = embs[sub_idx].astype(np.float32)

        order = mmr_select(scores, sub_embs, topk)
        out[anchor] = [int(cand_list[i]) for i in order]
    return out


def write_jsonl(records: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for anchor, recs in records.items():
            f.write(json.dumps({"item_id": int(anchor), "recommendations": recs}) + "\n")


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print(f"[ranker] loading logs from {args.logs}")
    raw = load_logs(Path(args.logs))
    print(f"  raw events: {len(raw)}")

    candidates = load_candidates(Path(args.candidates))
    print(f"  retrieval anchors: {len(candidates)}")

    idx_map, embs = load_meta(Path(args.meta))
    print(f"  embeddings: {embs.shape}")

    transitions = build_transitions(raw)
    print(f"  observed transitions: {len(transitions)}")

    pair_stats_df = aggregate_pair_stats(transitions)
    pair_stats_dict = {
        (int(r.anchor), int(r.candidate)): {"n_trans": r.n_trans, "n_skip": r.n_skip, "mean_y": r.mean_y}
        for r in pair_stats_df.itertuples(index=False)
    }
    anchor_stick, cand_stick, global_mean = stickiness_maps(transitions)

    print("[ranker] building training matrix")
    X, y = build_training_matrix(
        pair_stats_df, candidates, pair_stats_dict,
        anchor_stick, cand_stick, global_mean, embs, idx_map,
    )

    print(f"[ranker] fitting quantile regressor (alpha={QUANTILE_ALPHA})")
    model = fit_quantile_ranker(X, y, args)

    print(f"[ranker] reranking with MMR (lambda={MMR_LAMBDA}, topk={args.topk})")
    reranked = rerank_all(
        candidates, model, pair_stats_dict,
        anchor_stick, cand_stick, global_mean, embs, idx_map, args.topk,
    )

    write_jsonl(reranked, Path(args.output))
    print(f"[ranker] wrote {len(reranked)} reranked anchors -> {args.output}")


if __name__ == "__main__":
    main()
