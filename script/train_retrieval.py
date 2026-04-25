import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from lightfm import LightFM
from lightning_fabric import seed_everything
from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import LightFMWrapperModel
from sklearn.decomposition import TruncatedSVD


MIN_LISTEN = 0.5
TOP_ANCHORS = 15000
SVD_COMPONENTS = 64


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", required=True)
    p.add_argument("--tracks", required=True)
    p.add_argument("--out-candidates", required=True)
    p.add_argument("--out-meta", required=True)
    p.add_argument("--topk", type=int, default=500)
    p.add_argument("--components", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=31312)
    return p.parse_args()


def load_logs(log_dir: Path) -> pd.DataFrame:
    paths = sorted(glob.glob(str(log_dir / "**/data.json*"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"no data.json under {log_dir}")
    frames = [pd.read_json(p, lines=True) for p in paths]
    return pd.concat(frames, ignore_index=True)


def make_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["message"].isin(["next", "last"])].copy()
    df = df[df["time"] > MIN_LISTEN]
    df = df.rename(columns={
        "user": Columns.User,
        "track": Columns.Item,
        "timestamp": Columns.Datetime,
        "time": Columns.Weight,
    })
    df[Columns.Datetime] = pd.to_datetime(df[Columns.Datetime], unit="ms")
    return df[[Columns.User, Columns.Item, Columns.Datetime, Columns.Weight]]


def make_item_features(tracks_path: Path, items_in_train) -> pd.DataFrame:
    tracks = pd.read_json(tracks_path, lines=True).drop_duplicates(subset=["track"])
    tracks = tracks.rename(columns={"track": Columns.Item})
    tracks = tracks.loc[tracks[Columns.Item].isin(items_in_train)].copy()

    artist = tracks[[Columns.Item, "artist"]].copy()
    artist.columns = ["id", "value"]
    artist["feature"] = "artist"

    genres = tracks[[Columns.Item, "genres"]].explode("genres").dropna(subset=["genres"]).copy()
    genres.columns = ["id", "value"]
    genres["feature"] = "genre"

    return pd.concat([artist, genres], ignore_index=True)


def fit_retrieval(dataset: Dataset, components: int, epochs: int, seed: int) -> LightFMWrapperModel:
    backend = LightFM(no_components=components, loss="warp", random_state=seed)
    model = LightFMWrapperModel(backend, epochs=epochs, num_threads=4)
    model.fit(dataset)
    return model


def pick_anchors(interactions: pd.DataFrame, limit: int) -> list:
    counts = interactions.groupby(Columns.Item).size().sort_values(ascending=False)
    return counts.head(limit).index.tolist()


def to_jsonl(records: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for anchor, recs in records.items():
            f.write(json.dumps({"item_id": int(anchor), "recommendations": [int(r) for r in recs]}) + "\n")


def fit_svd_embeddings(interactions: pd.DataFrame, n_components: int, seed: int):
    items = interactions[Columns.Item].astype(int).to_numpy()
    users = interactions[Columns.User].astype(int).to_numpy()
    weights = interactions[Columns.Weight].astype(np.float32).to_numpy()

    item_ids = np.array(sorted(np.unique(items)))
    user_ids = np.array(sorted(np.unique(users)))
    item_idx = {t: i for i, t in enumerate(item_ids.tolist())}
    user_idx = {u: i for i, u in enumerate(user_ids.tolist())}

    rows = np.fromiter((user_idx[u] for u in users), dtype=np.int64, count=len(users))
    cols = np.fromiter((item_idx[t] for t in items), dtype=np.int64, count=len(items))
    matrix = sp.csr_matrix((weights, (rows, cols)), shape=(len(user_ids), len(item_ids)))

    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    svd.fit(matrix)
    return item_ids, svd.components_.T.astype(np.float32)


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)

    print(f"[retrieval] loading logs from {args.logs}")
    raw_df = load_logs(Path(args.logs))
    print(f"  raw events: {len(raw_df)}")

    interactions = make_interactions(raw_df)
    print(f"  positive interactions (time>{MIN_LISTEN}): {len(interactions)}")
    print(f"  unique users={interactions[Columns.User].nunique()}, items={interactions[Columns.Item].nunique()}")

    item_features = make_item_features(Path(args.tracks), interactions[Columns.Item].unique())
    print(f"  item-feature rows: {len(item_features)}")

    dataset = Dataset.construct(
        interactions_df=interactions,
        item_features_df=item_features,
        cat_item_features=["artist", "genre"],
    )

    print(f"[retrieval] fitting LightFM (components={args.components}, epochs={args.epochs})")
    model = fit_retrieval(dataset, args.components, args.epochs, args.seed)

    anchors = pick_anchors(interactions, TOP_ANCHORS)
    print(f"[retrieval] generating top-{args.topk} for {len(anchors)} anchors")

    i2i = model.recommend_to_items(
        target_items=anchors,
        dataset=dataset,
        k=args.topk,
        filter_itself=True,
        items_to_recommend=None,
    )

    grouped = (
        i2i.sort_values(["target_item_id", "rank"])
        .groupby("target_item_id")[Columns.Item]
        .apply(lambda s: s.tolist())
        .to_dict()
    )

    to_jsonl(grouped, Path(args.out_candidates))
    print(f"[retrieval] wrote {len(grouped)} anchors -> {args.out_candidates}")

    print(f"[retrieval] fitting TruncatedSVD for MMR embeddings (k={SVD_COMPONENTS})")
    item_ids, item_embs = fit_svd_embeddings(interactions, SVD_COMPONENTS, args.seed)
    np.savez(
        args.out_meta,
        item_ids=np.asarray(item_ids, dtype=np.int64),
        item_embs=item_embs.astype(np.float32),
    )
    print(f"[retrieval] wrote SVD embeddings ({item_embs.shape}) -> {args.out_meta}")


if __name__ == "__main__":
    main()
