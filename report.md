# Отчёт по домашнему заданию 2

## Abstract

Тритмент — двухстадийный i2i: retrieval через **LightFM с item-фичами** (artist, genres) на listen-time-взвешенных взаимодействиях, и реранкер на основе **LightGBM с quantile-loss (alpha=0.7)**, который предсказывает *ожидаемое время прослушивания* кандидата напрямую (continuous regression на target_time, не relevance-метки). Финальная выдача отбирается через **MMR на TruncatedSVD-эмбеддингах** для диверсификации. Идея: предсказывать «верхний квантиль» прослушки — оптимистично ранжируем кандидатов под метрику mean_session_time, а не под NDCG.

## Детали реализации

**Retrieval (LightFM, WARP):** строим `Dataset` через `rectools` с item features (artist, genre, multi-hot), фильтруем взаимодействия по `time > 0.5`, weight = listen_time. Обучаем `LightFMWrapperModel(LightFM(no_components=64, loss="warp"))`. Через `recommend_to_items` получаем top-500 кандидатов на каждый из топ-15000 anchor-треков. Параллельно фитим TruncatedSVD(k=64) на той же user-item матрице — эти эмбеддинги используются только для diversity-штрафа в MMR.

**Reranker (LightGBM, quantile loss):** для каждой пары `(anchor, candidate)` из ретривал-листа считаем 7 поведенческих фич: `cos_sim` (косинус по SVD-эмбеддингам), `log_transitions`, `log_skips`, `skip_rate_wilson_lb` (нижняя граница Wilson на skip-rate, чтобы давать больше веса пары с большим объёмом наблюдений), `anchor_stickiness` и `candidate_stickiness` (средний listen_time по anchor/candidate во всех логах), `inv_rank` (1/(rank+1) из ретривала). Никаких контентных фичей в реранкере. Таргет — continuous `target_time` (доля прослушки кандидата в наблюдённых переходах). Loss — `objective="quantile", alpha=0.7`: оптимистичная оценка ожидаемого listen-time. Для финальной выдачи применяем **MMR** с `λ=0.7`: на каждом шаге выбираем кандидата `argmax (λ · ŷ − (1−λ) · max_cos_to_picked)`. Top-200 на anchor пишем в `learned_i2i.jsonl`.

В сервисе experiment `LEARNED_I2I_QR` с разбивкой HALF_HALF: контроль — SasRec-I2I, тритмент — `I2IRecommender` поверх learned_i2i.jsonl с фолбэком на SasRec-I2I.

## Результаты A/B эксперимента

Эксперимент `LEARNED_I2I_QR`, разбивка HALF_HALF, EPISODES=30000, seed=31312. Контроль — SasRec-I2I, тритмент — LightFM retrieval + quantile reranker + MMR. Обучающие данные — 40k эпизодов на bootstrap-конфигурации (контроль SasRec-I2I, тритмент LightFM-i2i как разнообразие), 223k positive interactions, 107k обучающих пар (anchor, candidate).

| Метрика | Контроль | Тритмент | Эффект | 95% CI | Значимо |
|---|---|---|---|---|---|
| **mean_time_per_session** | 6.96 с | **8.87 с** | **+27.57%** | [+24.75%, +30.39%] | да |
| mean_tracks_per_session | 11.94 | 13.86 | +16.12% | [+14.39%, +17.84%] | да |
| time | 21.77 с | 27.02 с | +24.13% | [+20.53%, +27.73%] | да |
| sessions | 3.18 | 3.13 | −1.48% | [−3.55%, +0.58%] | нет |
| mean_request_latency | 0.46 мс | 0.58 мс | +27.73% | [−22.90%, +78.36%] | нет |

Главная метрика выросла на +27.57% при значимости. По важности фичей в LightGBM (gain) лидирует `skip_rate_wilson_lb` (165k) — нижняя граница Wilson на skip-rate действительно хорошо разделяет «прилипчивые» переходы от пропускаемых.
