"""Calibrate a wind-structure classifier and test event tracking on development data.

This is the final, more discriminating branch of Stage 6B.  Labels are used
only inside 1980--2001; no 2002--2023 data are read until a configuration is
frozen by the companion detector workflow.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import develop_vorticity_only_detector as det


FEATURES = [
    "log_vorticity", "log_vortex_ratio", "core_mean_vorticity_1e5",
    "core_coherence", "center_wind_ms", "wind_speed_deficit_ms", "circulation_ms",
]


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["log_vorticity"] = np.log(np.maximum(out.vorticity_s_1, 1e-7) / 1e-5)
    out["log_vortex_ratio"] = np.log1p(out.vortex_ratio.clip(lower=0, upper=1e4))
    out["core_mean_vorticity_1e5"] = out.core_mean_vorticity_s_1 * 1e5
    out["core_coherence"] = out.core_mean_vorticity_s_1 / out.vorticity_s_1.clip(lower=1e-7)
    return out


def label_centres(candidates: pd.DataFrame, ref_points: pd.DataFrame) -> np.ndarray:
    labels = np.zeros(len(candidates), dtype=np.int8)
    groups = candidates.groupby("time").groups
    for time, refs in ref_points.groupby("time"):
        idx = groups.get(time)
        if idx is None:
            continue
        c = candidates.loc[idx]
        dist = det.haversine_km(c.lat.values[:, None], c.lon.values[:, None],
                                refs.lat.values[None, :], refs.lon.values[None, :])
        labels[np.asarray(idx)] = (dist.min(axis=1) <= 300.0).astype(np.int8)
    return labels


def flatten_metrics(cfg: dict, metrics: dict) -> dict:
    row = {**cfg, **{k: v for k, v in metrics.items() if k != "seasonal"}}
    for sea, vals in metrics["seasonal"].items():
        row.update({f"{sea}_{k}": v for k, v in vals.items()})
    return row


def main() -> None:
    base = det.DetectorConfig(.25, .5, .75, 1e-5, 0.0, 2, 0)
    candidates = det.load_candidates(det.DEV_YEARS, base).reset_index(drop=True)
    ref_events, ref_points = det.reference_data(det.DEV_YEARS)
    labels = label_centres(candidates, ref_points)
    data = add_features(candidates)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(20260906)
    train_idx = np.r_[positive, rng.choice(negative, size=min(len(negative), 15 * len(positive)), replace=False)]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
    model.fit(data.loc[train_idx, FEATURES], labels[train_idx])
    data["structure_score"] = model.decision_function(data[FEATURES])
    quantiles = [0.94, 0.96, 0.97, 0.98, 0.99, 0.995]
    thresholds = {q: float(data.structure_score.quantile(q)) for q in quantiles}
    rows = []
    best_key, best_score, best_products = None, None, None
    for quantile, threshold in thresholds.items():
        selected = data.loc[data.structure_score >= threshold].copy()
        for min_steps in [2, 3, 4]:
            for gap in [0, 1]:
                cfg = det.DetectorConfig(.25, .5, .75, 1e-5, 0.0, min_steps, gap)
                events, points = det.track_candidates(selected, cfg)
                metrics, counts = det.evaluate(events, points, ref_events, ref_points, "1980-2001 development")
                params = {"score_quantile": quantile, "score_threshold": threshold,
                          "min_steps": min_steps, "max_gap_steps": gap, "n_selected_centres": len(selected)}
                rows.append(flatten_metrics(params, metrics))
                rank = det.config_score(metrics)
                if best_score is None or rank > best_score:
                    best_key, best_score, best_products = params, rank, (events, points, metrics, counts)
                print(f"q={quantile:g} steps={min_steps} gap={gap} events={len(events)} F1={metrics['event_f1']:.3f}", flush=True)
    out = det.OUT
    pd.DataFrame(rows).to_csv(out / "vorticity_structure_classifier_calibration.csv", index=False)
    events, points, metrics, counts = best_products
    events.to_csv(out / "vorticity_structure_events_development.csv.gz", index=False)
    points.to_csv(out / "vorticity_structure_points_development.csv.gz", index=False)
    counts.to_csv(out / "vorticity_structure_seasonal_counts_development.csv", index=False)
    joblib.dump(model, out / "vorticity_structure_classifier.joblib")
    summary = {
        "period": "1980-2001 development only", "n_candidate_centres": len(data),
        "n_positive_centres_within_300km": int(labels.sum()), "features": FEATURES,
        "selected_config": best_key, "selected_development_metrics": metrics,
        "validation_data_accessed": False,
    }
    (out / "vorticity_structure_classifier_development.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
