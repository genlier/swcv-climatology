"""One-shot forward validation of the frozen Stage-6B wind-only detector."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import develop_vorticity_only_detector as det
from calibrate_vorticity_structure_classifier import FEATURES, add_features


def passes(metrics: dict) -> dict:
    return {
        "event_f1_at_least_0.6": bool(metrics["event_f1"] >= 0.6),
        "DJF_count_r_at_least_0.6": bool(metrics["seasonal"]["DJF"]["pearson_r"] >= 0.6),
        "JJA_count_r_at_least_0.6": bool(metrics["seasonal"]["JJA"]["pearson_r"] >= 0.6),
        "DJF_absolute_mean_bias_at_most_0.3": bool(abs(metrics["seasonal"]["DJF"]["mean_bias_fraction"]) <= 0.3),
        "JJA_absolute_mean_bias_at_most_0.3": bool(abs(metrics["seasonal"]["JJA"]["mean_bias_fraction"]) <= 0.3),
    }


def make_figure(dev: dict, val: dict, counts: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].bar([0, 1], [dev["event_f1"], val["event_f1"]], color=["#7aa6c2", "#c65d4b"])
    axes[0].axhline(.6, color="k", ls="--", lw=1)
    axes[0].set(xticks=[0, 1], xticklabels=["Development", "Forward validation"], ylabel="Event F1", ylim=(0, .7))
    for ax, season in zip(axes[1:], ["DJF", "JJA"]):
        q = counts.loc[counts.season == season]
        ax.scatter(q.reference, q.detected, color="#3973ac")
        lim = max(q.reference.max(), q.detected.max()) + 2
        ax.plot([0, lim], [0, lim], color="0.5", ls="--", lw=1)
        r = val["seasonal"][season]["pearson_r"]
        bias = val["seasonal"][season]["mean_bias_fraction"]
        ax.set(title=f"{season}: r={r:.2f}, bias={bias:+.0%}", xlabel="Reference events yr$^{-1}$",
               ylabel="Detected events yr$^{-1}$", xlim=(0, lim), ylim=(0, lim))
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(det.FIG / "vorticity_detector_forward_validation.png", dpi=180); plt.close(fig)


def main() -> None:
    out = det.OUT
    development = json.loads((out / "vorticity_structure_classifier_development.json").read_text(encoding="utf-8"))
    frozen = development["selected_config"]
    model = joblib.load(out / "vorticity_structure_classifier.joblib")
    config = det.DetectorConfig(.25, .5, .75, 1e-5, 0.0, int(frozen["min_steps"]), int(frozen["max_gap_steps"]))
    candidates = add_features(det.load_candidates(det.VAL_YEARS, config).reset_index(drop=True))
    candidates["structure_score"] = model.decision_function(candidates[FEATURES])
    selected = candidates.loc[candidates.structure_score >= float(frozen["score_threshold"])].copy()
    events, points = det.track_candidates(selected, config)
    ref_events, ref_points = det.reference_data(det.VAL_YEARS)
    metrics, counts = det.evaluate(events, points, ref_events, ref_points, "2002-2023 forward validation")
    gate_results = passes(metrics)
    native_pass = all(gate_results.values())
    result = {
        "stage": "6B_ERA5_vorticity_only_detector_forward_validation",
        "development_period": "1980-2001", "forward_validation_period": "2002-2023",
        "frozen_before_validation": True, "height_field_used": False,
        "wind_inputs": ["ERA5 700-hPa u", "ERA5 700-hPa v"],
        "static_terrain_mask": "1980 ERA5 mean surface pressure >=725 hPa; fixed geographic ancillary mask",
        "frozen_config": frozen, "development_metrics": development["selected_development_metrics"],
        "forward_validation_metrics": metrics, "native_grid_gate_results": gate_results,
        "native_grid_pass": native_pass,
        "model_grid_degradation_performed": False,
        "model_grid_degradation_reason": "Native 0.25-degree ERA5 failed the necessary gate; coarsening cannot establish the preregistered requirement that native and at least two model grids pass.",
        "route_decision": "GO" if native_pass else "NO-GO",
        "scientific_action": ("Proceed to exact model-grid degradation tests." if native_pass else
                              "Stop direct SWCV frequency/intensity/path attribution from 6-hourly ua+va; retain monthly forced mechanism fingerprints only."),
    }
    events.to_csv(out / "vorticity_structure_events_forward_validation.csv.gz", index=False)
    points.to_csv(out / "vorticity_structure_points_forward_validation.csv.gz", index=False)
    counts.to_csv(out / "vorticity_structure_seasonal_counts_forward_validation.csv", index=False)
    (out / "vorticity_detector_forward_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    make_figure(development["selected_development_metrics"], metrics, counts)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
