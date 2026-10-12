"""Test whether DJF covariates can statistically account for the count trend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
DOC = ROOT / "docs" / "djf_mechanism_trend_closure_zh.md"
INPUT = OUT / "djf_mechanism_indices.csv"
CSV = OUT / "djf_mechanism_trend_closure.csv"
JSON = OUT / "djf_mechanism_trend_closure_summary.json"
RESPONSE = "cluster1_frequency"


def bh_adjust(values: pd.Series) -> pd.Series:
    order = values.sort_values()
    n = len(order)
    adjusted = (order * n / np.arange(1, n + 1)).iloc[::-1].cummin().iloc[::-1]
    return adjusted.clip(upper=1).reindex(values.index)


def main() -> None:
    table = pd.read_csv(INPUT)
    required = {"season_year", RESPONSE}
    if not required.issubset(table.columns) or len(table) != 44:
        raise RuntimeError("Expected 44 winters with season_year and cluster1_frequency")

    year_decade = (table["season_year"] - table["season_year"].mean()) / 10.0
    base_design = sm.add_constant(pd.DataFrame({"year_decade": year_decade}))
    base = sm.GLM(table[RESPONSE], base_design, family=sm.families.Poisson()).fit(
        cov_type="HAC", cov_kwds={"maxlags": 3}
    )
    base_year = float(base.params["year_decade"])
    predictors = [c for c in table.columns if c not in {"season_year", "frequency", RESPONSE}]
    rows: list[dict[str, float | str | bool]] = []

    for predictor in predictors:
        raw = table[predictor].astype(float)
        sd = float(raw.std(ddof=1))
        if not np.isfinite(sd) or sd == 0:
            raise RuntimeError(f"Degenerate predictor: {predictor}")
        z = (raw - raw.mean()) / sd

        trend = sm.OLS(z, base_design).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        joint_design = sm.add_constant(pd.DataFrame({"year_decade": year_decade, "index_z": z}))
        joint = sm.GLM(table[RESPONSE], joint_design, family=sm.families.Poisson()).fit(
            cov_type="HAC", cov_kwds={"maxlags": 3}
        )
        association = float(joint.params["index_z"])
        predictor_trend = float(trend.params["year_decade"])
        adjusted_year = float(joint.params["year_decade"])
        implied = association * predictor_trend
        attenuation = 100.0 * (base_year - adjusted_year) / base_year
        rows.append(
            {
                "predictor": predictor,
                "predictor_trend_sd_per_decade": predictor_trend,
                "predictor_trend_p_hac3": float(trend.pvalues["year_decade"]),
                "count_log_rate_per_sd_adjusted_for_year": association,
                "count_association_p_hac3": float(joint.pvalues["index_z"]),
                "base_year_log_rate_per_decade": base_year,
                "adjusted_year_log_rate_per_decade": adjusted_year,
                "year_trend_attenuation_percent": attenuation,
                "implied_log_rate_per_decade": implied,
                "direction_consistent_with_observed_trend": bool(np.sign(implied) == np.sign(base_year)),
            }
        )

    result = pd.DataFrame(rows)
    result["predictor_trend_p_fdr"] = bh_adjust(result["predictor_trend_p_hac3"])
    result["count_association_p_fdr"] = bh_adjust(result["count_association_p_hac3"])
    result["passes_trend_closure"] = (
        (result["predictor_trend_p_fdr"] < 0.05)
        & (result["count_association_p_fdr"] < 0.05)
        & result["direction_consistent_with_observed_trend"]
        & (result["year_trend_attenuation_percent"] > 0)
    )
    result = result.sort_values("year_trend_attenuation_percent", ascending=False)
    result.to_csv(CSV, index=False)

    passed = result.loc[result["passes_trend_closure"], "predictor"].tolist()
    summary = {
        "period": [int(table.season_year.min()), int(table.season_year.max())],
        "n_winters": int(len(table)),
        "response": RESPONSE,
        "base_log_rate_per_decade": base_year,
        "base_rate_percent_per_decade": 100 * np.expm1(base_year),
        "base_trend_p_hac3": float(base.pvalues["year_decade"]),
        "multiple_testing": "Benjamini-Hochberg separately for predictor trends and count associations",
        "trend_closure_criteria": "FDR<0.05 for both tests, consistent direction, and positive attenuation",
        "n_passing": len(passed),
        "passing_predictors": passed,
    }
    JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# DJF 候选机制的趋势闭合检验",
        "",
        "本检验区分‘控制时间后与年际频数相关’和‘能够解释频数年代际趋势’。后者要求候选指标自身具有显著趋势、与频数的条件关联显著、二者乘积方向与观测趋势一致，并且加入指标后年份系数减弱。两组检验分别采用 Benjamini–Hochberg FDR 校正。该分析仍是统计归因诊断，不等同于因果证明。",
        "",
        f"- 样本：{summary['n_winters']} 个冬季（{summary['period'][0]}--{summary['period'][1]}）",
        f"- 基础 Poisson 年份趋势：{summary['base_rate_percent_per_decade']:.2f}% decade⁻¹（HAC p={summary['base_trend_p_hac3']:.3g}）",
        f"- 通过预设趋势闭合标准的指标数：{summary['n_passing']}",
        f"- 通过指标：{', '.join(passed) if passed else '无'}",
        "",
        "完整逐指标结果见 `data/processed/djf_mechanism_trend_closure.csv`。若没有指标通过，不应把最强年际条件关联解释为观测增长的驱动机制。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
