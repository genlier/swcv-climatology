"""Inventory paired historical and hist-GHG zg members from frozen ESGF catalogues."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
OUT = ROOT / "data" / "processed"
DOC = ROOT / "docs" / "cmip6_multimember_availability_zh.md"
EXPERIMENTS = {"historical", "hist-GHG"}


def load_catalogues() -> pd.DataFrame:
    cloud = pd.read_csv(MANIFESTS / "cmip6_cloud_monthly_target_catalog.csv")
    cloud = cloud.rename(
        columns={
            "source_id": "model",
            "experiment_id": "experiment",
            "member_id": "member",
            "variable_id": "variable",
            "grid_label": "grid",
        }
    )
    cloud["catalog_source"] = "cloud_zarr_catalog"
    cloud["dataset_id"] = cloud.apply(
        lambda r: f"CMIP6.{r.activity_id}.{r.institution_id}.{r.model}.{r.experiment}.{r.member}.{r.table_id}.{r.variable}.{r.grid}.v{r.version}",
        axis=1,
    )
    cloud["access_url"] = cloud["zstore"].fillna("")

    files = pd.read_csv(MANIFESTS / "cmip6_monthly_esgf_file_manifest.csv")
    files = files.rename(columns={"source": "model"})
    files["catalog_source"] = "esgf_file_catalog"
    files["dataset_id"] = files["instance_id"]
    files["access_url"] = files["opendap_url"].fillna(files["http_url"]).fillna("")

    columns = [
        "model", "experiment", "member", "variable", "grid", "version",
        "dataset_id", "access_url", "catalog_source",
    ]
    combined = pd.concat([cloud[columns], files[columns]], ignore_index=True)
    return combined.drop_duplicates()


def main() -> None:
    registry = pd.read_csv(OUT / "cmip6_model_resolution_registry.csv")
    models = registry.model.tolist()
    data = load_catalogues()
    data = data.loc[
        data.model.isin(models)
        & data.experiment.isin(EXPERIMENTS)
        & data.variable.eq("zg")
    ].copy()

    detail_rows = []
    summary_rows = []
    for model in models:
        model_data = data.loc[data.model.eq(model)]
        members_by_experiment = {
            experiment: set(model_data.loc[model_data.experiment.eq(experiment), "member"])
            for experiment in sorted(EXPERIMENTS)
        }
        paired = sorted(set.intersection(*members_by_experiment.values()))
        summary_rows.append(
            {
                "model": model,
                "historical_zg_members": len(members_by_experiment["historical"]),
                "hist_ghg_zg_members": len(members_by_experiment["hist-GHG"]),
                "paired_member_count": len(paired),
                "paired_members": ";".join(paired),
                "eligible_at_least_3_pairs": len(paired) >= 3,
            }
        )
        for member in paired:
            for experiment in sorted(EXPERIMENTS):
                choices = model_data.loc[
                    model_data.member.eq(member) & model_data.experiment.eq(experiment)
                ].copy()
                # Prefer a cloud Zarr dataset, then the newest catalogue version.
                choices["is_zarr"] = choices.catalog_source.eq("cloud_zarr_catalog") & choices.access_url.ne("")
                choices = choices.sort_values(["is_zarr", "version"], ascending=[False, False])
                selected = choices.iloc[0]
                detail_rows.append(
                    {
                        "model": model,
                        "member": member,
                        "experiment": experiment,
                        "variable": "zg",
                        "grid": selected.grid,
                        "version": selected.version,
                        "dataset_id": selected.dataset_id,
                        "access_url": selected.access_url,
                        "catalog_source": selected.catalog_source,
                    }
                )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["eligible_at_least_3_pairs", "paired_member_count", "model"],
        ascending=[False, False, True],
    )
    detail = pd.DataFrame(detail_rows)
    summary.to_csv(OUT / "cmip6_multimember_availability.csv", index=False)
    detail.to_csv(MANIFESTS / "cmip6_multimember_zg_datasets.csv", index=False)

    eligible = summary.loc[summary.eligible_at_least_3_pairs]
    lines = [
        "# CMIP6/DAMIP 多成员可用性盘点",
        "",
        "本盘点只证明冻结 ESGF 目录中同时存在 historical 与 hist-GHG 的同名 `zg` 成员，尚不等于时间覆盖和文件读取均已通过。正式下载前还需逐数据集检查 1950--2020 年覆盖、850 hPa 层和网格一致性。",
        "",
        f"- 15 个模式中有 {len(eligible)} 个具备至少 3 对同名成员。",
        f"- 可优先处理：{', '.join(eligible.model)}。",
        "",
        "| 模式 | historical zg | hist-GHG zg | 可配对成员 | >=3 对 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.model} | {row.historical_zg_members} | {row.hist_ghg_zg_members} | "
            f"{row.paired_member_count} | {'是' if row.eligible_at_least_3_pairs else '否'} |"
        )
    lines.extend(
        [
            "",
            "详细的数据集标识符与访问入口见 `data/manifests/cmip6_multimember_zg_datasets.csv`。",
        ]
    )
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Eligible models with >=3 paired members: {len(eligible)}")
    print(eligible[["model", "paired_member_count", "paired_members"]].to_string(index=False))


if __name__ == "__main__":
    main()
