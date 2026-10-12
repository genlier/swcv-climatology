# SWCV 目录事件数核准

生成日期：2026-09-12 13:46 CST

## 结论

本地 84 个年度文件共有 151,213 行，逐文件以 `step = 0` 定义新事件，共得到 9,513 条轨迹，比论文和 Zenodo 描述中的 9,379 条多 134 条。该差异存在于发布文件内容本身，并非当前解析脚本额外拆分轨迹造成。

已完成的排查显示：

- step 序列或逐小时时间连续性异常轨迹：0 条；
- 完全相同轨迹：0 条；
- 起始时间、纬度和经度完全重复的轨迹：0 条；
- 2025 版与 2026 扩展版共有文件：84 个，其中校验值改变：0 个；
- 2026 扩展版仅新增：2024.txt, 2025.txt。

因此，134 条差异不能归因于文件损坏、重复轨迹、断裂的 step 序列或后续版本修改。现有证据更支持“文章摘要/Zenodo 描述采用了不同但未公开的计数或排除定义”。在获得作者排除清单前，不应任意删除 134 条轨迹。

## 可复核产物

- `data/manifests/zenodo_version_history.json`
- `data/processed/catalogue_zenodo_versions.csv`
- `data/processed/catalogue_reconciliation_by_year.csv`
- `data/processed/catalogue_reconciliation_event_flags.csv.gz`
- `data/processed/catalogue_reconciliation_summary.json`

## 下一步

1. 从原数据论文图表或作者处取得逐年事件数/排除规则，与 `catalogue_reconciliation_by_year.csv` 对齐。
2. 在无法取得规则时，保留 9,513 条主样本，并增加基于可识别边界事件和随机删除 134 条的结论敏感性范围；不得宣称已经恢复论文中的 9,379 条精确样本。
