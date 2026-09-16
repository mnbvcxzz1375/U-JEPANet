# Results

本目录存放可复现的实验产物摘要与分析（原始大文件仍留在服务器路径，此处只提交轻量 JSON/Markdown）。

## 约定

- 每次正式/筛选运行：`results/<run_id>/` 下至少包含
  - `summary.json`（或从服务器拉取的等价物）
  - `ANALYSIS.md`（结论、健康检查、限制）
  - `provenance.txt`（host / commit / env / GPU）
- 大 checkpoint、全量日志默认不进 Git；在 `ANALYSIS.md` 写明远端路径
- 任何“结果”必须能对应到 commit SHA 与数据协议

## Runs

| run_id | 说明 | 代码 |
|---|---|---|
| [pilot_3k_20260916_185115](pilot_3k_20260916_185115/) | A0–A3 3k 健康 pilot（40901 GPU1 / vllmenv） | `ab57558` + A1 None-fix |
