# `scripts/eval/` — 正式评测

环境已经用 [`../setup/`](../setup/README.md) 装好之后，按顺序跑：

```bash
bash scripts/eval/00_smoke.sh              # 00 冒烟
bash scripts/eval/01_eval.sh --dry-run     # 01 先看全量计划
bash scripts/eval/01_eval.sh               # 01 全量（paper_main）
bash scripts/eval/02_report.sh             # 02 一键统计
bash scripts/eval/03_analysis.sh           # 03 一键分析
```

01 可以裁剪；不写就是全量：

```bash
bash scripts/eval/01_eval.sh --backend libero
bash scripts/eval/01_eval.sh --backend libero --level 2 --model pi05
bash scripts/eval/01_eval.sh --level 1,2,3 --model pi05,xvla
DYNI_GPUS=0,1 bash scripts/eval/01_eval.sh --backend simplerenv --resume
```

RoboCasa 不在 `paper_main`：`bash scripts/eval/01_eval.sh --backend robocasa`

## 进阶（同一目录）

| 脚本 | 作用 |
|------|------|
| `cell.sh` | 单格：`<backend> <model> <suite> <level>` |
| `serve.sh` | 起服务型模型（另开终端） |
| `sweep.sh` | 直接调 `dynisafety sweep`（01 的底层） |

```bash
bash scripts/eval/cell.sh simplerenv random bridge 2 --task 0 --episodes 1
bash scripts/eval/serve.sh starvla --gpu 0
bash scripts/eval/sweep.sh --only backend=libero,level=2 --dry-run
```

调参不要走这里，用 [`../debug/`](../debug/)。
