# `scripts/eval/` — 正式评测

指南：[docs/user/eval.md](../../docs/user/eval.md)。环境已经用 [`../setup/`](../setup/README.md) 装好之后，按顺序跑：

```bash
bash scripts/eval/00_smoke.sh              # 00 冒烟
bash scripts/eval/01_eval.sh --dry-run     # 01 先看全量计划
bash scripts/eval/01_eval.sh               # 01 全量（paper_main）
bash scripts/eval/02_report.sh             # 02 一键统计
```

01 可以裁剪；不写就是全量：

```bash
bash scripts/eval/01_eval.sh --backend libero
bash scripts/eval/01_eval.sh --backend libero --level 2 --model pi05
bash scripts/eval/01_eval.sh --level 1,2,3 --model pi05,xvla
DYNI_GPUS=0,1 bash scripts/eval/01_eval.sh --backend simplerenv --resume
```

RoboCasa 不在 `paper_main`：`bash scripts/eval/01_eval.sh --backend robocasa`

TurboVLA 也不在 `paper_main`（独立 suite `turbovla_libero`）：

```bash
bash scripts/eval/serve.sh turbovla --suite safelibero_spatial --gpu 0
bash scripts/eval/01_eval.sh --model turbovla --dry-run
bash scripts/eval/01_eval.sh --suite turbovla_libero --level 0
bash scripts/eval/cell.sh libero turbovla safelibero_spatial 0 --task 0 --episodes 5
```

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

输出（[`configs/paths.yaml`](../../configs/paths.yaml)）：

```text
out/eval/{backend}/{suite}/l{level}/{run_id}/   # JSONL / summary / videos
out/logs/eval/<name>/                           # 过程日志
out/reports/{grid}/                             # 02_report
out/_sweep/{suite}/                             # sweep 失败表
```

整棵树：`DYNI_OUT`（writers 仍认 `DYNI_RESULTS` 为 `out/` 别名）。02 默认读 `out/eval/`；只读旧树用 `--results results` 或 `DYNI_EVAL_ROOT=results`（报告仍写新树，可用 `DYNI_REPORTS`）。
