# `scripts/debug/` — 自己调参

非正式矩阵。评测进 `out/eval/{backend}/{suite}/l{level}/`，过程日志进 `out/logs/debug/`（见 [`configs/paths.yaml`](../../configs/paths.yaml)）。正式表走 [`../eval/`](../eval/README.md)。

```bash
bash scripts/debug/eval_debug.sh --help
bash scripts/debug/eval_debug.sh --backend libero --model pi05 --level 1 --episodes 1 --dry-run

# 服务型：一个终端 serve，另一个 eval
bash scripts/debug/launch_vla_serve.sh --model pi05 --port 8011
bash scripts/debug/launch_vla_eval.sh --model pi05 --backend libero --level 1 --task 0 --episodes 1
```

按 backend 的薄包装在 `libero/` `robotwin/` `simplerenv/`。

**X-VLA 全动态走查**（paper_main 任务 × L1–L3 × 录像，只跑 X-VLA）：

```bash
bash scripts/debug/xvla_walkthrough/00_run.sh --dry-run
bash scripts/debug/xvla_walkthrough/00_run.sh
```

见 [`xvla_walkthrough/`](xvla_walkthrough/)。
