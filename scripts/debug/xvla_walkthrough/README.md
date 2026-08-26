# X-VLA 动态走查（`scripts/eval` 缩小版）

只用 **X-VLA**，把 `paper_main` 里三条动态 backend 走一遍：

| backend | 模型 stem | suite / 任务 | level | 默认轨迹 |
|---------|-----------|--------------|-------|----------|
| LIBERO（dynamic） | `xvla` | 4 suite × task 0–3 | 1–3 | 20 |
| RoboTwin | `xvla_official` | `main20` 全部 20 task | 1–3 | 20 |
| SimplerEnv | `xvla_official` | `bridge` 论文三题 0–2 | 1–3 | 20 |

不含 L0、不含 RoboCasa。默认 **117 job × 20 条轨迹**，**强制录像**，并打开 **L2/L3 diversity** 与 **LIBERO obstacle pack**，用来：

1. 确认 serve → eval → JSONL → 视频 这条链路通；
2. 看 SR / contact / termination / steps 是否像正常 rollout；
3. 看 L1–L3 的动态字段（`spawn_plan` / `l3_plan` / onset）有没有挂上。

```bash
bash scripts/debug/xvla_walkthrough/00_run.sh --dry-run   # 先看 117 job
bash scripts/debug/xvla_walkthrough/00_run.sh             # 跑 + 自动 01_check
bash scripts/debug/xvla_walkthrough/00_run.sh --backend libero
bash scripts/debug/xvla_walkthrough/00_run.sh --episodes 3
bash scripts/debug/xvla_walkthrough/00_run.sh --full      # 论文条数（视频会很多）
DYNI_GPUS=0 bash scripts/debug/xvla_walkthrough/00_run.sh --resume
```

`--full` = LIBERO / SimplerEnv 每格 50 条、RoboTwin 仍 20 条（对齐 `paper_main`）。  
默认开 diversity / obstacle pack；要关：`--no-diversity` / `--no-obstacle-pack`。开跑前会清 8010/8011 和遗留 `xvla_*` screen（`--skip-free` 可跳过）。

检查（也可单独跑）：

```bash
bash scripts/debug/xvla_walkthrough/01_check.sh
```

产物：

| 位置 | 内容 |
|------|------|
| `out/eval/{backend}/{suite}/l{level}/<run_id>/` | `episodes.jsonl` `summary.json` `videos/` |
| `out/_sweep/debug_xvla_dynamic/` | sweep 失败表、实际用的 suite |
| `out/logs/debug/xvla_walkthrough/` | 过程日志、`check.md`、`videos.txt` |

矩阵定义：[`configs/suites/debug_xvla_dynamic.yaml`](../../../configs/suites/debug_xvla_dynamic.yaml)。  
Sweep 自己管 X-VLA serve（不要只跑 `cell.sh` 却忘了起服务）。  
`xvla` / `xvla_official` 都是 `max_jobs_per_gpu: 2`。走查默认 `DYNI_SERVERS_PER_GPU=1`：单卡一份权重 + 两个 eval 共用。要每槽一份权重时设 `DYNI_SERVERS_PER_GPU=2`。矩阵是 117 job（三 backend × 全部 paper 任务 × L1–L3），调度器按双槽排队，不是只跑一个 task。
