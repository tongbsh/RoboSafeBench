# `scripts/extra/` — 拓展实验

不是论文主路径 `paper_main` 的日常评测。正式表走 [`../eval/`](../eval/README.md)。

| 路径 | 作用 |
|------|------|
| `feasibility/` | Oracle Stop-Wait-Resume / AEGIS 可行性矩阵。默认拒 GPU，需 `AUTHORIZE_GPU=1`。π₀.₅ 上帝视角 SWR + sim_gt AEGIS 一键入口：`run_pi05_oracle_swr_aegis.sh` |
| `models/lingbot_va/` | LingBot-VA serve 包装（`configs/models/lingbot_va.yaml` 会调到这里） |
| `libero/` | 从 GSO 生成 LIBERO overlay（默认关）；`resettle_spatial_ramekin_init.py` 只重写 spatial task 1 的落座 init |
| `verify_golden.sh` | 重构前后 golden 对照，不是复现论文数字 |

```bash
# π₀.₅ × DynamicLIBERO：上帝视角 SWR + AEGIS(sim_gt)，自动起 serve
# 评测 JSONL 与 scripts/eval 同树：out/eval/{backend}/{suite}/l{level}/
# 过程日志在 out/logs/feasibility/；CAPTURE=1 才写 out/capture/（robosafe_capture_1.0）
bash scripts/extra/feasibility/run_pi05_oracle_swr_aegis.sh
AUTHORIZE_GPU=1 bash scripts/extra/feasibility/run_pi05_oracle_swr_aegis.sh
AUTHORIZE_GPU=1 CAPTURE=1 bash scripts/extra/feasibility/run_pi05_oracle_swr_aegis.sh

# OpenVLA-OFT 小矩阵（不起 π₀.₅ serve）
bash scripts/extra/feasibility/run_authorized_aegis_feasibility.sh
AUTHORIZE_GPU=1 bash scripts/extra/feasibility/run_authorized_aegis_feasibility.sh
bash scripts/extra/models/lingbot_va/serve.sh <port> <checkpoint_dir>
```

过程日志进 `out/logs/feasibility/`；评测 JSONL 和视频进 `out/eval/`（`DYNI_OUT` 改整棵树；读旧 `results/` 做 02/03 用 `--results results`）。

设计说明：`docs/paper/experiment_upgrade/`。
