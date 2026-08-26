# `scripts/`

权威入口仍是 CLI：`dynisafety eval` / `sweep` / `report` / `analysis`。  
这里的 shell 只负责切 conda、拼参数。按用途分成四类。

输出约定见 [`configs/paths.yaml`](../configs/paths.yaml)：一个根 `out/`（`DYNI_OUT`），评测在 `eval/{backend}/{suite}/l{level}/`。  
覆盖整棵树：`DYNI_OUT` 或 `--output-root`。只读旧 `results/` 做报告时设 `DYNI_RESULTS=results`。

```text
scripts/
├── lib/env.sh     # 公共库（被 source，不要直接跑）
├── setup/         # 预先准备：装环境、打补丁、下权重
├── eval/          # 正式评测：冒烟 → 全量/裁剪 → 统计 → 分析
├── debug/         # 自己调参（非正式矩阵）
└── extra/         # 拓展实验：可行性 / 模型包装 / GSO / golden
```

环境装好后的日常路径：

```bash
bash scripts/eval/00_smoke.sh
bash scripts/eval/01_eval.sh --dry-run
bash scripts/eval/01_eval.sh                 # 或加 --backend / --level / --model
bash scripts/eval/02_report.sh
bash scripts/eval/03_analysis.sh
```

| 目录 | 给谁 | 入口 |
|------|------|------|
| [setup/](setup/README.md) | 第一次装机器 | `make install-libero` 或 `bash scripts/setup/setup.sh` |
| [eval/](eval/README.md) | 正式评测 | `bash scripts/eval/00_smoke.sh` |
| [debug/](debug/) | 自己调参 | `bash scripts/debug/eval_debug.sh --help`；X-VLA 走查 `bash scripts/debug/xvla_walkthrough/00_run.sh --dry-run` |
| [extra/](extra/README.md) | Oracle / AEGIS / 旁路 | `bash scripts/extra/feasibility/run_authorized_aegis_feasibility.sh` |

装环境（四选一，extras 互斥）：

```bash
make install-libero          # = bash scripts/setup/install_backend.sh libero
make install-simplerenv
make install-robotwin
make install-robocasa        # NumPy 2，env 名 dynisafety-robocasa
bash scripts/setup/doctor.sh --backend libero
python scripts/setup/download_assets.py --help
```
