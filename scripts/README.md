# `scripts/`

权威入口仍是 CLI：`dynisafety eval` / `sweep` / `report`。  
这里的 shell 只负责切 conda、拼参数。

输出约定见 [`configs/paths.yaml`](../configs/paths.yaml)：一个根 `out/`（`DYNI_OUT`），评测在 `eval/{backend}/{suite}/l{level}/`。  
覆盖整棵树：`DYNI_OUT`。只读旧 `results/` 做报告时用 `--results results`（或 `DYNI_EVAL_ROOT=results`）。

```text
scripts/
├── lib/env.sh     # 公共库（被 source，不要直接跑）
├── setup/         # 预先准备：装环境、打补丁、下权重
├── eval/          # 正式评测：冒烟 → 全量/裁剪 → 统计
└── extra/         # 模型 serve 包装、可选 GSO overlay
```

环境装好后的日常路径：

```bash
bash scripts/eval/00_smoke.sh
bash scripts/eval/01_eval.sh --dry-run
bash scripts/eval/01_eval.sh                 # 或加 --backend / --level / --model
bash scripts/eval/02_report.sh
```

| 目录 | 给谁 | 入口 |
|------|------|------|
| [setup/](setup/README.md) | 第一次装机器 | `make install-libero` 或 `bash scripts/setup/setup.sh` |
| [eval/](eval/README.md) | 正式评测 | `bash scripts/eval/00_smoke.sh` |
| [extra/](extra/README.md) | LingBot-VA serve、GSO overlay | `scripts/extra/README.md` |

装环境（四选一，extras 互斥）。先 `conda activate` **YAML 里的 eval 名**
（`openvla-oft` / `simpler_env` / `internvla_a1` / `dynisafety-robocasa`），
`make install-*` **不会**新建 conda。可选一键（只建**缺失**的 eval env，已有的跳过）：
`bash scripts/setup/install_conda_env.sh`。指南：[`docs/user/README.md`](../docs/user/README.md)、[`docs/user/install.md`](../docs/user/install.md)。
**不要重命名**已经能跑的 env。

```bash
conda activate openvla-oft
make install-libero          # = bash scripts/setup/install_backend.sh libero
# make install-simplerenv    # activate simpler_env first
# make install-robotwin      # activate internvla_a1 first（不是 RoboTwin）
# make install-robocasa      # NumPy 2，env 名 dynisafety-robocasa
bash scripts/setup/audit_envs.sh
python scripts/setup/download_assets.py --help
python scripts/setup/download_assets.py --paper --dry-run
```
