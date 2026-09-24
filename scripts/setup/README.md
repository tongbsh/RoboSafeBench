# `scripts/setup/` — 预先准备

第一次克隆、换机器、补权重时用。装好之后日常评测走 [`../eval/`](../eval/README.md)。

conda **名字**以 `configs/models/*.yaml` 为准（论文机是 `openvla-oft` /
`simpler_env` / `internvla_a1`）。总表：[`docs/user/install.md`](../../docs/user/install.md)。
`install_*.sh` **不创建** conda；不要重命名已经能跑的 env。

| 脚本 | 作用 |
|------|------|
| `setup.sh` | submodule + 打补丁 + `pip install -e ".[dev]"` + 审计 + doctor |
| `doctor.sh` | `dynisafety doctor`，可加 `--backend` / `--strict` |
| `audit_envs.sh` | 对照 `configs/models/*.yaml` 查缺哪个 conda env |
| `install_conda_env.sh` | **可选**：选 backend + VLA，只给**缺失**的 eval env 按短配方新建；已有 env 不改。serve 不装 |
| `install_backend.sh` | 四选一装仿真器：`libero\|simplerenv\|robotwin\|robocasa`（假定 env 已 activate） |
| `install_{libero,simplerenv,robotwin,robocasa}.sh` | 各 backend 的实际安装 |
| `apply_patches.sh` | 把 `patches/` 打到 third_party 树 |
| `download_assets.py` | 下 VLA 权重；不带 MANO 二进制 |
| `repair_editable_installs.py` | 修 editable 安装路径 |

```bash
bash scripts/setup/setup.sh
bash scripts/setup/install_backend.sh libero
bash scripts/setup/doctor.sh --backend libero
python scripts/setup/download_assets.py --list
python scripts/setup/download_assets.py --paper --dry-run
python scripts/setup/download_assets.py --model xvla_official
# 可选一键（新机器缺 env 时；已有的会跳过）：
bash scripts/setup/install_conda_env.sh --list
bash scripts/setup/install_conda_env.sh --backend libero --model openvla_oft --dry-run
bash scripts/setup/install_conda_env.sh          # 交互选 backend / model
```

也可用根目录 `make install-libero`（等价 `install_backend.sh libero`）。
