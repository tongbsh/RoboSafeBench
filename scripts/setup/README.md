# `scripts/setup/` — 预先准备

第一次克隆、换机器、补权重时用。装好之后日常评测走 [`../eval/`](../eval/README.md)。

| 脚本 | 作用 |
|------|------|
| `setup.sh` | submodule + 打补丁 + `pip install -e ".[dev]"` + 审计 + doctor |
| `doctor.sh` | `dynisafety doctor`，可加 `--backend` / `--strict` |
| `audit_envs.sh` | 对照 `configs/models/*.yaml` 查缺哪个 conda env |
| `install_backend.sh` | 四选一装仿真器：`libero\|simplerenv\|robotwin\|robocasa` |
| `install_{libero,simplerenv,robotwin,robocasa}.sh` | 各 backend 的实际安装 |
| `apply_patches.sh` | 把 `patches/` 打到 third_party 树 |
| `download_assets.py` | 下 VLA 权重；不带 MANO 二进制 |
| `repair_editable_installs.py` | 修 editable 安装路径 |

```bash
bash scripts/setup/setup.sh
bash scripts/setup/install_backend.sh libero
bash scripts/setup/doctor.sh --backend libero
python scripts/setup/download_assets.py --list
```

也可用根目录 `make install-libero`（等价 `install_backend.sh libero`）。
