# 权重与资产

模型 checkpoint、下载脚本、MANO。不要把权重、视频、API key、MANO 二进制提交进 git。`/assets/` 已 gitignore。conda 见 [install.md](install.md)。

路径占位只允许 `${workspace}`、`${third_party}`、`${env:VAR}`，不要把本机绝对路径写进要提交的 YAML。

---

## 下载入口

```bash
python scripts/setup/download_assets.py --help
python scripts/setup/download_assets.py --list
python scripts/setup/download_assets.py --list --paper
python scripts/setup/download_assets.py --paper --dry-run
python scripts/setup/download_assets.py --model xvla_official
python scripts/setup/download_assets.py --model starvla --suite main20
python scripts/setup/download_assets.py --asset internvla-m1-bridge
python scripts/setup/download_assets.py --mano-notice
```

| 参数 | 作用 |
|------|------|
| `--list` | 打印目录（可再加 `--paper` / `--model` / `--suite` 过滤） |
| `--paper` | 按 `configs/suites/paper_main.yaml` 下齐论文权重 |
| `--model STEM` | 按 `configs/models/<stem>.yaml` 下该模型全部 catalog 项（可重复） |
| `--suite SUITE` | 只留下覆盖该 suite 的项（如 `main20`、`bridge`、`safelibero_spatial`） |
| `--asset NAME` | 按目录名下（可重复；`--name` 是别名） |
| `--dest DIR` | 覆盖落盘根 |
| `--dry-run` | 只打印计划，不联网 |

落盘顺序（代码为准）：

1. `--dest DIR`
2. 环境变量 `$DYNISAFETY_CHECKPOINTS`
3. `<workspace>/external/checkpoints`

本地目录必须和 `configs/models/<stem>.yaml` 的 `checkpoint.default` / `per_suite` 对得上。`paper_main` 每个 stem × suite 都有 catalog 项。远程模型还要 `dynisafety serve`（或上游 server）。

根目录旧 `download_hf.py` 已弃用，一律用 `scripts/setup/download_assets.py`。

需要 `huggingface_hub`。镜像可设 `HF_ENDPOINT`（如 `https://hf-mirror.com`）。π₀ / π₀.₅ 走 `gs://openpi-assets`，要 `gsutil` 或 `gcloud storage`，或在 openpi env 里用官方 `maybe_download`（`OPENPI_DATA_HOME=<dest>/openpi_cache`）。

---

## `paper_main` 覆盖

权威矩阵是 `configs/suites/paper_main.yaml`。`--list` 是目录原文。下表只写 stem → 落盘（相对 dest 根）：

| stem | 来源 | 相对 dest |
|------|------|-----------|
| `openvla_oft` | HF `moojink/openvla-7b-oft-finetuned-libero-{spatial,object,goal,10}` | 本地镜像 `openvla-oft/…`；YAML 仍写 HF repo id，运行时走 transformers |
| `xvla` | HF `2toINF/X-VLA-Libero` | `xVLA/xvla-libero-full` |
| `starvla` | HF `StarVLA/Qwen3-VL-OFT-LIBERO-4in1`、`…-RoboTwin2-All`、`Qwen-GR00T-Bridge` | `starVLA/starvla-libero-4in1`、`starvla-robotwin2-all`、`Qwen-GR00T-Bridge` |
| `pi0` / `pi05` | `gs://openpi-assets/checkpoints/pi0_libero`、`pi05_libero` | `openpi_cache/openpi-assets/checkpoints/…`（π₀.₅ YAML 指向嵌套的 `…/pi05_libero/pi05_libero`） |
| `openvla_oft_rlinf` | HF `RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-{spatial,object,goal,long}` | `openvla-oft_RLinf/…` |
| `internvla_a1` | HF `InternRobotics/InternVLA-A1-3B-RoboTwin` | `InternVLA/InternVLA-A1-3B-RoboTwin` |
| `fastwam` | HF `yuanty/fastwam` | `fastwam/robotwin`（`.pt` + stats） |
| `lingbot_va` | HF `robbyant/lingbot-va-posttrain-robotwin` | `lingbot-va/lingbot-va-posttrain-robotwin` |
| `internvla_m1` | HF `InternRobotics/InternVLA-M1-Pretrain-RT-1-Bridge` | `InternVLA/InternVLA-M1-Pretrain-RT-1-Bridge`（须保留 `config.yaml` + `dataset_statistics.json`） |
| `xvla_official` | HF `2toINF/X-VLA-RoboTwin2`、`2toINF/X-VLA-WidowX` | `xVLA/X-VLA-RoboTwin2`、`xVLA/xvla-simpler-widowx` |

`univla` 不在 `paper_main`：`python scripts/setup/download_assets.py --model univla`。LIBERO 落盘 `univla/sft-libero/univla-libero-{spatial,object,goal,10}`；SimplerEnv Bridge（`univla/sft-simpler-bridge`）同样只走 `--model univla` 或 `--asset univla-simpler-bridge`。`lingbot-va-base`、`fastwam-robotwin2-dataset`、`bridgevla-*` 同样只走 `--asset`。

`turbovla` 也不在 `paper_main`：`python scripts/setup/download_assets.py --model turbovla`。落盘相对 dest：`turbovla/checkpoints/libero/turbovla_libero.pth`（四个 LIBERO suite 共用）、`turbovla/libero_all4_stats.json`、`turbovla/backbones/dinov3-vitb16`、`turbovla/backbones/bert-base-uncased`。权重含 DINOv3 衍生参数。

`vla_adapter` 同样不在 `paper_main`：`python scripts/setup/download_assets.py --model vla_adapter`。落盘相对 dest：`vla_adapter/LIBERO-{Spatial,Object,Goal,Long}-Pro`（HF `VLA-Adapter/LIBERO-*-Pro`，公开，无 DINOv3）。Long 的 unnorm_key 是 `libero_10_no_noops`，不是 `libero_long`。源码克隆到 `external/model_repos/VLA-Adapter`。

个别 HF 仓库名和 YAML 目录名不同（X-VLA LIBERO / WidowX、StarVLA LIBERO / RoboTwin），脚本按 **YAML 路径** 落盘。

---

## MANO / SMPL+H（L3 手）

**不随仓库分发。** 接受 MPI/MPG 条款后，自行放到（gitignore）：

```text
<workspace>/assets/mano_v1_2/
```

```bash
python scripts/setup/download_assets.py --mano-notice
```

说明见 [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) § MANO。RoboTwin L3 当前用仓库自带的小 hand mesh，不是 MANO；只有你显式要 MANO 实验时才需要。

---

## 仿真器资产（不是 VLA 权重）

| 资产 | 怎么来 |
|------|--------|
| RoboTwin ~16GB 场景 | 不在 submodule 里；按上游 / [install.md](install.md) 放到 backend 期望路径 |
| RoboCasa 厨房（默认轻量） | `scripts/setup/install_robocasa.sh`；缺 `tex` / `fixtures_lw` 等则 `doctor` FAIL |
| 全量 RoboCasa ~10GB | 仅当 `DYNISAFETY_ROBOCASA_FULL_ASSETS=1` |
