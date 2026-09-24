# `scripts/extra/` — 可选包装

正式评测走 [`../eval/`](../eval/README.md)。这里只放评测需要、但不在 `eval/` 主路径上的薄包装。

| 路径 | 作用 |
|------|------|
| `models/lingbot_va/` | LingBot-VA serve 包装（`configs/models/lingbot_va.yaml` 会调到这里） |
| `libero/materialize_obstacle_pack_gso.py` | 从 GSO 生成 LIBERO overlay（默认关） |

```bash
bash scripts/extra/models/lingbot_va/serve.sh <port> <checkpoint_dir>
```
