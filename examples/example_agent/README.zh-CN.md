# 具身智能体接入示例

[English](README.md)

把 VLA / WAM 接到 DyniSafety 的操作说明。本目录是可运行示例，不是第二套契约。

规范流程、冻结契约和安装规则在仓库文档里，这里不重复：

- [docs/user/adding_a_model.md](../../docs/user/adding_a_model.md) — 6 步流程
- [docs/user/contracts.md](../../docs/user/contracts.md) — 冻结契约
- [docs/user/installation.md](../../docs/user/installation.md) — 每个环境只装一个 backend；extras 互斥
- [docs/user/benchmark_design.md](../../docs/user/benchmark_design.md)、[docs/user/metrics.md](../../docs/user/metrics.md)

## 这是什么（以及不是什么）

`MockBackbone` 是确定性的玩具投影器，**不是**真实策略。它产出的数字**绝不能**当作 DyniSafety 基准结果汇报。用这棵树检查适配器、YAML 和传输接线是否符合 harness，然后再把玩具权重换成你的模型。

## 5 分钟跑通

在仓库根目录，当前环境已安装 DyniSafety：

```bash
pip install -e examples/example_agent
# 或：make example-agent-install
python examples/example_agent/verify.py
```

期望表格（若 `PATH` 上没有 `dynisafety`，CLI 行会显示 SKIP）：

```
check                                result
-----------------------------------  ------
offline libero                       PASS
offline simplerenv                   PASS
offline robotwin                     PASS
chunk modes                          PASS
http round-trip                      PASS
eval libero --dry-run                PASS
eval simplerenv --dry-run            PASS
eval robotwin --dry-run              PASS
sweep smoke_example_agent --dry-run  PASS
```

## 两文件规则

| 部分 | 可以放在哪 | 原因 |
|------|------------|------|
| 适配器类 | 任何可 import 的位置 | YAML 的 `adapter:` 是导入路径。本包装**不**在 `src/` 下。 |
| 模型 YAML | 只能是 `configs/models/<stem>.yaml` | 注册表扫描该目录。注册键**就是**文件名 stem。`--model` 不接受路径。 |

YAML 里的 `name:` **必须等于 stem**。`create_policy` 按 `cfg.name` 查找；不一致会 `KeyError`。

本示例：`configs/models/example_agent.yaml` →
`adapter: dyni_example_agent.local_adapter:ExampleAgentAdapter`。

## 契约面

`Observation` 是 dataclass，不是原始 env dict。

| 字段 | 类型 | 说明 |
|------|------|------|
| `images` | `dict[str, np.ndarray]` | 键**只能**是 `primary` / `wrist` / `left` / `right` |
| `state` | `np.ndarray \| None` | 当 `observation.needs_state: true` 时必填 |
| `instruction` | `str` | 任务语言 |
| `step` | `int` | episode 步数 |
| `extra` | `dict` | backend 附加字段 |

`act()` 返回 `ActionChunk(actions, space)`：

- `actions` 为 `(C, D)` 的 `float32`
- `space` 必须等于 `cfg.action_space`
- `D` 必须等于 `dim_for_space(space)`

| `action_space` | `D` |
|----------------|-----|
| `delta_ee_7d` | 7 |
| `abs_ee_7d` | 7 |
| `abs_ee_16d` | 16 |
| `qpos_14d` | 14 |
| `hybrid_mobile_12d` | 12 |

动作必须已经是**环境单位**。harness 的 `convert_action` 只做 delta↔abs EE 转换，**不会**做数据集 q01/q99 反归一化。

## Action chunk 语义

`ActionChunkQueue.refill` 只保留 `min(YAML chunk_size, returned_rows)`，**其余丢弃**。`chunk_size: 8` 时若返回 16 行，只会执行 8 行，另外 8 行永远不会跑。

本示例用一个 `deque` 同时实现两种模式（本地与 HTTP）：

| YAML | `act()` 返回 | harness 行为 | 对标 |
|------|--------------|--------------|------|
| `replan_steps == chunk_size` | `(chunk_size, D)` | 开环整块执行 | `openvla_oft`、`starvla` |
| `replan_steps == 1` | 每次 `(1, D)` | 每 `chunk_size` 次调用才重新规划 | `internvla_a1`、`fastwam` |

第二种模式下，YAML 的 `chunk_size` **只是**内部规划地平线，**不会**让 harness 开环跑完整块。

HTTP 适配器一次请求 `horizon=cfg.chunk_size`，再从本地缓冲每次最多弹出 `replan_steps` 行——与远程 WAM 客户端的组合方式相同。

## 各 backend 的观测

| Backend | `action_space` | `D` | 实际送达的图像 | 典型 `state` |
|---------|----------------|-----|----------------|--------------|
| `libero` | `delta_ee_7d` | 7 | `primary`、`wrist` | 8 维 |
| `simplerenv` | `delta_ee_7d` | 7 | **只有 `primary`** | 7 维 |
| `robotwin` | `qpos_14d` | 14 | `primary`、`left`、`right` | 14 维 |

SimplerEnv backend 的 `bind_run` 会丢弃 `observation_images`。必须声明
`per_backend.simplerenv.observation.images: [primary]`，否则
`require_image(obs, "wrist")` 会直接报错。

RoboCasa 强制 `hybrid_mobile_12d`，本示例不覆盖。参见
`configs/models/random_mobile.yaml` 和 `configs/models/xiaomi.yaml`。

## `per_backend` 覆盖层

允许的键只有：`action_space`、`chunk_size`、`observation`、`extra`。其它键会报错。

- `observation` 和 `extra` 是**浅合并**（覆盖层优先）。
- `action_space` 和 `chunk_size` 是**整段替换**。

本示例顶层默认面向 LIBERO。SimplerEnv 与 RoboTwin 的覆盖写在 `per_backend` 下。

## 远程 / HTTP 路径

本示例用 HTTP 而不是 WebSocket：`build_transport` 的 websocket 路径要求
`extra.ws_codec` ∈ `{openpi, starvla, lingbot, xiaomi}`，这些都是
`src/dynisafety/policies/transports/codecs/` 里的模型专用编解码。通用玩具
websocket 服务端做不到。

`configs/models/example_agent_http.yaml` 需要完整的 `server:` 块
（`env`、`root`、`launch`、`default_port`、`health_path`、`startup_timeout_s`），
以及 `extra.act_timeout_s`、`extra.reset_path`、`extra.act_path`、
`extra.server_host`。`connect_transport` → `_build_http` 会在
`wait_until_ready()` 里**阻塞**，轮询 `/health`，直到健康或
`server.startup_timeout_s` 耗尽。调用 `from_config` 之前，进程必须已经在听。

Harness 路径（调度器 / `dynisafety serve` 会注入 host 和 port；适配器**禁止**读 `os.getenv("PORT")`）：

```bash
dynisafety serve --model example_agent_http
# 若发现不到 GPU，加 --gpu N；需要固定端口时加 --port
```

手动路径：

```bash
python examples/example_agent/server/toy_vla_server.py \
  --port 8090 \
  --checkpoint examples/example_agent/assets/mock_vla_weights.npz
```

然后让客户端指向该套接字（这是一次 live eval，不是 `--dry-run`）：

```bash
dynisafety eval \
  --backend libero \
  --model example_agent_http \
  --suite safelibero_spatial \
  --level 0 \
  --task 0 \
  --set extra.server_host=127.0.0.1 \
  --set extra.server_port=8090
```

sweep 会把 `HOST` / `PORT` 注入为 `extra.server_host` / `extra.server_port`。

## 验证阶梯

在仓库根目录：

```bash
ruff check examples tests/test_example_agent.py
python -m pytest tests/test_example_agent.py -q
python examples/example_agent/verify.py

dynisafety eval --backend libero --model example_agent --suite safelibero_spatial --level 0 --task 0 --dry-run
dynisafety eval --backend simplerenv --model example_agent --suite bridge --level 0 --task 0 --dry-run
dynisafety eval --backend robotwin --model example_agent --suite main20 --level 0 --task 0 --dry-run
dynisafety sweep --suite smoke_example_agent --dry-run
```

`dynisafety models` 是**桩实现**（`not implemented yet`）。查看注册状态请用 `available()`：

```bash
python -c "from pathlib import Path; from dynisafety.policies.registry import available; [print(s.name, s.state, s.detail) for s in available(configs_root=Path('configs')) if s.name.startswith('example_agent')]"
```

两阶段信号（与 `random` 以及所有树内模型相同）：

| 时机 | `state` | 原因 |
|------|---------|------|
| 尚未 `pip install -e examples/example_agent`（适配器不可导入） | `missing_dependency` | YAML 已声明，但类 import 失败 |
| 安装之后 | `missing_checkpoint` | 目录列举不会展开 `${workspace}` |

安装之后出现 `missing_checkpoint` 是**预期行为**，不是错误。

## 常见错误

- 在 Python 里写死端口、超时或 chunk size。它们只在 YAML 里出现一次；适配器用 `require_extra` 读取，缺失必须**抛错**。
- 直接调用 `requests.post` / `websockets.connect` / `zmq.Context`，或读取 `os.getenv("PORT")`。远程 I/O 必须走 `connect_transport(cfg)`。
- 返回一维数组。必须返回 `(C, D)`。
- 返回的 `space` 与 `cfg.action_space` 不一致。
- 指望 harness 做 q01/q99 反归一化。它不会。返回环境单位。
- `name:` 与 YAML 文件名 stem 不一致。
- 增加第五个图像键（只允许 `primary` / `wrist` / `left` / `right`）。
- 以为 `dynisafety models` 会列出目录。请用上面的 `available()`。

## 文件地图

| 路径 | 作用 |
|------|------|
| `src/dyni_example_agent/local_adapter.py` | `transport: local` Policy |
| `src/dyni_example_agent/http_adapter.py` | `transport: http` Policy |
| `src/dyni_example_agent/mock_backbone.py` | 玩具投影器（不是真实 VLA） |
| `server/toy_vla_server.py` | 标准库 HTTP `/health` `/reset` `/act` |
| `assets/make_mock_weights.py` | 生成提交用的 `.npz`（`.pt` 已被 gitignore） |
| `assets/mock_vla_weights.npz` | 玩具权重 |
| `verify.py` | 一次性 PASS/FAIL 表 |
| `../../configs/models/example_agent.yaml` | 本地模型（stem = `name`） |
| `../../configs/models/example_agent_http.yaml` | HTTP 模型 |
| `../../configs/suites/smoke_example_agent.yaml` | smoke sweep 单元 |
| `../../tests/test_example_agent.py` | CI 测试（无仿真、无 GPU） |
