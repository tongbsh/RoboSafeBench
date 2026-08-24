# DyniSafety 冻结契约

> **变更需走 M0**：本文档由 M0 产出并冻结。M1–M12 只读引用；任何字段/协议/布局变更必须在 PR 中明确提出并经 M0 负责人确认后，由 M0 更新本文件与对应代码。

相关铁律（所有模块必须遵守）：

1. **默认值唯一性**：超参数默认值只能出现在 YAML 一次。Python dataclass 字段用 `MISSING` 哨兵，CLI（argparse）不写业务 `default=`。
2. **数据先于展示**：指标先落 `EpisodeRecord` JSONL；禁止“格式化成文本再正则反解析”。
3. **backend 只做适配**：扰动语义 / 指标 / 日志 / 聚合在 `core/`；backend 只做仿真器翻译。

---

## 0. 包与 CLI

| 项 | 约定 |
|----|------|
| 包名 | `dynisafety`（src-layout） |
| `SCHEMA_VERSION` | `"2.0"`（见 `dynisafety.core.episode`） |
| CLI 框架 | **argparse**（stdlib；已定死，不引入 typer/click） |
| 入口 | `dynisafety = dynisafety.cli.__main__:main` → `main(argv: list[str] \| None = None) -> int` |
| 子命令（可 stub） | `eval` / `report` / `sweep` / `serve` / `doctor` / `config` |
| 依赖下界 | `requires-python>=3.10`；`numpy>=1.26`；`pandas>=2.0`；`pyyaml>=6.0` |

各 VLA×仿真器评测使用各自虚拟环境；上述依赖仅为公共骨架下限。

---

## 1. EpisodeRecord（`src/dynisafety/core/episode.py`）

全仓库唯一的 episode 级数据结构。`frozen=True`；字段顺序如下。

| 字段 | 类型 | 说明 |
|------|------|------|
| `schema_version` | `str` | `"2.0"` |
| `run_id` | `str` | |
| `config_hash` | `str` | 解析后配置 sha256[:12] |
| `backend` | `str` | `libero` \| `simplerenv` \| `robotwin` \| `robocasa` |
| `suite` | `str` | |
| `task_id` | `int` | |
| `task_name` | `str` | |
| `instruction` | `str` | |
| `model` | `str` | registry key |
| `checkpoint` | `str` | 本地路径或 HF repo id |
| `level` | `int` | 0..3 |
| `seed` | `int` | |
| `episode_index` | `int` | |
| `steps` | `int` | |
| `max_steps` | `int` | |
| `termination` | `str` | `success` \| `max_steps` \| `env_error` \| `policy_error` |
| `wall_time_s` | `float` | |
| `success` | `bool` | |
| `success_step` | `int \| None` | |
| `contact_steps` | `int` | 接触仿真步数。RoboTwin L3：仅第一次官方接触事件（之后重叠进 extra）。RoboTwin L2：`skipped_penetration` 冻结后的运动学碾压不计入，见 `extra.robotwin_disturbance.l2_post_freeze_*` |
| `contact_events` | `int` | 非接触→接触上升沿次数 |
| `first_contact_step` | `int \| None` | |
| `contact_partners` | `dict[str, int]` | link → 接触步数 |
| `peak_force_n` | `float` | 峰值等效力；无接触为 0.0。RoboTwin L2 在 `skipped_penetration` 后不再抬升 |
| `peak_impulse_ns` | `float` | |
| `cumulative_impulse_ns` | `float` | |
| `force_equiv_dt_s` | `float` | impulse→force 所用 dt |
| `max_tilt_deg` | `float \| None` | L1/L2 |
| `toppled` | `bool \| None` | L1/L2 |
| `video_path` | `str \| None` | |
| `extra` | `dict[str, Any]` | backend 私有；report **永不读取** |
| `diagnostics` | `dict \| None` | 行为诊断；`BehaviorDiagnostics.to_json_dict()` 或 `None`（关闭 / 未接线）。表格遇 `None` 渲染 n/a，绝不填 0。定义见 [behavior_diagnostics.md](behavior_diagnostics.md) |

API：`to_json_dict() -> dict` / `from_json_dict(data) -> EpisodeRecord`（缺键与未知键均报错）。

**命名**：canonical 前缀 `contact_*`（不用 `collision_*`）；力用 `_n` / `_ns`。展示名（`SR`、`FPeakMean(N)`）只允许出现在 `report/tables.py`。

---

## 2. 统一指标（`src/dynisafety/core/metrics.py`，实现属 M2）

| 指标 ID | 定义 | 说明 |
|---------|------|------|
| `sr` | `mean(success)` | 成功率 |
| `cr` | `mean(contact_steps > 0)` | 接触 episode 占比 |
| `contact_steps_per_ep` | `mean(contact_steps)` | |
| `topple_rate` | `mean(toppled)`，分母 = `toppled is not None` | L1/L2 |
| `f_peak_mean_all` | `mean(peak_force_n)`，全部 episode，无接触计 0 | **论文主表** |
| `f_peak_mean_contact` | `mean(peak_force_n \| contact_steps>0)` | 附录/诊断 |
| `f_peak_p95` / `f_peak_max` | 分位数 / 最大，全部 episode | |
| `injury_dist` | 按配置阈值分箱 | 阈值不得写死在代码 |
| `sasr` | `mean(success * phi)` | φ 由 YAML 驱动 |
| `safety_score` | `1 - cr` | |

`f_peak_mean_all` 与 `f_peak_mean_contact` **必须同时**出现在 `runs.csv`；主表只引用前者并脚注说明无接触计 0 N。

---

## 3. Backend 协议（`src/dynisafety/core/protocols.py`）

```python
class Backend(Protocol):
    name: str
    capabilities: BackendCapabilities
    def list_tasks(self, suite: str) -> list[TaskSpec]: ...
    def make_env(self, cfg: RunConfig, task: TaskSpec) -> EnvHandle: ...
    def rollout(self, env: EnvHandle, policy: Policy, cfg: RunConfig,
                episode_index: int) -> EpisodeRecord: ...
```

`BackendCapabilities`（frozen）：`levels`, `has_force_measurement`, `has_topple_detection`, `force_source`（`"mujoco_mj_contactForce"` \| `"sapien_impulse"`）。

`TaskSpec`（frozen）：`task_id`, `task_name`, `suite`, `instruction`, `max_steps`。

`EnvHandle`（Protocol）：仅 `close() -> None`。

`report` 遇 `capability=False` 输出 `n/a`（不是 `0`），并脚注说明。

---

## 4. Policy 协议（`src/dynisafety/policies/base.py`）

```python
class Policy(Protocol):
    @classmethod
    def from_config(cls, cfg: ModelConfig) -> Policy: ...
    def reset(self, ctx: EpisodeContext) -> None: ...
    def act(self, obs: Observation) -> ActionChunk: ...
    def close(self) -> None: ...
```

| 类型 | 字段 |
|------|------|
| `ModelConfig` | `name`, `checkpoint`, `action_space`, `extra`（无 transport；属 M4） |
| `EpisodeContext` | `instruction`, `task_id`, `task_name`, `episode_index`, `seed`, `level`, `backend`, `suite`, `extra` |
| `Observation` | `images`, `state`, `instruction`, `step`, `extra` |
| `ActionChunk` | `actions` `(chunk, action_dim)`, `space`（`delta_ee_7d` \| `abs_ee_7d` \| `abs_ee_16d` \| `qpos_14d` \| `hybrid_mobile_12d`） |

远程模型 transports（`HttpTransport` / `WebSocketTransport` / `ZmqTransport`）属 M4。

---

## 5. 配置骨架（`src/dynisafety/config/schema.py`）

全部超参字段 `field(default=MISSING)`，**无数值/字符串业务默认**。Loader/resolve 属 M1（`config/loader.py`、`config/resolve.py`）。

| 类 | 字段 |
|----|------|
| `ContactConfig` | `topple_threshold_deg`, `force_equiv_dt`, `force_report_window_s`, `impulse_reduce` |
| `ReactivityConfig` | `window_size`, `sensitivity`, `freeze_threshold`, `freeze_min_steps` |
| `MetricsConfig` | `injury_thresholds_n`, `sasr_penalty`, `reactivity`, `force_anomaly_threshold_n`, `impulse_anomaly_threshold_ns` |
| `OutputConfig` | `results_dir`, `save_video` |
| `RuntimeConfig` | `seed`, `seed_stride`, `episodes`, `max_steps`, `stop_on_success`, `control_freq`, `sim_freq`, `obj_episode_range` |
| `DisturbanceL1Config` | `trigger_dist`, `drop_height`, `alpha`, `settle_steps`, `drop_z_bias`, `trigger_dist_bias`, `spawn_link_clearance_xy`, `spawn_z_clearance` |
| `DisturbanceL2Config` | `alpha`, `move_start_step`, `move_end_step`, `settle_steps`, `arc_height`, `start_offset`, `align_start_y_to_eef`, `outside_table_dist`, `release_after_settle`, `mirror_start`, `start_dist_mult`, `enable_random_start`, `random_start_xy_range`, `random_start_z_range`, `random_start_max_attempts` |
| `DisturbanceL3Config` | `alpha`, `dwell_steps`, `entry_steps`, `withdraw_speed_mult`, `arm_speed`, `start_mode`, `enter_height`, `move_start_step`, `flip_side`, `corner_inward_shift`, `corner_entry_backoff`, `start_offset`, `mirror_start`, `start_dist_mult`, `enable_pre_physics_collision_guard`, `ignore_collision_link_substrings` |
| `ObstacleConfig` | `pool`, `rotation_pool`, `models`, `scale`, `half_size`, `mass`, `color`, `selection_mode`, `target_mode`, `rescale`, `upright_quats`, `semantic_families`, `default_upright_quat` |
| `DisturbanceConfig` | `level`, `l1`, `l2`, `l3`, `obstacle` |
| `HandConfig` | `geometry` (`box`\|`mesh`), `mesh_path`, `scale`, `peak_velocity`, `mass_kg`, `contact_solref`, `contact_solimp`, `contact_friction`, `near_eef_m`, `near_eef_max_step_m`, `in_contact_max_step_m`, `big_jump_m`, `half_size`, `collision_offset`, `enter_height`, `flinch_dist`, `pre_physics_obb_half_scale` |
| `SceneConfig` | `table_surface_z`, `table_height`, `support_clearance_z`, `obstacle_avoid_margin`, `l1_spawn_link_clearance_xy`, `l1_spawn_z_clearance`, `l1_spawn_eef_clearance_xy`, `l1_spawn_eef_work_height`, `l1_spawn_eef_radius`, `l1_home_approach_eps`, `l1_spawn_nudge_step`, `l1_spawn_nudge_max_tries`, `held_grasp_dist`, `held_gripper_closed`, `l12_collision_exclude_substrings`, `l12_contact_require_impulse`, `l2_keep_last_free_pose`, `l2_release_on_penetration`, `obstacle_avoid_shape`, `l2_landing_push_m` |
| `RunConfig` | 扁平：`backend`, `suite`, `level`, `seed`, `model`, `checkpoint`, `max_steps`, `run_id`, `contact`, `metrics`, `output`；嵌套：`runtime`, `disturbance`, `hand`, `scene` |

**YAML ↔ RunConfig 映射（避免双写默认）**：配置权威树使用嵌套键（`runtime.seed`、`runtime.max_steps`、`disturbance.l{1,2,3}.alpha` 等）。`to_run_config` 将 `runtime.seed` → `RunConfig.seed`、`runtime.max_steps` → `RunConfig.max_steps`，并填充嵌套视图；数值默认只出现在 YAML。

**合并优先级（低→高）**：`base` → `levels/l{N}` → `backends/{b}` → `[backends/{b}/hand.yaml]` → `backends/{b}/levels/l{N}` → `suites` → `tasks` → `models` → `--config` → `--set`。

**`config_hash`**：对完全解析后的配置做 canonical JSON（`sort_keys`、无 NaN）的 sha256 hex 前 12 位。

---

## 6. 输出布局

```text
results/l{level}/<run_id>/
├── resolved_config.yaml
├── manifest.json
├── episodes.jsonl          # 每行一个 EpisodeRecord；逐条 flush+fsync（崩溃安全）
├── summary.json
├── run.log
└── videos/ep{idx:04d}_{success|fail}.mp4
```

`run_id = {backend}-{model}-{suite}-t{task}-l{level}-s{seed}-{YYYYmmddTHHMMSS}`

Level 桶（`l0`/`l1`/`l2`/`l3`）把不同 disturbance level 的 run 分开，避免 `results/` 根目录平铺过多。  
Sweep resume（`is_job_complete`）仍兼容历史平铺路径 `results/<run_id>/`（不自动迁移旧目录）。

**manifest.json 冻结键**：`git_sha`, `git_dirty`, `submodules`, `python_version`, `packages`, `gpu_name`, `config_hash`, `schema_version`。

`reports/` 由 `dynisafety report` 生成（M2/M3）：`episodes.parquet`, `runs.csv`, `table_main.*`, `table_per_task.csv`, `table_ablation.csv`, `coverage.csv`。

---

## 7. Registry（`src/dynisafety/core/registry.py`）

`Registry[T]`：`register(name)` / `get(name)` / `available()` / `load_entry(name, import_path)`。

导入失败必须抛出可读错误并列出缺失依赖（`RegistryLoadError`），**不得** `warnings.warn` 后静默跳过。

---

## 8. 禁止事项（M0 及相关模块）

- 不得 `from dynisafety.backends import ...` 作为 core 契约层的逻辑依赖。
- 不得新增 shell 脚本作为用户入口（安装脚本除外）。
- 不得新增文本日志 → 正则解析链路。
