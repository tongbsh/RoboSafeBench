# X-VLA Official Bridge 路径问题说明

## 1. 问题是什么

在下面这条命令下，`xvla_official` 之前会出现“机械臂乱动”、抓取失败、频繁碰错物体的问题：

```bash
bash dynamic_SimplerEnv/scripts/x-vla/run_bridge_xvla_eval.sh \
    --official \
    --task bridge_carrot \
    --level 0
```

当时的典型现象是：

- `bridge_carrot` 的 baseline episode 很快失败
- 日志里能看到 `moved_wrong_obj=True`
- 生成的视频里机械臂姿态变化明显不合理，像是在抖动或乱甩

## 2. 为什么会这样

根因不是 X-VLA server 本身，而是 **官方 WidowX client 语义** 和 **当前 DyniSafety-SimplerEnv 执行环境语义** 不一致。

### 2.1 官方 X-VLA WidowX 评测假设的是 absolute-EE 环境

`vlaModelHome/X-VLA/evaluation/simpler/README.md` 明确说明，官方 SimplerEnv 做过小改动，以支持 **absolute end-effector control**。

而当前 `dynamic_SimplerEnv` bundled 的环境仍然是标准的 delta 控制：

- `arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos`

也就是说：

- 官方 client 输出的 7 维动作，更接近“绝对 EE 目标语义”
- 当前环境执行器却把动作当成“每一步的 delta EE”

这两者直接混用，就会让动作解释错位。

### 2.2 旧的 `xvla_official` 适配层还有两个额外问题

#### 问题 A：`proprio` 构造和官方 client 不一致

官方 WidowX client 的状态更新方式是：

- 初始化 `proprio = pos(3) + [1,0,0,1,0,0,0] + zeros(10)`
- 每消费一步预测后，执行 `self.proprio[:10] = action_pred[:10]`

也就是后续请求使用的是 **模型预测出来的绝对状态记忆**。

旧适配层不是这样做的，而是把真实 `eef_pos` 的 `quat -> rot6d + grip` 重新编码回去。这样会让 server 看到的输入分布和官方客户端不一样，首步输出就会明显偏掉。

#### 问题 B：把 `euler+bias` 直接当成 `rot_axangle`

官方 WidowX client 会把：

- `rot6d -> euler(xyz) + [0, pi/2, 0]`

作为官方环境需要的 3 维旋转语义输出。

但在当前 delta-EE SimplerEnv 里，`rot_axangle` 应该是 **相对旋转的 rotvec/axis-angle**，不是 Euler 三元组。

旧适配层直接把 `euler+bias` 塞给 `rot_axangle`，会把单步旋转放大到极不合理的量级。一次运行时检查里：

- 错误做法的旋转量级约为 `1.58 rad`
- 正确的 delta 旋转量级约为 `0.05 ~ 0.06 rad`

这就是机械臂“乱甩”的最直接原因。

## 3. 我是如何解决的

本次修复分成三部分。

### 3.1 做成双环境共存

目标是：

- 其他模型继续默认使用 bundled 的 `dynamic_SimplerEnv/SimplerEnv-OpenVLA`
- 只有 `xvla_official` 可以切到另一套 X-VLA 官方 absolute-EE SimplerEnv

为此，在 `scripts/x-vla/run_bridge_xvla_eval.sh` 中加入了：

- `--official-simplerenv-root`
- `SIMPLERENV_ROOT_OFFICIAL`
- 自动打印当前实际使用的 `SIMPLERENV_ROOT`

并且只在 `--official` 时才覆盖 `SIMPLERENV_ROOT`。

### 3.2 把 `xvla_official` loader 从当前 SimplerEnv 父目录解耦

之前 `xvla_official` 的工厂函数默认从“当前被导入的 SimplerEnv checkout 的父目录”去找：

- `scripts/x-vla/xvla_official_inference.py`

一旦切到另一套 `SIMPLERENV_ROOT`，这里就容易找错脚本。

现在改成优先使用：

- `DYNAMIC_SIMPLERENV_ROOT`

这样即使 `SIMPLERENV_ROOT` 切到另一套 official checkout，真正的 adapter 代码仍然从当前仓库的 `dynamic_SimplerEnv/scripts/x-vla/` 加载。

### 3.3 把 `xvla_official` 适配层拆成两种执行语义

现在 `scripts/x-vla/xvla_official_inference.py` 支持两套模式。

#### `simpler_delta`

给当前 bundled 的 delta-EE 环境使用。

它保留官方 client 的关键语义：

- action chunk / `deque` 机制不变
- `proprio` 初始化改为官方风格
- 每步继续执行 `self._proprio_20[:10] = action_pred[:10]`

但在执行前显式把动作从 **绝对 EE** 转成 **delta EE**：

- 平移：`pred_xyz - current_xyz`
- 旋转：先恢复目标旋转，再与当前 `eef_pos` 求相对 `rotvec`

#### `upstream_literal`

给真正的官方 absolute-EE SimplerEnv 使用。

这个模式尽量逐字对齐官方 WidowX client：

- `proprio` 初始化方式与官方一致
- chunk 行为与官方一致
- 输出维持官方的 7 维语义：
  - `xyz`
  - `euler(xyz) + [0, pi/2, 0]`
  - binary gripper

## 4. 安装了什么包

### 4.1 本次修复没有新增任何 pip/conda 包

这次修改是纯代码修复，没有额外安装新的 Python 包。

沿用的是仓库原本已经在使用的环境：

- X-VLA server 运行环境：`XVLA`
- DyniSafety / SimplerEnv 评测环境：`simpler_env`

### 4.2 如果你要接入另一套 official SimplerEnv

那一套 root 自己仍然需要像普通 SimplerEnv 一样安装成 editable package：

```bash
conda activate simpler_env
pip install -e /path/to/SimplerEnv-OpenVLA
pip install -e /path/to/SimplerEnv-OpenVLA/ManiSkill2_real2sim
```

### 4.3 这次修复依赖的现有包

以下依赖是当前环境中已经存在并被本路径使用的，不是这次新增安装的：

- `requests`
- `json_numpy`
- `numpy`
- `scipy`
- `opencv-python`
- `matplotlib`
- `tensorflow`
- `sapien`

## 5. 如何启动推理和评测

### 5.1 启动 X-VLA server

```bash
bash dynamic_SimplerEnv/scripts/x-vla/start_xvla_server.sh \
    --model-path download_checkpoints/xVLA/xvla-simpler-widowx
```

默认：

- 端口：`8010`
- conda env：`XVLA`
- domain id：`0`（WidowX）

### 5.2 用 bundled root 跑 `xvla_official` 回退路径

这条命令主要用于验证：

- `--official-simplerenv-root` 路由是否生效
- 在 bundled delta-EE 环境下是否自动走 `simpler_delta`

```bash
bash dynamic_SimplerEnv/scripts/x-vla/run_bridge_xvla_eval.sh \
    --official \
    --official-simplerenv-root /root/autodl-tmp/code/dynamic_safty_bench/dynamic_SimplerEnv/SimplerEnv-OpenVLA \
    --task bridge_carrot \
    --level 0 \
    --run-id-note official-root-e2e
```

### 5.3 用真正的 alternate official root 跑

```bash
bash dynamic_SimplerEnv/scripts/x-vla/run_bridge_xvla_eval.sh \
    --official \
    --official-simplerenv-root /path/to/SimplerEnv-XVLA-Official \
    --task bridge_carrot \
    --level 0
```

如果脚本识别到这不是 bundled root，且你没有手动设置 `XVLA_EXEC_MODE`，它会自动切到：

```bash
XVLA_EXEC_MODE=upstream_literal
```

## 6. 如何测试推理是否正常运行

### 6.1 看终端启动信息

运行 `--official` 路径后，终端应打印：

- `SimplerEnv root: ...`
- `Exec mode     : simpler_delta` 或 `upstream_literal`
- `Adapter root  : .../dynamic_SimplerEnv`

这三项能直接看出：

- 是否真的切到了目标 root
- 当前在用哪套动作语义
- adapter 是否仍从当前仓库加载

### 6.2 看日志与视频是否生成

本次验证命令生成的日志示例：

- `dynamic_SimplerEnv/outputs/xvla_official/logs/EVAL-dynisafety-2026_04_12-17_28_40--official-root-e2e-L0.txt`

本次验证命令生成的成功视频示例：

- `dynamic_SimplerEnv/outputs/xvla_official/videos/bridge_carrot_level0/2026_04_12/task0_put_carrot_on_plate/2026_04_12-17_28_40--dynisafety--episode=1--success=True--task=put_carrot_on_plate.mp4`

### 6.3 看行为是否恢复正常

这次修复后的 bundled-root 回退验证中，`bridge_carrot` 已经出现成功 rollout，至少前几个 episode 可见：

- `episode=1 -> success=True`
- `episode=2 -> success=True`
- `episode=3 -> success=True`

这说明：

- 入口脚本路由是通的
- `xvla_official` adapter 正在正常向 server 请求动作
- `simpler_delta` 的绝对位姿到 delta 位姿转换已经恢复正常

### 6.4 运行时快速检查点

如果需要进一步排查，可重点检查：

- `XVLAOfficialBridgeInference` 是否成功实例化
- `simpler_delta` 下 `world_vector` 是否接近 `pred_abs_pos - current_eef`
- `simpler_delta` 下单步旋转量级是否已经回到小量级（约 `0.05 ~ 0.06 rad`）
- `upstream_literal` 下 `rot_axangle` 是否严格等于 `rotate6d_to_euler_xyz(rot6) + [0, pi/2, 0]`

## 7. 当前已知边界

不是任意一份“看起来像 SimplerEnv”的目录都可以直接拿来当 `--official-simplerenv-root`。

至少需要满足：

- 能导入 `simpler_env.evaluation.argparse`
- 能导入 `simpler_env.policies.create_policy`
- 有可用的 `ManiSkill2_real2sim`
- 与当前 Bridge 任务所需资源路径兼容

例如这次检查过的：

- `past_files_cache/official_github/SimplerEnv-OpenVLA`

它虽然能导入 `simpler_env.evaluation.argparse`，但缺少我们这套 DyniSafety 入口依赖的 `create_policy` registry，因此 **不能直接** 作为真实 alternate root 使用。

## 8. 结论

这次问题的本质是：

- 官方 X-VLA WidowX client 的动作语义偏 absolute-EE
- 当前 DyniSafety-SimplerEnv bundled 环境执行的是 delta-EE
- 旧 adapter 又把官方 Euler 旋转直接塞进了 delta 控制器

修复后，这条路径已经支持：

- 默认 bundled root 下安全回退到 `simpler_delta`
- 未来切官方 absolute-EE root 时自动切到 `upstream_literal`
- `xvla_official` adapter 与当前仓库路径解耦，支持真正的双环境共存
