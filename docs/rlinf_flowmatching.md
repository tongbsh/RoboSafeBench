# RLinf 中 OpenPI / Flow-Matching VLA 的强化学习说明

本文档基于仓库内 `past_files_cache/RLinf_dynamicEval` 源码整理，说明 **OpenPI 这类基于 Flow Matching 的 VLA** 在 RLinf 中如何做 RL，以及为何不是 token 分布仍能定义策略梯度与 PPO。

---

## 1. 核心思路

策略不再对离散 **token** 计算 `log π(a|s)`，而是把 **固定步数的去噪过程** 看成一条马尔可夫链：

1. 从噪声初始化 `x_T`（或外部给定的噪声），沿 `num_steps` 迭代；
2. 每一步由网络预测速度 `v_θ`，再按 `flow_ode` / `flow_sde` / `flow_cps` / `flow_noise` 等规则得到下一步分布的 **均值 `x_t_mean` 与标准差 `x_t_std`**；
3. 实际采样 `x_{next} = x_t_mean + noise * x_t_std`，视为从高斯中抽样；
4. 用各维高斯对数密度 `get_logprob_norm(sample, mu, sigma)` 得到 **单步转移 log-prob**，再按配置在时间维、动作维上做 **求和或均值**，得到可与旧策略做比率的 `log π`。

因此：**PPO 仍使用 `exp(log π_new - log π_old)`，但 log π 定义在连续去噪链的转移密度上。**

主要实现文件：

- `past_files_cache/RLinf_dynamicEval/rlinf/models/embodiment/openpi/openpi_action_model.py`
- `past_files_cache/RLinf_dynamicEval/rlinf/workers/actor/fsdp_actor_worker.py`
- `past_files_cache/RLinf_dynamicEval/rlinf/algorithms/utils.py`（`logprob_type` 聚合）
- `past_files_cache/RLinf_dynamicEval/rlinf/workers/actor/fsdp_nft_policy_worker.py`（NFT 变体）

---

## 2. Rollout：采样动作并缓存整条链

`predict_action_batch` 调用 `sample_actions`，除最终动作外，将 `chains`、`denoise_inds`、`prev_logprobs`、`prev_values` 写入 `forward_inputs`，供后续 actor 重算新策略 log-prob：

```python
# openpi_action_model.py — predict_action_batch（节选）
outputs = self.sample_actions(observation, mode=mode, compute_values=compute_values)
actions = self.output_transform(...)["actions"]
prev_logprobs = outputs["prev_logprobs"]
prev_values = outputs["prev_values"]

forward_inputs = {
    "chains": outputs["chains"],
    "denoise_inds": outputs["denoise_inds"],
    "tokenized_prompt": processed_obs["tokenized_prompt"],
    "tokenized_prompt_mask": processed_obs["tokenized_prompt_mask"],
    "action": actions.reshape(actions.shape[0], -1).contiguous(),
    "model_action": outputs["actions"].reshape(...).contiguous(),
}
```

`sample_actions` 内对每一步调用 `sample_mean_var_val` 得到 `(x_t_mean, x_t_std)`，再采样并计算 `get_logprob_norm(x_t, x_t_mean, x_t_std)`。

**训练时** 若未开启 `joint_logprob`，通常只在 **一个随机 denoise 下标** 上用随机策略（`noise_method`），其余步可走 `flow_ode`，以降低方差与算力；`joint_logprob` 则对多步 log-prob 做平均。评估时 `denoise_inds` 为 -1，表示按完整流程推理。

---

## 3. PPO 更新：`default_forward` 与 `get_log_prob_value`

Actor（`EmbodiedFSDPActor.run_training`）对 batch 调用 `self.model(forward_inputs=..., compute_logprobs=True, ...)`。OpenPI 的 `forward` 默认走 `default_forward`，用 **与 rollout 相同的 `chains` 和 `denoise_inds`** 重算当前参数 θ 下的转移 log-prob：

```python
# openpi_action_model.py — default_forward（逻辑摘要）
log_probs, value_t, entropy = self.get_log_prob_value(
    images, img_masks, lang_tokens, lang_masks, state,
    chains, denoise_inds, compute_values,
)
log_probs = log_probs[:, :, : self.config.action_chunk, : self.config.action_env_dim]
log_probs = log_probs.mean(dim=1)  # 时间维聚合方式之一
return {"logprobs": log_probs, "values": value_t, "entropy": entropy}
```

`get_log_prob_value` 从 `chains_pre`、`chains_next` 一步，用 `sample_mean_var_val(..., noise_method=...)` 再算 `μ, σ`，并对 **实际 rollout 中出现的 `chains_next`** 求 `get_logprob_norm(chains_next, x_t_mean, x_t_std)`。

`get_logprob_norm` 实现单步高斯对数密度（`safe_get_logprob` 时可用简化二次项）；`flow_ode` 下 σ 为 0 时通过 mask 避免除零。

---

## 4. 与 PPO 损失对接：`logprob_type`

`EmbodiedFSDPActor` 将 `output_dict["logprobs"]` 与 `prev_logprobs`、`advantages` 等传入 `policy_loss`。对 embodied 任务会先经 `preprocess_loss_inputs` 按 `logprob_type` 变形：

| `logprob_type` | 含义（形状从 `[B, chunks, dim]` 聚合） |
|----------------|----------------------------------------|
| `token_level`  | 保持与 `single_action_dim` 对齐的逐维结构 |
| `action_level` | 在最后一维求和 → 每个 chunk 一个标量 |
| `chunk_level`  | 在 chunk 与 dim 上求和 → 每个样本一个标量 |

例如 Libero OpenPI 配置中常见 `logprob_type: chunk_level`，便于与 `reward_type: chunk_level`、GAE 的 advantage 形状对齐。

---

## 5. 变体：NFT（非 PPO 比率）

当配置启用 NFT（`is_nft` 等）时，`EmbodiedNFTFSDPPolicy` 使用 `forward_type=ForwardType.NFT`，在显式 `(x_t, t)` 上预测 `v_θ`，并用 **DPO 风格** 的能量差 `ΔE` 与由 advantage 映射的偏好 `y` 构造 `softplus` 损失，同时对 `v_θ - v_old` 做 drift clip。此时优化的是 **流空间中的速度场与一步预测**，而不是 token 上的 KL。

相关实现：`rlinf/workers/actor/fsdp_nft_policy_worker.py` 中 `_nft_forward_and_loss`、`_compute_embodied_nft_loss`。

---

## 6. 变体：DSRL（SAC 噪声 + Flow 生成动作）

`OpenPi0Config.use_dsrl=True` 时，`predict_action_batch` 中由 `sac_forward` 对 **低维噪声动作** 产生 logprob，OpenPI 流模型将噪声映射为环境动作；RL 的密度主要落在 SAC 的高斯噪声策略上，Flow 头可按需冻结（见 `freeze_vlm` 与 DSRL 分支）。这是将「策略随机性」主要放在小维噪声空间的一种拆分。

---

## 7. 配置与评测注意

- **`num_steps`**：RL 训练使用固定离散去噪步数；**评测 RL 微调模型时必须与训练一致**（SFT 连续时间训练下步数可更灵活，但 RL 路径不同）。参考 `past_files_cache/RLinf_dynamicEval/toolkits/eval_scripts_openpi/README.md`。
- **`noise_method`**：如 `flow_sde` 给转移非零方差，使随机策略的 log-prob 更完整；`flow_ode` 为确定性 ODE 一步，需结合实现中的 mask 理解 log-prob 行为。
- **`add_value_head` + `adv_type: gae`**：在 suffix 或 VLM 上估计 value，与 PPO critic 配套。

---

## 8. 小结

| 问题 | RLinf / OpenPI 的做法 |
|------|------------------------|
| 没有 token 如何定义 π？ | 定义在 **去噪链每步的高斯转移** 上；链上聚合得到 rollout / 更新用的 log-prob。 |
| PPO 的比率从哪里来？ | `default_forward` 重算的 `logprobs` 与 rollout 存的 `prev_logprobs`，经 `logprob_type` 聚合后进入 `policy_loss`。 |
| 不用 PPO 比率？ | **NFT**：优势加权的速度场能量 / DPO 式损失。 |
| 进一步解耦随机性？ | **DSRL**：SAC 在噪声空间 logprob + Flow 生成动作。 |

---

*文档版本：与 `past_files_cache/RLinf_dynamicEval` 目录下 RLinf 快照一致；若上游 RLinf 更新，请以实际源码为准。*
