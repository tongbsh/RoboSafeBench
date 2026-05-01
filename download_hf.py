
from huggingface_hub import snapshot_download

# 下载 BridgeVLA 预训练 PaliGemma 模型（替代 gated 的 google/paligemma-3b-pt-224）
# snapshot_download(
#     repo_id='RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-object',
#     # repo_type='dataset',
#     max_workers=3,
#     local_dir='/root/autodl-tmp/code/dynamic_safty_bench/download_checkpoints/openvla-oft_RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-object'
# )
# snapshot_download(
#     repo_id='RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-goal',
#     # repo_type='dataset',
#     max_workers=3,
#     local_dir='/root/autodl-tmp/code/dynamic_safty_bench/download_checkpoints/openvla-oft_RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-goal'
# )
# snapshot_download(
#     repo_id='robbyant/lingbot-va-posttrain-robotwin',
#     # repo_type='dataset',
#     max_workers=3,
#     local_dir='/root/autodl-tmp/code/dynamic_safty_bench/download_checkpoints/lingbot-va/lingbot-va-posttrain-robotwin'
# )

# snapshot_download(
#     repo_id='InternRobotics/InternVLA-M1-Pretrain-RT-1-Bridge',
#     # repo_type='dataset',
#     max_workers=3,
#     # NOTE: must include sibling artifacts of checkpoints/*.pt — InternVLA_M1.from_pretrained
#     # asserts both `<RUN>/config.yaml` and `<RUN>/dataset_statistics.json` exist next to the
#     # checkpoints/ directory (see InternVLA/model/framework/share_tools.py:read_mode_config).
#     # Previously this was `allow_patterns='checkpoints/*'` which silently dropped them.
#     allow_patterns=[
#         'checkpoints/*',
#         'config.yaml',
#         'dataset_statistics.json',
#         'summary.jsonl',
#         'README.md',
#     ],
#     local_dir='/root/autodl-tmp/code/dynamic_safty_bench/download_checkpoints/InternVLA/InternVLA-M1-Pretrain-RT-1-Bridge'
# )
#     snapshot_download(
#         repo_id='robbyant/lingbot-va-base',
#         # repo_type='dataset',
#         max_workers=3,
#         # NOTE: must include sibling artifacts of checkpoints/*.pt — InternVLA_M1.from_pretrained
#         # asserts both `<RUN>/config.yaml` and `<RUN>/dataset_statistics.json` exist next to the
#         # checkpoints/ directory (see InternVLA/model/framework/share_tools.py:read_mode_config).
#         # Previously this was `allow_patterns='checkpoints/*'` which silently dropped them.
#         local_dir='/root/autodl-tmp/code/dynamic_safty_bench/download_checkpoints/lingbot-va/lingbot-va-base'
#     )
# snapshot_download(
#     repo_id='StarVLA/Qwen-GR00T-Bridge',
#     # repo_type='dataset',
#     max_workers=3,
#     local_dir='/root/autodl-tmp/code/dynamic_safty_bench/download_checkpoints/starVLA/Qwen-GR00T-Bridge'
# )

# snapshot_download(
#     repo_id='2toINF/X-VLA-RoboTwin2',
#     # repo_type='dataset',
#     max_workers=3,
#     local_dir='/root/autodl-tmp/code/dynamic_safty_bench/download_checkpoints/xVLA/X-VLA-RoboTwin2'
# )
# snapshot_download(
#     repo_id='InternRobotics/InternVLA-A1-3B-RoboTwin',
#     # repo_type='dataset',
#     max_workers=3,
#     local_dir='/root/autodl-tmp/code/dynamic_safty_bench/InternVLA-A1-3B-RoboTwin'
# )

# Fast-WAM（yuanty/fastwam）：仅下载 RoboTwin 评测用 checkpoint 与 dataset_stats（约各 12GB + json）
# 官方说明：https://huggingface.co/yuanty/fastwam
snapshot_download(
    repo_id='yuanty/fastwam',
    repo_type='model',
    max_workers=3,
    allow_patterns=[
        'robotwin_uncond_3cam_384.pt',
        'robotwin_uncond_3cam_384_dataset_stats.json',
        'README.md',
    ],
    local_dir='/root/autodl-tmp/code/dynamic_safty_bench/download_checkpoints/fastwam/robotwin',
)
# Fast-WAM 预处理 RoboTwin 2.0 训练数据（LeRobot v2.1；分卷 tar，下载后执行：
#   cd <local_dir> && cat robotwin2.0.tar.gz.part-* | tar -xzf -
# 数据集页：https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam
# snapshot_download(
#     repo_id='yuanty/robotwin2.0-fastwam',
#     repo_type='dataset',
#     max_workers=3,
#     local_dir='/root/autodl-tmp/code/dynamic_safty_bench/download_datasets/robotwin2.0-fastwam',
# )

# 下载 RLBench 评测数据集
# snapshot_download(
#     repo_id='LPY/BridgeVLA_RLBench_EVAL_DATA',
#     repo_type='dataset',
#     local_dir='/root/autodl-tmp/code/dynamic_safty_bench/download_datasets/rlbench_eval'
# )

# 下载 RLBench checkpoint
# snapshot_download(
#     repo_id='LPY/BridgeVLA',
#     repo_type='dataset',
#     allow_patterns='checkpoints/RLBench/*',
#     local_dir='/root/autodl-tmp/code/dynamic_safty_bench/download_checkpoints/bridgeVLA'
# )