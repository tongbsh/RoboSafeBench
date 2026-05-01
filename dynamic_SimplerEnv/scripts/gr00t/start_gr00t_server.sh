#!/bin/bash
# ============================================================
# GR00T in DyniSafety-SimplerEnv — NO SERVER NEEDED
#
# Unlike server-based VLAs (MemoryVLA, π₀, X-VLA, etc.), the GR00T
# policy for SimplerEnv loads the model IN-PROCESS via --ckpt-path.
# You do NOT need to start a separate server.
#
# Usage:
#   # Run evaluation directly (no server required):
#   GR00T_CKPT=/path/to/gr00t-checkpoint \
#   bash scripts/eval_dynisafety.sh --vla gr00t --task pick_coke_can --level 1
#
#   # Or equivalently:
#   bash scripts/run_dynisafety.sh \
#       --model gr00t \
#       --ckpt /path/to/gr00t-checkpoint \
#       --task pick_coke_can \
#       --level 1
#
# The checkpoint directory should be a standard GR00T / Isaac-GR00T
# checkpoint produced by fine-tuning on the target embodiment.
#
# If you mistakenly ran this script expecting a server, exit and use
# the commands above instead.
# ============================================================

echo "============================================================"
echo "GR00T in DyniSafety-SimplerEnv — no server required."
echo ""
echo "GR00T loads in-process via --ckpt-path. Run evaluation with:"
echo ""
echo "  GR00T_CKPT=/path/to/checkpoint \\"
echo "  bash scripts/eval_dynisafety.sh --vla gr00t --task pick_coke_can --level 1"
echo ""
echo "  or:"
echo ""
echo "  bash scripts/run_dynisafety.sh \\"
echo "      --model gr00t \\"
echo "      --ckpt /path/to/checkpoint \\"
echo "      --task pick_coke_can \\"
echo "      --level 1"
echo "============================================================"
exit 0
