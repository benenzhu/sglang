# export ZZD=1
export HF_HOME=/B/huggingface
export SGLANG_TORCH_PROFILER_DIR=/root


SGLANG_USE_AITER=1 SGLANG_ROCM_FUSED_DECODE_MLA=0 \
python3 -m sglang.launch_server \
--model-path moonshotai/Kimi-K2.5 \
--tp 8 \
--trust-remote-code \
--decode-attention-backend triton \
--prefill-attention-backend aiter \
--reasoning-parser kimi_k2 \
--tool-call-parser kimi_k2 \
--disable-cuda-graph \
--load-format dummy \
--host 0.0.0.0 --port 30000 \
2>&1 |tee run_server.log

# --cuda-graph-max-bs=32 \




# python benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
#     --model moonshotai/Kimi-K2.5 \
#     --tp-size 8 \
#     --tune


# export HF_HOME=/B/huggingface
# SGLANG_USE_AITER=1 SGLANG_ROCM_FUSED_DECODE_MLA=0 \
# python3 -m sglang.launch_server \
# --model-path /sgl-workspace/sglang/DeepSeek-R1 \
# --tp 1 \
# --trust-remote-code \
# --decode-attention-backend triton \
# --prefill-attention-backend aiter \
# --reasoning-parser kimi_k2 \
# --tool-call-parser kimi_k2 \
# --host 0.0.0.0 --port 30000 \
# --load-format dummy
