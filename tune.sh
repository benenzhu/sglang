#!/bin/bash

# Distributed tuning on multiple GPUs
# DeepSeek V3 MLA shapes:
#   - 7168,16384  (o_proj: hidden_size -> num_heads * v_head_dim)
#   - 24576,1536  (q_b_proj: q_lora_rank -> num_heads * qk_head_dim)
#   - 2112,7168   (fused_qkv_a_proj: hidden_size -> q_lora_rank + kv_lora_rank + qk_rope_head_dim)

set -x
set -e
NK_LIST=("7168,16384" "24576,1536")
M_LIST=(16 32 64 128)

gpu_id=0
max_gpus=8

for nk in "${NK_LIST[@]}"; do
    for m in "${M_LIST[@]}"; do
        # 等待有空闲 GPU
        while [ $(jobs -r | wc -l) -ge $max_gpus ]; do
            sleep 1
        done
        
        # 生成文件名: nk 用下划线替换逗号
        nk_str=${nk//,/_}
        logfile="tune_m${m}_nk${nk_str}.log"
        
        echo "Starting: GPU $gpu_id, -m $m, -nk $nk -> $logfile"
        CUDA_VISIBLE_DEVICES=$gpu_id HIP_VISIBLE_DEVICES=$gpu_id \
            python -u tune_triton_gemm_a8w8_blockscale.py -m $m -nk $nk 2>&1 | tee $logfile &
        
        # 轮询 GPU
        gpu_id=$(( (gpu_id + 1) % max_gpus ))
    done
done

# Wait for all background jobs to complete
wait
echo "All tuning jobs completed!"