set -e
unset TORCHINDUCTOR_MAX_AUTOTUNE
unset TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE
cp sgl-kernel/python/sgl_kernel/hadamard.py /opt/venv/lib/python3.10/site-packages/sgl_kernel-0.3.18-py3.10-linux-x86_64.egg/sgl_kernel/hadamard.py
export MODEL="/data/DeepSeek-R1-0528"
export MODEL="/data/DeepSeek-V3.2-Exp"
export PORT=30000
export TP=8
set +e
ps aux | grep sglang |awk '{print $2}' | xargs kill -9
set -e
sleep 1

export SGLANG_NSA_FUSE_TOPK=false
export SGLANG_NSA_KV_CACHE_STORE_FP8=false
export SGLANG_NSA_USE_REAL_INDEXER=true
# export SGLANG_NSA_USE_TILELANG_PREFILL=True
# --nsa-decode "tilelang"
python3 -m sglang.launch_server \
	--model-path=$MODEL --host=0.0.0.0 --port=$PORT --trust-remote-code \
	--tensor-parallel-size=$TP \
	--mem-fraction-static=0.8 \
	--cuda-graph-max-bs=128 \
	--chunked-prefill-size=196608 \
	--num-continuous-decode-steps=4 \
	--max-prefill-tokens=196608 \
	--load-format dummy\
	--disable-radix-cache 2>&1 |tee a.log
