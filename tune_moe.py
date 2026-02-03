# SPDX-License-Identifier: MIT
# Triton Fused MoE Kernel Tuning Script
#
# Usage:
#     # Tune using saved MoE inputs
#     python tune_moe.py --input /tmp/topk/moe_inputs_M32__0.pt
#
#     # Tune with specific config
#     python tune_moe.py --input /tmp/topk/moe_inputs_M32__0.pt --config "64,64,128,8,4,2"

import argparse
import json
import os
import sys
import itertools
from typing import Dict, List, Tuple
from tqdm.auto import tqdm

import torch
import triton
import triton.language as tl
from torch.profiler import ProfilerActivity, profile

# Add sglang to path
# sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python"))

from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import (
    invoke_fused_moe_kernel,
)
from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import (
    moe_align_block_size,
)


def generate_moe_configs() -> List[Dict]:
    """Generate MoE kernel config search space."""
    configs = []
    
    block_sizes_m = [16, 32, 64, 128]
    block_sizes_n = [32, 64, 128, 256]
    block_sizes_k = [32, 64, 128]
    group_sizes_m = [1, 4, 8]
    num_warps_list = [1, 2, 4, 8]
    num_stages_list = [2, 3, 4]
    waves_per_eu_list = [1, 2, 4, 8]
    matrix_instr_nonkdim_list = [16]


    for (
        block_m,
        block_n,
        block_k,
        group_m,
        num_warps,
        num_stages,
        waves_per_eu,
        matrix_instr_nonkdim,
    ) in itertools.product(
        block_sizes_m,
        block_sizes_n,
        block_sizes_k,
        group_sizes_m,
        num_warps_list,
        num_stages_list,
        waves_per_eu_list,
        matrix_instr_nonkdim_list,
    ):
        config = {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": group_m,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "waves_per_eu": waves_per_eu,
            "matrix_instr_nonkdim": matrix_instr_nonkdim,
        }
        configs.append(config)

    return configs


def do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode="median"):
    """
    Benchmark with large cache to reduce Triton overhead.
    Adapted from triton.testing.do_bench with larger cache size.
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]
    from triton.testing import runtime
    di = runtime.driver.active.get_device_interface()

    fn()
    di.synchronize()

    # Use larger cache (16 * 256 MB) to reduce overhead
    cache_size = 16 * 256 * 1024 * 1024
    cache = torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')

    # Estimate the runtime of the function
    start_event = di.Event(enable_timing=True)
    end_event = di.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        runtime.driver.active.clear_cache(cache)
        fn()
    end_event.record()
    di.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # Compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    
    # Warm-up
    for _ in range(n_warmup):
        fn()
    
    # Benchmark
    for i in range(n_repeat):
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # Clear L2 cache before each run
        runtime.driver.active.clear_cache(cache)
        runtime.driver.active.clear_cache(cache)
        start_event[i].record()
        fn()
        end_event[i].record()
    
    di.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    
    from triton.testing import _summarize_statistics
    return _summarize_statistics(times, quantiles, return_mode)


def load_moe_inputs(input_path: str) -> Dict:
    """Load saved MoE inputs from file."""
    print(f"Loading MoE inputs from: {input_path}")
    data = torch.load(input_path)
    
    # Move to GPU
    for key in data:
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].cuda()
    
    print(f"  M={data['M']}, E={data['E']}, N={data['N']}")
    print(f"  hidden_states: {data['hidden_states'].shape}")
    print(f"  w1: {data['w1'].shape}")
    print(f"  w2: {data['w2'].shape}")
    print(f"  topk_ids: {data['topk_ids'].shape}")
    
    return data


def benchmark_moe_config(
    data: Dict,
    config: Dict,
    use_int4_w4a16: bool = True,
    num_iters: int = 100,
) -> float:
    """Benchmark a single MoE config."""
    
    hidden_states = data["hidden_states"]
    w1 = data["w1"]
    # w2 = data["w2"]
    w1_scale = data["w1_scale"]
    # w2_scale = data["w2_scale"]
    topk_ids = data["topk_ids"]
    topk_weights = data["topk_weights"]
    
    M = data["M"]
    E = data["E"]
    N = data["N"]
    
    # Get block_shape from w1_scale shape if available
    if w1_scale is not None and w1_scale.ndim == 3:
        # block_shape = [block_n, group_size]
        # w1_scale shape: [E, N, K // group_size]
        K_original = hidden_states.shape[1]
        num_groups = w1_scale.shape[2]
        group_size = K_original // num_groups
        block_shape = [0, group_size]

    else:
        block_shape = None
    assert block_shape == [0, 32], block_shape
    
    # Re-run moe_align_block_size with new BLOCK_SIZE_M
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], E
    )
    
    # Prepare intermediate buffers
    topk = topk_ids.shape[1]
    total_tokens = M * topk + (E + 1) * (config["BLOCK_SIZE_M"] - 1)
    
    intermediate_cache1 = torch.empty(
        (total_tokens, N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    
    compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16
    
    def run_kernel():
        invoke_fused_moe_kernel(
            hidden_states,
            w1,
            None,  # bias
            intermediate_cache1,
            None,  # a_scale
            w1_scale,
            None,  # w_zp
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,  # mul_routed_weight
            topk,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=False,
            block_shape=block_shape,
        )
    
    # Warmup
    run_kernel()
    torch.cuda.synchronize()
    
    # Benchmark
    elapsed_ms = do_bench(run_kernel, warmup=100, rep=num_iters, return_mode="median")
    elapsed_us = elapsed_ms * 1000
    
    return elapsed_us


def tune_moe_kernel(
    data: Dict,
    use_int4_w4a16: bool = True,
    verbose: bool = False,
) -> Tuple[Dict, float]:
    """Tune MoE kernel for loaded data."""
    
    configs = generate_moe_configs()
    best_config = None
    best_time = float("inf")
    
    M, E, N = data["M"], data["E"], data["N"]
    print(f"\nTuning MoE kernel: M={M}, E={E}, N={N}")
    print(f"Total configs to try: {len(configs)}")
    


    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        elapsed_us = benchmark_moe_config(data, data["config"], True, num_iters=100)
        print(f"defualt_config time: {elapsed_us:.2f} us")
        default_config = {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 1, 'num_stages': 3, 'waves_per_eu': 1, 'matrix_instr_nonkdim': 16}
        elapsed_us = benchmark_moe_config(data, default_config, True, num_iters=100)
        print(f"Best config time: {elapsed_us:.2f} us")
    prof.export_chrome_trace(f"tune.json")
    for i, config in enumerate(tqdm(configs, desc="Tuning")):
        try:
            elapsed_us = benchmark_moe_config(
                data, config, use_int4_w4a16=use_int4_w4a16, num_iters=100
            )
            
            if elapsed_us < best_time * 1.1:
                # Re-verify with more iterations
                elapsed_us2 = benchmark_moe_config(
                    data, config, use_int4_w4a16=use_int4_w4a16, num_iters=1000
                )
                
                if elapsed_us2 < best_time:
                    best_time = elapsed_us2
                    best_config = config.copy()
                    print(f"\n  NEW BEST: {elapsed_us2:.2f} us - {config}")
            
            if verbose and i % 50 == 0:
                print(f"  Progress: {i}/{len(configs)}, best so far: {best_time:.2f} us")
                
        except Exception as e:
            if verbose:
                print(f"  Config {i} ERROR: {e}")
            continue
    
    if best_config:
        print(f"\n{'='*60}")
        print(f"Best config: {best_time:.2f} us")
        print(f"  {best_config}")
        print(f"{'='*60}")
    
    return best_config, best_time


def benchmark_single_config(
    data: Dict,
    config: Dict,
    use_int4_w4a16: bool = True,
) -> float:
    """Benchmark a single specific config."""
    
    print(f"\nBenchmarking config: {config}")
    
    elapsed_us = benchmark_moe_config(
        data, config, use_int4_w4a16=use_int4_w4a16, num_iters=500
    )
    
    print(f"Result: {elapsed_us:.2f} us")
    return elapsed_us


def main():
    parser = argparse.ArgumentParser(description="Tune Fused MoE Triton kernel")
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to saved MoE inputs (.pt file)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Specific config to benchmark: 'BLOCK_M,BLOCK_N,BLOCK_K,GROUP_M,num_warps,num_stages'",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file for best config",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["int4", "fp8", "auto"],
        default="int4",
        help="Quantization type",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    # Device info
    device_props = torch.cuda.get_device_properties(0)
    print(f"Device: {device_props.name}")
    
    # Load data
    data = load_moe_inputs(args.input)
    
    use_int4_w4a16 = args.dtype == "int4"
    
    if args.config:
        # Benchmark specific config
        parts = list(map(int, args.config.split(",")))
        config = {
            "BLOCK_SIZE_M": parts[0],
            "BLOCK_SIZE_N": parts[1],
            "BLOCK_SIZE_K": parts[2],
            "GROUP_SIZE_M": parts[3],
            "num_warps": parts[4],
            "num_stages": parts[5],
        }
        benchmark_single_config(data, config, use_int4_w4a16=use_int4_w4a16)
    else:
        # Full tuning
        best_config, best_time = tune_moe_kernel(
            data, use_int4_w4a16=use_int4_w4a16, verbose=args.verbose
        )
        
        if args.output and best_config:
            result = {
                "M": data["M"],
                "E": data["E"],
                "N": data["N"],
                "best_time_us": best_time,
                "config": best_config,
            }
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
