# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Triton GEMM A8W8 Blockscale Tuning Script for gfx942

Usage:
    # Tune for specific N,K shape with all M ranges
    python tune_triton_gemm_a8w8_blockscale.py -nk 2112,7168

    # Tune for specific M,N,K
    python tune_triton_gemm_a8w8_blockscale.py -m 16 -nk 2112,7168

    # Tune multiple shapes
    python tune_triton_gemm_a8w8_blockscale.py --shapes "2112,7168;4096,7168;7168,2048"

    # Output to specific file
    python tune_triton_gemm_a8w8_blockscale.py -nk 2112,7168 -o my_config.json
"""

import argparse
import itertools
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import triton
from aiter.ops.triton.gemm_a8w8_blockscale import gemm_a8w8_blockscale

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiter import dtypes
from aiter.ops.triton._triton_kernels.gemm_a8w8_blockscale import (
    _gemm_a8w8_blockscale_kernel,
    _gemm_a8w8_blockscale_reduce_kernel,
)
from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params

# Block scale shape
BLOCK_SHAPE = (128, 128)  # (block_n, block_k)

def generate_configs(K: int) -> List[Dict]:
    """Generate all possible config combinations."""
    configs = [None]

    # Config search space
    block_sizes_m = [16, 32, 64, 128]
    block_sizes_n = [16, 32, 64, 128]
    block_sizes_k = [128]  # Must match GROUP_K (typically 128 for blockscale)
    group_sizes_m = [1, 4]
    num_warps_list = [2, 4]
    num_stages_list = [1, 2]
    waves_per_eu_list = [1, 2, 4]
    cache_modifiers = [None, ".cg"]

    # NUM_KSPLIT depends on K
    num_ksplit_options = [1]
    if K >= 1024:
        num_ksplit_options.extend([2, 4])
    if K >= 4096:
        num_ksplit_options.extend([7])
    if K >= 7168:
        num_ksplit_options.extend([14])

    for (
        block_m,
        block_n,
        block_k,
        group_m,
        num_warps,
        num_stages,
        waves_per_eu,
        cache_mod,
        num_ksplit,
    ) in itertools.product(
        block_sizes_m,
        block_sizes_n,
        block_sizes_k,
        group_sizes_m,
        num_warps_list,
        num_stages_list,
        waves_per_eu_list,
        cache_modifiers,
        num_ksplit_options,
    ):
        config = {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": group_m,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "waves_per_eu": waves_per_eu,
            "matrix_instr_nonkdim": 16,
            "cache_modifier": cache_mod,
            "NUM_KSPLIT": num_ksplit,
            "kpack": 2,
        }
        configs.append(config)
    
    if False:
        configs = []
        for i in [ {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 2, 'num_stages': 1, 'waves_per_eu': 1, 'matrix_instr_nonkdim': 16, 'cache_modifier': None, 'NUM_KSPLIT': 1, 'SPLITK_BLOCK_SIZE': 7168, 'GROUP_K': 128, 'GROUP_N': 128},
    {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 2, 'num_stages': 1, 'waves_per_eu': 1, 'matrix_instr_nonkdim': 16, 'cache_modifier': None, 'NUM_KSPLIT': 2, 'SPLITK_BLOCK_SIZE': 3584, 'GROUP_K': 128, 'GROUP_N': 128},
    {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 2, 'num_stages': 1, 'waves_per_eu': 1, 'matrix_instr_nonkdim': 16, 'cache_modifier': None, 'NUM_KSPLIT': 4, 'SPLITK_BLOCK_SIZE': 1792, 'GROUP_K': 128, 'GROUP_N': 128},
    {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 2, 'num_stages': 1, 'waves_per_eu': 1, 'matrix_instr_nonkdim': 16, 'cache_modifier': None, 'NUM_KSPLIT': 7, 'SPLITK_BLOCK_SIZE': 1024, 'GROUP_K': 128, 'GROUP_N': 128},
    {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 2, 'num_stages': 1, 'waves_per_eu': 1, 'matrix_instr_nonkdim': 16, 'cache_modifier': '.cg', 'NUM_KSPLIT': 4, 'SPLITK_BLOCK_SIZE': 1792, 'GROUP_K': 128, 'GROUP_N': 128},
    {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 2, 'num_stages': 2, 'waves_per_eu': 1, 'matrix_instr_nonkdim': 16, 'cache_modifier': None, 'NUM_KSPLIT': 7, 'SPLITK_BLOCK_SIZE': 1024, 'GROUP_K': 128, 'GROUP_N': 128},
    {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 2, 'num_stages': 2, 'waves_per_eu': 1, 'matrix_instr_nonkdim': 16, 'cache_modifier': '.cg', 'NUM_KSPLIT': 4, 'SPLITK_BLOCK_SIZE': 1792, 'GROUP_K': 128, 'GROUP_N': 128},
    {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 2, 'num_stages': 1, 'waves_per_eu': 1, 'matrix_instr_nonkdim': 16, 'cache_modifier': None, 'NUM_KSPLIT': 7, 'SPLITK_BLOCK_SIZE': 1024, 'GROUP_K': 128, 'GROUP_N': 128},
    {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 2, 'num_stages': 1, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'cache_modifier': None, 'NUM_KSPLIT': 7, 'SPLITK_BLOCK_SIZE': 1024, 'GROUP_K': 128, 'GROUP_N': 128},
    {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 2, 'num_stages': 2, 'waves_per_eu': 1, 'matrix_instr_nonkdim': 16, 'cache_modifier': None, 'NUM_KSPLIT': 7, 'SPLITK_BLOCK_SIZE': 1024, 'GROUP_K': 128, 'GROUP_N': 128},
    {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 2, 'num_stages': 2, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'cache_modifier': None, 'NUM_KSPLIT': 7, 'SPLITK_BLOCK_SIZE': 1024, 'GROUP_K': 128, 'GROUP_N': 128},
    {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 2, 'num_stages': 2, 'waves_per_eu': 1, 'matrix_instr_nonkdim': 16, 'cache_modifier': None, 'NUM_KSPLIT': 14, 'SPLITK_BLOCK_SIZE': 512, 'GROUP_K': 128, 'GROUP_N': 128},
    {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 2, 'num_stages': 2, 'waves_per_eu': 2, 'matrix_instr_nonkdim': 16, 'cache_modifier': None, 'NUM_KSPLIT': 14, 'SPLITK_BLOCK_SIZE': 512, 'GROUP_K': 128, 'GROUP_N': 128},
    {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 2, 'num_stages': 2, 'waves_per_eu': 4, 'matrix_instr_nonkdim': 16, 'cache_modifier': None, 'NUM_KSPLIT': 14, 'SPLITK_BLOCK_SIZE': 512, 'GROUP_K': 128, 'GROUP_N': 128},
    {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4, 'num_warps': 2, 'num_stages': 2, 'waves_per_eu': 1, 'matrix_instr_nonkdim': 16, 'cache_modifier': None, 'NUM_KSPLIT': 14, 'SPLITK_BLOCK_SIZE': 512, 'GROUP_K': 128, 'GROUP_N': 128},
    {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4, 'num_warps': 4, 'num_stages': 2, 'waves_per_eu': 1, 'matrix_instr_nonkdim': 16, 'cache_modifier': '.cg', 'NUM_KSPLIT': 7, 'SPLITK_BLOCK_SIZE': 1024, 'GROUP_K': 128, 'GROUP_N': 128},
    {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 2, 'num_stages': 2, 'waves_per_eu': 1, 'matrix_instr_nonkdim': 16, 'cache_modifier': '.cg', 'NUM_KSPLIT': 14, 'SPLITK_BLOCK_SIZE': 512, 'GROUP_K': 128, 'GROUP_N': 128},
    {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4, 'num_warps': 2, 'num_stages': 2, 'waves_per_eu': 1, 'matrix_instr_nonkdim': 16, 'cache_modifier': '.cg', 'NUM_KSPLIT': 14, 'SPLITK_BLOCK_SIZE': 512, 'GROUP_K': 128, 'GROUP_N': 128},
        ]:
            configs.append(i.copy())
            i['kpack'] = 2
            configs.append(i)

    return configs



"""
从 triton dobench 抄过来的, 稍微改一点 cache 大小，方便调试
"""
def do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode="mean"):
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float], optional
    :param return_mode: The statistical measure to return. Options are "min", "max", "mean", "median", or "all". Default is "mean".
    :type return_mode: str
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]
    from triton.testing import runtime
    di = runtime.driver.active.get_device_interface()

    fn()
    di.synchronize()

    cache_size = 16 * 256 * 1024 * 1024
    cache = torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')
    # cache = runtime.driver.active.get_empty_cache_for_benchmark()

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

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        runtime.driver.active.clear_cache(cache)
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    di.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    from triton.testing import _summarize_statistics
    return _summarize_statistics(times, quantiles, return_mode)

def tune_for_shape(
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype = torch.bfloat16,
    verify: bool = True,
    verbose: bool = False,
) -> Tuple[Dict, float]:
    """Tune kernel for specific M, N, K shape."""
    block_shape_n, block_shape_k = BLOCK_SHAPE
    scale_m = M
    scale_n = (N + block_shape_n - 1) // block_shape_n
    scale_k = (K + block_shape_k - 1) // block_shape_k

    # Create test tensors
    x = (torch.rand((M, K), dtype=torch.float32, device="cuda") / 10).to(dtypes.fp8)
    weight = (torch.rand((N, K), dtype=torch.float32, device="cuda") / 10).to(dtypes.fp8)
    x_scale = torch.rand([scale_m, scale_k], dtype=torch.float32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=torch.float32, device="cuda")

    # Get reference
    # if verify:
    #     ref = get_reference(x, weight, x_scale, w_scale, dtype)

    configs = generate_configs(K)
    best_config = None
    best_time = float("inf")
    best_tflops = 0

    print(f"Tuning M={M}, N={N}, K={K} with {len(configs)} configs...")

    for i, config in enumerate(configs):
        try:
            fn = lambda: gemm_a8w8_blockscale(x, weight, x_scale, w_scale, config=config)
            elapsed_us = triton.testing.do_bench(fn, warmup=10, rep=100, return_mode="median") * 1000

            tflops = 2 * M * N * K / elapsed_us / 1e6

            if elapsed_us < best_time * 1.1:
                elapsed_us2 = do_bench(fn, warmup=100, rep=1000, return_mode="median") * 1000
                if elapsed_us2 < best_time:
                    best_time = elapsed_us2
                    best_config = config.copy() if config else None
                    best_tflops = tflops
                    # if verbose:
                    print(
                        f"  Config {i}: {elapsed_us2:.2f} us, {tflops:.2f} TFLOPS (NEW BEST) {elapsed_us:.2f} us {best_config}"
                    )
                else:
                    print(f" {elapsed_us2:.2f} us is not better than {elapsed_us:.2f} us")
            elif verbose and i % 50 == 0:
                print(f"  Progress: {i}/{len(configs)}")

        except Exception as e:
            # if verbose:
            print(f"  Config {i}: ERROR - {e}")

    if best_config:
        print(
            f"  Best: {best_time:.2f} us, {best_tflops:.2f} TFLOPS"
        )
        print(f"  Config: {best_config}")

    return best_config, best_time


def tune_all_m_ranges(
    N: int,
    K: int,
    dtype: torch.dtype = torch.bfloat16,
    verify: bool = True,
    verbose: bool = False,
) -> Dict:
    """Tune for all standard M ranges and return config dict."""
    # Standard M bounds
    m_bounds = [16, 32, 64, 128]
    m_test_values = {
        16: [1, 8, 16],
        32: [17, 24, 32],
        64: [33, 48, 64],
        128: [65, 96, 128],
    }

    config_dict = {}

    for bound in m_bounds:
        test_ms = m_test_values[bound]
        best_config = None
        best_avg_time = float("inf")

        print(f"\n{'='*60}")
        print(f"Tuning M_LEQ_{bound} (N={N}, K={K})")
        print(f"{'='*60}")

        # For each M bound, find config that works well across representative M values
        configs = generate_configs(K)
        config_times = {}

        for config_idx, config in enumerate(configs):
            total_time = 0
            valid = True

            for m in test_ms:
                try:
                    block_shape_n, block_shape_k = BLOCK_SHAPE
                    scale_m = m
                    scale_n = (N + block_shape_n - 1) // block_shape_n
                    scale_k = (K + block_shape_k - 1) // block_shape_k

                    x = (torch.rand((m, K), dtype=torch.float32, device="cuda") / 10).to(
                        dtypes.fp8
                    )
                    weight = (
                        torch.rand((N, K), dtype=torch.float32, device="cuda") / 10
                    ).to(dtypes.fp8)
                    x_scale = torch.rand(
                        [scale_m, scale_k], dtype=torch.float32, device="cuda"
                    )
                    w_scale = torch.rand(
                        [scale_n, scale_k], dtype=torch.float32, device="cuda"
                    )

                    out, elapsed_us = run_kernel_with_config(
                        x, weight, x_scale, w_scale, config, dtype
                    )

                    if out is None or elapsed_us == float("inf"):
                        valid = False
                        break

                    if verify:
                        ref = get_reference(x, weight, x_scale, w_scale, dtype)
                        max_diff = (out - ref).abs().max().item()
                        rel_err = max_diff / (ref.abs().max().item() + 1e-6)
                        if rel_err > 0.05:
                            valid = False
                            break

                    total_time += elapsed_us

                except Exception as e:
                    valid = False
                    break

            if valid:
                avg_time = total_time / len(test_ms)
                config_times[config_idx] = avg_time

                if avg_time < best_avg_time:
                    best_avg_time = avg_time
                    best_config = config.copy()

            if verbose and config_idx % 100 == 0:
                print(f"  Progress: {config_idx}/{len(configs)}")

        if best_config:
            config_dict[f"M_LEQ_{bound}"] = best_config
            print(f"  Best avg time: {best_avg_time:.2f} us")
            print(f"  Config: {best_config}")
        else:
            print(f"  WARNING: No valid config found for M_LEQ_{bound}")

    # Add "any" config (for large M)
    print(f"\n{'='*60}")
    print(f"Tuning 'any' (large M, N={N}, K={K})")
    print(f"{'='*60}")

    large_m_values = [256, 512, 1024]
    configs = generate_configs(K)
    best_config = None
    best_avg_time = float("inf")

    for config_idx, config in enumerate(configs):
        total_time = 0
        valid = True

        for m in large_m_values:
            try:
                block_shape_n, block_shape_k = BLOCK_SHAPE
                scale_m = m
                scale_n = (N + block_shape_n - 1) // block_shape_n
                scale_k = (K + block_shape_k - 1) // block_shape_k

                x = (torch.rand((m, K), dtype=torch.float32, device="cuda") / 10).to(
                    dtypes.fp8
                )
                weight = (torch.rand((N, K), dtype=torch.float32, device="cuda") / 10).to(
                    dtypes.fp8
                )
                x_scale = torch.rand([scale_m, scale_k], dtype=torch.float32, device="cuda")
                w_scale = torch.rand([scale_n, scale_k], dtype=torch.float32, device="cuda")

                out, elapsed_us = run_kernel_with_config(
                    x, weight, x_scale, w_scale, config, dtype
                )

                if out is None or elapsed_us == float("inf"):
                    valid = False
                    break

                if verify:
                    ref = get_reference(x, weight, x_scale, w_scale, dtype)
                    max_diff = (out - ref).abs().max().item()
                    rel_err = max_diff / (ref.abs().max().item() + 1e-6)
                    if rel_err > 0.05:
                        valid = False
                        break

                total_time += elapsed_us

            except Exception as e:
                valid = False
                break

        if valid:
            avg_time = total_time / len(large_m_values)
            if avg_time < best_avg_time:
                best_avg_time = avg_time
                best_config = config.copy()

        if verbose and config_idx % 100 == 0:
            print(f"  Progress: {config_idx}/{len(configs)}")

    if best_config:
        config_dict["any"] = best_config
        print(f"  Best avg time: {best_avg_time:.2f} us")
        print(f"  Config: {best_config}")

    return config_dict


def save_config(config_dict: Dict, output_path: str):
    """Save config dict to JSON file."""
    with open(output_path, "w") as f:
        json.dump(config_dict, f, indent=4)
    print(f"\nConfig saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Tune Triton GEMM A8W8 Blockscale kernel"
    )
    parser.add_argument("-m", type=int, default="32", help="Specific M value to tune")
    parser.add_argument(
        "-nk",
        type=str,
        default="2112,7168",
        help="N,K values (e.g., '2112,7168')",
    )
    parser.add_argument(
        "--shapes",
        type=str,
        default=None,
        help="Multiple N,K shapes separated by ';' (e.g., '2112,7168;4096,7168')",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip correctness verification",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Get device info
    device_props = torch.cuda.get_device_properties(0)
    device_name = device_props.name
    print(f"Device: {device_name}")

    # Determine arch name
    arch = "gfx942"  # Default, can be detected
    try:
        from aiter.ops.triton.utils._triton import arch_info
        arch = arch_info.get_arch()
    except:
        pass
    print(f"Architecture: {arch}")

    verify = not args.no_verify

    if args.shapes:
        # Multiple shapes
        shapes = [s.strip().split(",") for s in args.shapes.split(";")]
        for n_str, k_str in shapes:
            N, K = int(n_str), int(k_str)
            config_dict = tune_all_m_ranges(N, K, verify=verify, verbose=args.verbose)

            output_path = args.output or f"aiter/ops/triton/configs/gemm/{arch}-GEMM-A8W8_BLOCKSCALE-N={N}-K={K}.json"
            save_config(config_dict, output_path)

    elif args.nk:
        N, K = map(int, args.nk.split(","))

        if args.m is not None:
            # Tune specific M
            best_config, best_time = tune_for_shape(
                args.m, N, K, verify=verify, verbose=args.verbose
            )
            if best_config:
                print(f"\nBest config for M={args.m}, N={N}, K={K}:")
                print(json.dumps(best_config, indent=2))
        else:
            # Tune all M ranges
            config_dict = tune_all_m_ranges(N, K, verify=verify, verbose=args.verbose)

            output_path = args.output or f"aiter/ops/triton/configs/gemm/{arch}-GEMM-A8W8_BLOCKSCALE-N={N}-K={K}.json"
            save_config(config_dict, output_path)
    else:
        print("Please specify -nk or --shapes")
        parser.print_help()
        return


if __name__ == "__main__":
    main()