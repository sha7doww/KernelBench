#!/usr/bin/env python3
"""
Batch Evaluation from JSON Input

This script evaluates custom kernels against reference implementations
from a JSON file with the following structure:

[
  {
    "kernel_id": "kernel_1_000",
    "task_id": "task_1",
    "ref_src": "import torch...(Model class)",
    "custom_src": "import torch...(ModelNew class)"
  },
  ...
]

Usage:
    python scripts/eval_from_json.py input_file=Input.json
    python scripts/eval_from_json.py input_file=Input.json num_gpus=4 measure_performance=True
    python scripts/eval_from_json.py input_file=Input.json output_file=results.json verbose=True
"""

import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional

import pydra
from pydra import Config, REQUIRED

import torch
from tqdm import tqdm

# Add src to path
REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_TOP_DIR, "src"))

from kernelbench.eval import (
    eval_kernel_against_ref,
    KernelExecResult,
    check_metadata_serializable_all_types,
)


class EvalFromJsonConfig(Config):
    """Configuration for JSON-based batch evaluation"""

    def __init__(self):
        # Input/Output
        self.input_file = REQUIRED  # Path to JSON file with kernels to evaluate
        self.output_file = "eval_results.json"  # Path to output results file

        # Evaluation settings
        self.num_correct_trials = 5  # Number of correctness trials per kernel
        self.num_perf_trials = 100  # Number of performance trials
        self.measure_performance = True  # Whether to measure performance
        self.timing_method = "cuda_event"  # Timing method: cuda_event, do_bench, etc.

        # Backend settings
        self.backend = "cuda"  # Backend: cuda, triton, tilelang, cute
        self.precision = "fp32"  # Precision: fp32, fp16, bf16

        # Parallelization
        self.num_gpus = 1  # Number of GPUs to use for parallel evaluation
        self.timeout = 300  # Timeout per kernel evaluation in seconds

        # Misc
        self.verbose = False  # Verbose output
        self.subset = (None, None)  # (start_idx, end_idx) to evaluate subset
        self.check_excessive_speedup = True  # Check for suspicious speedups


@dataclass
class SingleKernelResult:
    """Result for a single kernel evaluation"""
    kernel_id: str
    task_id: str
    compiled: bool = False
    correctness: bool = False
    runtime: float = -1.0  # microseconds
    ref_runtime: float = -1.0  # microseconds
    speedup: float = -1.0
    runtime_stats: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    eval_time: float = -1.0  # evaluation time in seconds


@dataclass
class BatchEvalResults:
    """Aggregate results for batch evaluation"""
    total_kernels: int = 0
    compiled_count: int = 0
    correct_count: int = 0
    faster_than_ref_count: int = 0  # speedup > 1.0
    fast_2x_count: int = 0  # speedup > 2.0

    # Rates
    compile_rate: float = 0.0
    correctness_rate: float = 0.0
    fast_1_rate: float = 0.0  # fraction correct AND faster
    fast_2_rate: float = 0.0  # fraction correct AND 2x faster

    # Timing
    total_eval_time: float = 0.0
    avg_eval_time: float = 0.0

    # Individual results
    results: List[Dict[str, Any]] = field(default_factory=list)

    # Config used
    config: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""


def get_precision_dtype(precision: str) -> torch.dtype:
    """Convert precision string to torch dtype"""
    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if precision not in mapping:
        raise ValueError(f"Unknown precision: {precision}. Must be one of {list(mapping.keys())}")
    return mapping[precision]


def load_json_input(input_file: str) -> List[Dict[str, Any]]:
    """Load kernel entries from JSON file"""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON input must be a list of kernel entries")

    # Validate entries
    required_keys = {"kernel_id", "task_id", "ref_src", "custom_src"}
    for i, entry in enumerate(data):
        missing = required_keys - set(entry.keys())
        if missing:
            raise ValueError(f"Entry {i} missing required keys: {missing}")

    return data


def evaluate_single_kernel(
    entry: Dict[str, Any],
    device: int,
    config: EvalFromJsonConfig,
) -> SingleKernelResult:
    """Evaluate a single kernel entry"""
    kernel_id = entry["kernel_id"]
    task_id = entry["task_id"]
    ref_src = entry["ref_src"]
    custom_src = entry["custom_src"]

    result = SingleKernelResult(kernel_id=kernel_id, task_id=task_id)

    start_time = time.time()

    try:
        # Create build directory for this kernel
        build_dir = os.path.join(
            REPO_TOP_DIR,
            "runs",
            "json_eval_cache",
            f"kernel_{kernel_id}"
        )
        os.makedirs(build_dir, exist_ok=True)

        precision_dtype = get_precision_dtype(config.precision)

        # Run evaluation
        exec_result: KernelExecResult = eval_kernel_against_ref(
            original_model_src=ref_src,
            custom_model_src=custom_src,
            seed_num=42,
            num_correct_trials=config.num_correct_trials,
            num_perf_trials=config.num_perf_trials,
            measure_performance=config.measure_performance,
            timing_method=config.timing_method,
            verbose=config.verbose,
            build_dir=build_dir,
            device=device,
            backend=config.backend,
            precision=precision_dtype,
            check_for_excessive_speedup=config.check_excessive_speedup,
        )

        if exec_result is None:
            result.error = "Evaluation returned None (possible lock error, retry recommended)"
            return result

        # Extract results
        result.compiled = exec_result.compiled
        result.correctness = exec_result.correctness
        result.runtime = exec_result.runtime
        result.ref_runtime = exec_result.ref_runtime
        result.runtime_stats = exec_result.runtime_stats
        result.metadata = check_metadata_serializable_all_types(exec_result.metadata)

        # Calculate speedup
        if result.runtime > 0 and result.ref_runtime > 0:
            result.speedup = result.ref_runtime / result.runtime

    except Exception as e:
        result.error = f"{type(e).__name__}: {str(e)}"
        if config.verbose:
            import traceback
            traceback.print_exc()

    result.eval_time = time.time() - start_time
    return result


def worker_process(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    device: int,
    config_dict: Dict[str, Any],
):
    """Worker process for parallel evaluation"""
    # Set CUDA device
    torch.cuda.set_device(device)

    # Reconstruct config
    config = EvalFromJsonConfig()
    for key, value in config_dict.items():
        setattr(config, key, value)

    while True:
        item = task_queue.get()
        if item is None:  # Poison pill
            break

        idx, entry = item
        result = evaluate_single_kernel(entry, device, config)
        result_queue.put((idx, result))


def batch_eval_parallel(
    entries: List[Dict[str, Any]],
    config: EvalFromJsonConfig,
) -> List[SingleKernelResult]:
    """Evaluate kernels in parallel across multiple GPUs"""
    num_gpus = min(config.num_gpus, torch.cuda.device_count())
    if num_gpus == 0:
        raise RuntimeError("No CUDA devices available")

    print(f"[Eval] Using {num_gpus} GPU(s) for parallel evaluation")

    # Create queues
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # Convert config to dict for pickling
    config_dict = {
        "num_correct_trials": config.num_correct_trials,
        "num_perf_trials": config.num_perf_trials,
        "measure_performance": config.measure_performance,
        "timing_method": config.timing_method,
        "backend": config.backend,
        "precision": config.precision,
        "verbose": config.verbose,
        "check_excessive_speedup": config.check_excessive_speedup,
    }

    # Start worker processes
    workers = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=worker_process,
            args=(task_queue, result_queue, gpu_id, config_dict),
        )
        p.start()
        workers.append(p)

    # Add tasks to queue
    for idx, entry in enumerate(entries):
        task_queue.put((idx, entry))

    # Add poison pills
    for _ in workers:
        task_queue.put(None)

    # Collect results with progress bar
    results = [None] * len(entries)
    with tqdm(total=len(entries), desc="Evaluating kernels") as pbar:
        for _ in range(len(entries)):
            idx, result = result_queue.get()
            results[idx] = result

            # Update progress bar with status
            status = "✓" if result.correctness else ("⚠" if result.compiled else "✗")
            pbar.set_postfix_str(f"Last: {result.kernel_id} [{status}]")
            pbar.update(1)

    # Wait for workers to finish
    for p in workers:
        p.join()

    return results


def batch_eval_sequential(
    entries: List[Dict[str, Any]],
    config: EvalFromJsonConfig,
) -> List[SingleKernelResult]:
    """Evaluate kernels sequentially on a single GPU"""
    device = 0
    results = []

    for entry in tqdm(entries, desc="Evaluating kernels"):
        result = evaluate_single_kernel(entry, device, config)
        results.append(result)

        # Print status
        status = "✓ PASS" if result.correctness else ("⚠ COMPILED" if result.compiled else "✗ FAIL")
        if config.verbose:
            print(f"  {result.kernel_id}: {status}")

    return results


def compute_aggregate_stats(
    results: List[SingleKernelResult],
    config: EvalFromJsonConfig,
) -> BatchEvalResults:
    """Compute aggregate statistics from individual results"""
    batch_results = BatchEvalResults()
    batch_results.total_kernels = len(results)
    batch_results.timestamp = datetime.now().isoformat()

    # Store config
    batch_results.config = {
        "input_file": config.input_file,
        "backend": config.backend,
        "precision": config.precision,
        "num_correct_trials": config.num_correct_trials,
        "num_perf_trials": config.num_perf_trials,
        "measure_performance": config.measure_performance,
    }

    total_eval_time = 0.0

    for result in results:
        # Store individual result
        batch_results.results.append(asdict(result))

        # Count statistics
        if result.compiled:
            batch_results.compiled_count += 1
        if result.correctness:
            batch_results.correct_count += 1
            if result.speedup > 1.0:
                batch_results.faster_than_ref_count += 1
            if result.speedup > 2.0:
                batch_results.fast_2x_count += 1

        total_eval_time += result.eval_time

    # Compute rates
    n = batch_results.total_kernels
    if n > 0:
        batch_results.compile_rate = batch_results.compiled_count / n
        batch_results.correctness_rate = batch_results.correct_count / n
        batch_results.fast_1_rate = batch_results.faster_than_ref_count / n
        batch_results.fast_2_rate = batch_results.fast_2x_count / n
        batch_results.total_eval_time = total_eval_time
        batch_results.avg_eval_time = total_eval_time / n

    return batch_results


def print_summary(batch_results: BatchEvalResults):
    """Print evaluation summary"""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total kernels evaluated: {batch_results.total_kernels}")
    print(f"Timestamp: {batch_results.timestamp}")
    print("-" * 60)
    print(f"Compiled:              {batch_results.compiled_count:4d} ({batch_results.compile_rate*100:5.1f}%)")
    print(f"Correct:               {batch_results.correct_count:4d} ({batch_results.correctness_rate*100:5.1f}%)")
    print(f"Faster than reference: {batch_results.faster_than_ref_count:4d} ({batch_results.fast_1_rate*100:5.1f}%)")
    print(f"2x+ faster:            {batch_results.fast_2x_count:4d} ({batch_results.fast_2_rate*100:5.1f}%)")
    print("-" * 60)
    print(f"Total eval time:       {batch_results.total_eval_time:.1f}s")
    print(f"Avg eval time/kernel:  {batch_results.avg_eval_time:.2f}s")
    print("=" * 60)

    # Print table of results
    print("\nDETAILED RESULTS:")
    print("-" * 80)
    print(f"{'Kernel ID':<20} {'Task':<10} {'Compiled':<10} {'Correct':<10} {'Speedup':<10} {'Error':<20}")
    print("-" * 80)

    for r in batch_results.results:
        speedup_str = f"{r['speedup']:.2f}x" if r['speedup'] > 0 else "N/A"
        error_str = r['error'][:17] + "..." if r['error'] and len(r['error']) > 20 else (r['error'] or "")
        print(f"{r['kernel_id']:<20} {r['task_id']:<10} {str(r['compiled']):<10} {str(r['correctness']):<10} {speedup_str:<10} {error_str:<20}")

    print("-" * 80)


def save_results(batch_results: BatchEvalResults, output_file: str):
    """Save results to JSON file"""
    output_path = os.path.join(REPO_TOP_DIR, output_file)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(batch_results), f, indent=2, default=str)

    print(f"\n[Eval] Results saved to: {output_path}")


@pydra.main(EvalFromJsonConfig)
def main(config: EvalFromJsonConfig):
    """Main entry point"""
    print(f"[Eval] Loading kernels from: {config.input_file}")

    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU required for evaluation.")

    print(f"[Eval] Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Load input
    input_path = os.path.join(REPO_TOP_DIR, config.input_file)
    if not os.path.exists(input_path):
        input_path = config.input_file  # Try absolute path

    entries = load_json_input(input_path)
    print(f"[Eval] Loaded {len(entries)} kernel entries")

    # Apply subset if specified
    start_idx, end_idx = config.subset
    if start_idx is not None or end_idx is not None:
        start_idx = start_idx or 0
        end_idx = end_idx or len(entries)
        entries = entries[start_idx:end_idx]
        print(f"[Eval] Evaluating subset [{start_idx}:{end_idx}] ({len(entries)} kernels)")

    # Run evaluation
    print(f"\n[Eval] Configuration:")
    print(f"  Backend: {config.backend}")
    print(f"  Precision: {config.precision}")
    print(f"  Correctness trials: {config.num_correct_trials}")
    print(f"  Performance trials: {config.num_perf_trials}")
    print(f"  Measure performance: {config.measure_performance}")
    print()

    start_time = time.time()

    if config.num_gpus > 1 and torch.cuda.device_count() > 1:
        results = batch_eval_parallel(entries, config)
    else:
        results = batch_eval_sequential(entries, config)

    total_time = time.time() - start_time
    print(f"\n[Eval] Total evaluation time: {total_time:.1f}s")

    # Compute aggregate statistics
    batch_results = compute_aggregate_stats(results, config)

    # Print summary
    print_summary(batch_results)

    # Save results
    save_results(batch_results, config.output_file)

    return batch_results


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
