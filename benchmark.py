import torch
import argparse
import yaml
import time
import numpy as np
from pathlib import Path
from src.serving.model_server import ModelServer
from src.config.config_validation import validate_config_file
from src.utils.model_utils import initialize_model
from src.utils.profiler import ModelProfiler
import matplotlib.pyplot as plt
from typing import List, Dict
import pandas as pd

def run_latency_benchmark(
    model: ModelServer,
    sequence_lengths: List[int],
    batch_sizes: List[int],
    num_runs: int = 100
):
    results = []
    
    for seq_len in sequence_lengths:
        for batch_size in batch_sizes:
            # Create dummy input
            input_ids = torch.randint(
                0, 50000,
                (batch_size, seq_len),
                device="cuda"
            )
            attention_mask = torch.ones_like(input_ids)
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model.process_sequence(
                        input_ids,
                        sequence_id="benchmark",
                        attention_mask=attention_mask
                    )
            
            # Benchmark runs
            latencies = []
            
            for run in range(num_runs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                with torch.no_grad():
                    _ = model.process_sequence(
                        input_ids,
                        sequence_id=f"benchmark_{run}",
                        attention_mask=attention_mask
                    )
                end.record()
                
                torch.cuda.synchronize()
                latencies.append(start.elapsed_time(end))
            
            results.append({
                "sequence_length": seq_len,
                "batch_size": batch_size,
                "mean_latency": np.mean(latencies),
                "p90_latency": np.percentile(latencies, 90),
                "p99_latency": np.percentile(latencies, 99),
                "throughput": batch_size * seq_len / (np.mean(latencies) / 1000)
            })
    
    return pd.DataFrame(results)

def plot_results(df: pd.DataFrame, output_dir: Path):
    # Latency vs Sequence Length
    plt.figure(figsize=(10, 6))
    for batch_size in df["batch_size"].unique():
        data = df[df["batch_size"] == batch_size]
        plt.plot(
            data["sequence_length"],
            data["mean_latency"],
            label=f"Batch Size {batch_size}"
        )
    plt.xlabel("Sequence Length")
    plt.ylabel("Latency (ms)")
    plt.title("Latency vs Sequence Length")
    plt.legend()
    plt.savefig(output_dir / "latency_vs_seqlen.png")
    
    # Throughput vs Batch Size
    plt.figure(figsize=(10, 6))
    for seq_len in df["sequence_length"].unique():
        data = df[df["sequence_length"] == seq_len]
        plt.plot(
            data["batch_size"],
            data["throughput"],
            label=f"Seq Len {seq_len}"
        )
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (tokens/sec)")
    plt.title("Throughput vs Batch Size")
    plt.legend()
    plt.savefig(output_dir / "throughput_vs_batchsize.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="benchmark_results")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration and initialize model
    config = validate_config_file(args.config)
    model = initialize_model(config["attention"])
    model.cuda()
    
    # Run benchmarks
    sequence_lengths = [512, 1024, 2048, 4096, 8192, 16384]
    batch_sizes = [1, 2, 4, 8, 16, 32]
    
    results_df = run_latency_benchmark(
        model,
        sequence_lengths,
        batch_sizes
    )
    
    # Save results
    results_df.to_csv(output_dir / "benchmark_results.csv")
    plot_results(results_df, output_dir)
    
    # Profile memory usage
    profiler = ModelProfiler(model)
    for seq_len in [1024, 4096, 16384]:
        input_tensor = torch.randint(
            0, 50000,
            (1, seq_len, config["attention"].hidden_size),
            device="cuda"
        )
        memory_stats = profiler.profile_memory(input_tensor)
        print(f"\nMemory stats for sequence length {seq_len}:")
        for k, v in memory_stats.items():
            print(f"{k}: {v:.2f} MB")

if __name__ == "__main__":
    main()
