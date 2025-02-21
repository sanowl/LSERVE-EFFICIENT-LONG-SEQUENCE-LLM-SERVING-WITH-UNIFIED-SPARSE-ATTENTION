import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import wandb
import torch
import numpy as np
from collections import defaultdict

class MetricsLogger:
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "logs",
        use_wandb: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project="lserve",
                name=experiment_name,
                config=config
            )
        
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        
        # Initialize metric files
        self.metric_file = self.log_dir / "metrics.jsonl"
        self.summary_file = self.log_dir / "summary.json"
        
    def log_metric(
        self,
        name: str,
        value: float,
        step: int,
        commit: bool = True
    ):
        """Log a single metric"""
        timestamp = time.time()
        
        metric_data = {
            "name": name,
            "value": value,
            "step": step,
            "timestamp": timestamp
        }
        
        # Store locally
        self.metrics[name].append((step, value))
        
        # Write to file
        with self.metric_file.open("a") as f:
            f.write(json.dumps(metric_data) + "\n")
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({name: value}, step=step, commit=commit)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = ""
    ):
        """Log multiple metrics"""
        for name, value in metrics.items():
            metric_name = f"{prefix}{name}" if prefix else name
            self.log_metric(metric_name, value, step, commit=False)
            
        if self.use_wandb:
            wandb.log(metrics, step=step)
    
    def log_histogram(
        self,
        name: str,
        values: torch.Tensor,
        step: int
    ):
        """Log value distribution as histogram"""
        if self.use_wandb:
            wandb.log({name: wandb.Histogram(values.detach().cpu().numpy())}, step=step)
    
    def log_attention_patterns(
        self,
        name: str,
        attention_weights: torch.Tensor,
        step: int
    ):
        """Log attention pattern visualization"""
        if self.use_wandb:
            attention_cpu = attention_weights.detach().cpu()
            wandb.log({
                name: wandb.Image(attention_cpu, caption=f"Step {step}")
            }, step=step)
    
    def compute_summary_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics for all metrics"""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            steps, metric_values = zip(*values)
            metric_values = np.array(metric_values)
            
            summary[metric_name] = {
                "mean": float(np.mean(metric_values)),
                "std": float(np.std(metric_values)),
                "min": float(np.min(metric_values)),
                "max": float(np.max(metric_values)),
                "last": float(metric_values[-1])
            }
        
        # Save summary
        with self.summary_file.open("w") as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def close(self):
        """Compute final statistics and close logger"""
        summary = self.compute_summary_stats()
        
        if self.use_wandb:
            wandb.log({"summary": summary})
            wandb.finish()
        
        return summary
