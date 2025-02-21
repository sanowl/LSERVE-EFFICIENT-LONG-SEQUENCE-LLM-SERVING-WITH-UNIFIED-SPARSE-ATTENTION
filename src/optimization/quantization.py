import torch
from typing import Dict, Optional, Tuple
import numpy as np
from torch.quantization import quantize_dynamic
from torch.ao.quantization import get_default_qconfig
import logging

class ModelQuantizer:
    def __init__(
        self,
        model: torch.nn.Module,
        calibration_data: Optional[torch.Tensor] = None,
        quantization_bits: int = 8
    ):
        self.model = model
        self.calibration_data = calibration_data
        self.bits = quantization_bits
        self.original_state_dict = None
        
    def prepare_for_quantization(self):
        """Prepare model for quantization"""
        self.original_state_dict = self.model.state_dict()
        
        # Set quantization configuration
        self.model.qconfig = get_default_qconfig('fbgemm')
        
        # Fuse modules where possible
        self.model = torch.quantization.fuse_modules(
            self.model,
            [['query', 'key', 'value']]
        )
        
    def calibrate(self, num_batches: int = 100):
        """Calibrate quantization using calibration data"""
        if self.calibration_data is None:
            raise ValueError("Calibration data not provided")
            
        self.model.eval()
        with torch.no_grad():
            for i in range(num_batches):
                _ = self.model(self.calibration_data[i:i+1])
    
    def quantize_model(self) -> Tuple[torch.nn.Module, Dict[str, float]]:
        """Quantize the model and measure performance impact"""
        # Measure pre-quantization performance
        pre_metrics = self._measure_performance()
        
        # Quantize model
        quantized_model = quantize_dynamic(
            self.model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        # Measure post-quantization performance
        self.model = quantized_model
        post_metrics = self._measure_performance()
        
        # Compare metrics
        comparison = {
            "memory_reduction": pre_metrics["memory_mb"] / post_metrics["memory_mb"],
            "speed_up": pre_metrics["latency_ms"] / post_metrics["latency_ms"],
            "size_reduction": pre_metrics["model_size_mb"] / post_metrics["model_size_mb"]
        }
        
        logging.info("Quantization Results:")
        for metric, value in comparison.items():
            logging.info(f"  {metric}: {value:.2f}x")
        
        return quantized_model, comparison
    
    def _measure_performance(self) -> Dict[str, float]:
        """Measure model performance metrics"""
        # Memory usage
        torch.cuda.reset_peak_memory_stats()
        
        # Latency measurement
        latencies = []
        for _ in range(100):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            with torch.no_grad():
                _ = self.model(self.calibration_data[:1])
            end_event.record()
            
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))
        
        return {
            "memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
            "latency_ms": np.mean(latencies),
            "model_size_mb": sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024**2
        }
    
    def restore_original_model(self):
        """Restore model to pre-quantization state"""
        if self.original_state_dict is not None:
            self.model.load_state_dict(self.original_state_dict)
