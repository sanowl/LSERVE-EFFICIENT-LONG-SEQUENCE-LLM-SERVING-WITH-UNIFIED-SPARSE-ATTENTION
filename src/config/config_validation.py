from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
import yaml
import json
from pathlib import Path

@dataclass
class AttentionConfig:
    hidden_size: int
    num_attention_heads: int
    attention_dropout: float
    window_size: int
    global_tokens: int
    sparsity_factor: float
    lsh_num_buckets: int
    lsh_hash_size: int

@dataclass
class ChunkingConfig:
    min_chunk_size: int
    max_chunk_size: int
    chunk_growth_factor: float
    overlap_size: int

@dataclass
class MemoryConfig:
    max_memory_size: int
    cache_size: int
    prefetch_window: int
    prediction_threshold: float

@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    warmup_steps: int
    max_steps: int
    gradient_accumulation: int
    weight_decay: float
    max_grad_norm: float

class ConfigValidator:
    def __init__(self):
        self.schema = {
            "attention": {
                "hidden_size": (64, 4096),
                "num_attention_heads": (1, 64),
                "attention_dropout": (0.0, 1.0),
                "window_size": (32, 4096),
                "global_tokens": (1, 512),
                "sparsity_factor": (0.0, 1.0),
                "lsh_num_buckets": (8, 128),
                "lsh_hash_size": (8, 64)
            },
            "chunking": {
                "min_chunk_size": (64, 2048),
                "max_chunk_size": (512, 8192),
                "chunk_growth_factor": (1.0, 4.0),
                "overlap_size": (0, 1024)
            },
            "memory": {
                "max_memory_size": (100, 1000000),
                "cache_size": (100, 1000000),
                "prefetch_window": (32, 2048),
                "prediction_threshold": (0.0, 1.0)
            },
            "training": {
                "batch_size": (1, 512),
                "learning_rate": (1e-6, 1.0),
                "warmup_steps": (0, 100000),
                "max_steps": (1000, 1000000),
                "gradient_accumulation": (1, 64),
                "weight_decay": (0.0, 1.0),
                "max_grad_norm": (0.0, 100.0)
            }
        }
    
    def validate_value(
        self,
        name: str,
        value: Union[int, float],
        bounds: tuple
    ) -> Optional[str]:
        """Validate a single configuration value"""
        min_val, max_val = bounds
        if not isinstance(value, (int, float)):
            return f"{name} must be a number"
        if value < min_val or value > max_val:
            return f"{name} must be between {min_val} and {max_val}"
        return None

    def validate_config_section(
        self,
        section_name: str,
        config: Dict[str, Any]
    ) -> List[str]:
        """Validate a configuration section"""
        errors = []
        schema_section = self.schema.get(section_name)
        
        if not schema_section:
            errors.append(f"Unknown configuration section: {section_name}")
            return errors
        
        for param_name, bounds in schema_section.items():
            if param_name not in config:
                errors.append(f"Missing parameter '{param_name}' in {section_name} config")
                continue
                
            error = self.validate_value(
                f"{section_name}.{param_name}",
                config[param_name],
                bounds
            )
            if error:
                errors.append(error)
        
        return errors

    def validate_full_config(self, config: Dict[str, Dict[str, Any]]) -> List[str]:
        """Validate complete configuration"""
        errors = []
        
        for section_name in self.schema:
            if section_name not in config:
                errors.append(f"Missing configuration section: {section_name}")
                continue
                
            section_errors = self.validate_config_section(
                section_name,
                config[section_name]
            )
            errors.extend(section_errors)
        
        return errors

    @staticmethod
    def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with config_path.open('r') as f:
            if config_path.suffix == '.yaml':
                config = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                config = json.load(f)
            else:
                raise ValueError("Config file must be .yaml or .json")
        
        return config

    def validate_and_load(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load and validate configuration"""
        config = self.load_config(config_path)
        errors = self.validate_full_config(config)
        
        if errors:
            error_msg = "\n".join(f"- {error}" for error in errors)
            raise ValueError(f"Configuration validation failed:\n{error_msg}")
        
        return config

    def create_dataclass_configs(
        self,
        config: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Convert dictionary config to dataclass instances"""
        return {
            "attention": AttentionConfig(**config["attention"]),
            "chunking": ChunkingConfig(**config["chunking"]),
            "memory": MemoryConfig(**config["memory"]),
            "training": TrainingConfig(**config["training"])
        }

def validate_config_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Utility function to validate and load config file"""
    validator = ConfigValidator()
    config = validator.validate_and_load(config_path)
    return validator.create_dataclass_configs(config)
