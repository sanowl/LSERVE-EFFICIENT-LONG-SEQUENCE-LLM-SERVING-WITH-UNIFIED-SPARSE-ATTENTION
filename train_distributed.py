import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse
from src.training.distributed_trainer import DistributedTrainer
from src.utils.metrics_logger import MetricsLogger
from src.data.data_processor import DataProcessor
from src.utils.model_utils import initialize_model
from src.config.config_validation import validate_config_file
from transformers import AutoTokenizer
from datasets import load_dataset

def setup_distributed(rank: int, world_size: int):
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )

def cleanup_distributed():
    """Cleanup distributed training environment"""
    dist.destroy_process_group()

def train(rank, world_size, args):
    # Setup distributed training
    setup_distributed(rank, world_size)
    
    # Load and validate config
    config = validate_config_file(args.config_path)
    
    # Initialize tokenizer and data processor
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_processor = DataProcessor(tokenizer)
    
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        dataset["train"],
        num_replicas=world_size,
        rank=rank
    )
    
    eval_sampler = DistributedSampler(
        dataset["validation"],
        num_replicas=world_size,
        rank=rank
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        dataset["train"],
        batch_size=config["training"].batch_size,
        sampler=train_sampler,
        num_workers=4
    )
    
    eval_loader = torch.utils.data.DataLoader(
        dataset["validation"],
        batch_size=config["training"].batch_size,
        sampler=eval_sampler,
        num_workers=4
    )
    
    # Initialize model
    model = initialize_model(config["attention"])
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"].learning_rate,
        weight_decay=config["training"].weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["training"].learning_rate,
        total_steps=config["training"].max_steps,
        pct_start=config["training"].warmup_steps / config["training"].max_steps
    )
    
    # Initialize metrics logger (only on main process)
    metrics_logger = None
    if rank == 0:
        metrics_logger = MetricsLogger(
            experiment_name=args.experiment_name,
            config=config
        )
    
    # Initialize distributed trainer
    trainer = DistributedTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        rank=rank,
        world_size=world_size,
        local_rank=rank,
        data_processor=data_processor,
        metrics_logger=metrics_logger
    )
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint_path:
        start_epoch = trainer.load_checkpoint(args.checkpoint_path)
    
    # Training loop
    best_eval_loss = float('inf')
    for epoch in range(start_epoch, config["training"].max_steps):
        # Train epoch
        train_metrics = trainer.train_epoch(
            train_loader,
            epoch,
            grad_accum_steps=config["training"].gradient_accumulation
        )
        
        # Evaluate
        if rank == 0:  # Only evaluate on main process
            eval_metrics = trainer.evaluate(eval_loader)
            
            # Log metrics
            metrics_logger.log_metrics(
                {**train_metrics, **eval_metrics},
                step=epoch
            )
            
            # Save checkpoint if best
            if eval_metrics["loss"] < best_eval_loss:
                best_eval_loss = eval_metrics["loss"]
                trainer.save_checkpoint(
                    os.path.join(args.output_dir, "best_model.pt"),
                    epoch,
                    is_best=True
                )
        
        # Save regular checkpoint
        if rank == 0 and epoch % args.save_frequency == 0:
            trainer.save_checkpoint(
                os.path.join(args.output_dir, f"checkpoint_{epoch}.pt"),
                epoch
            )
    
    # Cleanup
    trainer.cleanup()
    cleanup_distributed()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--save_frequency", type=int, default=10)
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Launch distributed training
    torch.multiprocessing.spawn(
        train,
        args=(args.num_gpus, args),
        nprocs=args.num_gpus,
        join=True
    )

if __name__ == "__main__":
    main()
