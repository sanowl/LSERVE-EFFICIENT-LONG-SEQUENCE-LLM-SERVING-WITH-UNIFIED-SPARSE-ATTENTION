import torch
import argparse
import logging
import os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
from src.utils.model_utils import initialize_model, save_model_checkpoint, load_model_checkpoint
from src.data.data_processor import DataProcessor
from src.training.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator
from config.model_config import ModelConfig
from datasets import load_dataset
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="LSERVE: Long Sequence LLM Serving")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--overlap_size", type=int, default=128)
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    
    # System configuration
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    
    # Paths and logging
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=1000)
    
    return parser.parse_args()

def setup_logging(args):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    
    # Initialize wandb
    wandb.init(
        project="lserve",
        config=vars(args),
        name=f"lserve_{args.model_name}_{args.hidden_size}"
    )

def setup_data(args, tokenizer):
    # Load dataset (example using WikiText)
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    
    # Initialize data processor
    data_processor = DataProcessor(tokenizer)
    
    # Process datasets
    def process_batch(examples):
        return data_processor.preprocess_text(examples["text"])
    
    train_dataset = dataset["train"].map(
        process_batch,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    eval_dataset = dataset["validation"].map(
        process_batch,
        batched=True,
        remove_columns=dataset["validation"].column_names
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    return train_loader, eval_loader

def main():
    # Parse arguments and setup
    args = parse_args()
    setup_logging(args)
    torch.manual_seed(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize tokenizer and model config
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_config = ModelConfig(
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        chunk_size=args.chunk_size,
        overlap_size=args.overlap_size
    )
    
    # Initialize model and move to device
    model = initialize_model(model_config)
    model = model.to(args.device)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=args.max_steps,
        pct_start=args.warmup_steps / args.max_steps
    )
    
    # Setup data
    train_loader, eval_loader = setup_data(args, tokenizer)
    
    # Initialize trainer and evaluator
    trainer = ModelTrainer(
        model=model,
        data_processor=DataProcessor(tokenizer),
        optimizer=optimizer,
        device=args.device
    )
    
    evaluator = ModelEvaluator(
        model=model,
        device=args.device
    )
    
    # Training loop
    best_eval_loss = float('inf')
    global_step = 0
    
    for epoch in range(args.num_epochs):
        logging.info(f"Starting epoch {epoch + 1}/{args.num_epochs}")
        
        # Training
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            loss = trainer.train_step(batch)
            
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            if global_step % args.log_steps == 0:
                logging.info(f"Step {global_step}: loss = {loss:.4f}")
                wandb.log({
                    "train/loss": loss,
                    "train/lr": scheduler.get_last_lr()[0]
                }, step=global_step)
            
            if global_step % args.save_steps == 0:
                # Evaluation
                eval_metrics = evaluator.evaluate(eval_loader)
                wandb.log({
                    "eval/loss": eval_metrics["loss"],
                    "eval/perplexity": eval_metrics["perplexity"],
                    "eval/throughput": eval_metrics["throughput"]
                }, step=global_step)
                
                # Save checkpoint if best
                if eval_metrics["loss"] < best_eval_loss:
                    best_eval_loss = eval_metrics["loss"]
                    save_path = os.path.join(
                        args.checkpoint_dir,
                        f"checkpoint_best.pt"
                    )
                    save_model_checkpoint(model, save_path, optimizer)
            
            global_step += 1
            if global_step >= args.max_steps:
                break
                
        # Save epoch checkpoint
        save_path = os.path.join(
            args.checkpoint_dir,
            f"checkpoint_epoch_{epoch+1}.pt"
        )
        save_model_checkpoint(model, save_path, optimizer)
    
    # Final evaluation
    final_metrics = evaluator.evaluate(eval_loader)
    logging.info("Training completed. Final metrics:")
    logging.info(f"Loss: {final_metrics['loss']:.4f}")
    logging.info(f"Perplexity: {final_metrics['perplexity']:.4f}")
    logging.info(f"Throughput: {final_metrics['throughput']:.2f} tokens/sec")
    
    wandb.finish()

if __name__ == "__main__":
    main()
