import os
import argparse
from contextlib import nullcontext
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import (initialize_model, initialize_optimizer_and_scheduler, 
                     save_model, load_model_from_checkpoint, setup_wandb, ensure_dirs)
from transformers import T5TokenizerFast
from load_data_old import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def is_bf16_supported():
    """Check whether the current environment supports bfloat16"""
    if DEVICE.type != 'cuda':
        return False
    return hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()

def precision_context(args):
    """Return the autocast context based on the configuration"""
    if getattr(args, 'use_bf16', False) and DEVICE.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return nullcontext()

def get_args():
    '''Training argument configuration'''
    parser = argparse.ArgumentParser(description='T5 training for Text-to-SQL')
    
    # Model configuration
    parser.add_argument('--finetune', action='store_true', 
                       help="Fine-tune pretrained T5 (otherwise train from scratch)")
    parser.add_argument('--bf16', action='store_true',
                       help="Use bfloat16 mixed precision when supported")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"])
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help="Learning rate (suggested: 1e-4 for finetune, 3e-4 for scratch)")
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--scheduler_type', type=str, default="linear", 
                       choices=["none", "cosine", "linear"])
    parser.add_argument('--num_warmup_epochs', type=int, default=1)
    parser.add_argument('--max_n_epochs', type=int, default=25)
    parser.add_argument('--patience_epochs', type=int, default=5)

    # Experiment tracking
    parser.add_argument('--use_wandb', action='store_true',
                       help="Use Weights & Biases for experiment tracking")
    parser.add_argument('--experiment_name', type=str, default='baseline',
                       help="Name for this experiment run")
    parser.add_argument('--force_sql_eval', action='store_true',
                       help="Force SQL-based evaluation each epoch even when not fine-tuning")
    parser.add_argument('--sql_eval_every', type=int, default=None,
                        help="If set, run SQL/F1 evaluation every N epochs. "
                             "Use 0 or a negative value to skip until final evaluation.")
    parser.add_argument('--test_only', action='store_true',
                        help="Skip training and only run test inference using saved checkpoints.")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--test_batch_size', type=int, default=16)

    return parser.parse_args()

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    """Main training loop"""
    ensure_dirs()
    best_f1 = -1
    best_loss = float('inf')
    epochs_since_improvement = 0
    sql_metrics_observed = False
    
    default_sql_eval_interval = 1 if (args.finetune or args.force_sql_eval) else 5
    if args.sql_eval_every is None:
        sql_eval_interval = default_sql_eval_interval
    elif args.sql_eval_every <= 0:
        sql_eval_interval = None
    else:
        sql_eval_interval = args.sql_eval_every
    
    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    args.checkpoint_dir = checkpoint_dir
    
    # Evaluation file paths
    gt_sql_path = os.path.join('data', 'dev.sql')
    gt_record_path = os.path.join('records', 'ground_truth_dev.pkl')
    model_sql_path = os.path.join('results', f't5_{model_type}_{args.experiment_name}_dev.sql')
    model_record_path = os.path.join('records', f't5_{model_type}_{args.experiment_name}_dev.pkl')
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(model_sql_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_record_path), exist_ok=True)
    
    print(f"Starting training for {args.max_n_epochs} epochs...")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    print(f"Results will be saved with prefix: t5_{model_type}_{args.experiment_name}")
    if sql_eval_interval is None:
        print("SQL/F1 evaluation during training is disabled (will only run after training).")
    elif sql_eval_interval == 1:
        print("SQL/F1 evaluation will run after every epoch.")
    else:
        print(f"SQL/F1 evaluation will run every {sql_eval_interval} epochs.")
    
    for epoch in range(args.max_n_epochs):
        # Train for one epoch
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch+1}/{args.max_n_epochs}: Train Loss = {tr_loss:.4f}")

        # Evaluation (run SQL eval according to configuration)
        should_run_sql_eval = (
            sql_eval_interval is not None and (epoch + 1) % sql_eval_interval == 0
        )
        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path,
            run_sql_eval=should_run_sql_eval
        )
        
        if record_f1 is not None:
            sql_metrics_observed = True
            print(f"Epoch {epoch+1}: Dev Loss = {eval_loss:.4f}, Record F1 = {record_f1:.4f}, "
                  f"Record EM = {record_em:.4f}, SQL EM = {sql_em:.4f}")
            print(f"Epoch {epoch+1}: Error Rate = {error_rate*100:.2f}%")
        else:
            print(f"Epoch {epoch+1}: Dev Loss = {eval_loss:.4f} (SQL metrics skipped)")

        # Log to Weights & Biases
        if args.use_wandb:
            metrics_to_log = {
                'epoch': epoch + 1,
                'train/loss': tr_loss,
                'dev/loss': eval_loss,
            }
            if record_f1 is not None:
                metrics_to_log.update({
                    'dev/record_f1': record_f1,
                    'dev/record_em': record_em,
                    'dev/sql_em': sql_em,
                    'dev/error_rate': error_rate,
                })
            wandb.log(metrics_to_log)

        # Early stopping logic
        if record_f1 is not None:
            if record_f1 > best_f1:
                best_f1 = record_f1
                epochs_since_improvement = 0
                save_model(checkpoint_dir, model, best=True)
                print(f"New best model with F1 = {best_f1:.4f}")
            else:
                epochs_since_improvement += 1
        else:
            if eval_loss < best_loss:
                best_loss = eval_loss
                epochs_since_improvement = 0
                save_model(checkpoint_dir, model, best=True)
                print(f"New best model with Dev Loss = {best_loss:.4f}")
            else:
                epochs_since_improvement += 1

        # Save the latest model
        save_model(checkpoint_dir, model, best=False)

        # Check the early stopping condition
        if epochs_since_improvement >= args.patience_epochs:
            print(f"Early stopping after {epoch + 1} epochs without improvement")
            break

    if sql_metrics_observed:
        print(f"Training completed. Best F1: {best_f1:.4f}")
    else:
        print(f"Training completed. Best Dev Loss: {best_loss:.4f}")
    return best_f1

def train_epoch(args, model, train_loader, optimizer, scheduler):
    """Train for a single epoch"""
    model.train()
    total_loss = 0
    total_batches = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch in progress_bar:
        # The standard collate function returns five values
        encoder_input, encoder_mask, decoder_input, decoder_targets, _ = batch
        
        # Move tensors to the device
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        # Forward pass
        with precision_context(args):
            outputs = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
                labels=decoder_targets
            )
            loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Statistics
        total_loss += loss.item()
        total_batches += 1
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / total_batches if total_batches > 0 else 0

def eval_epoch(args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path,
               run_sql_eval=True):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    total_batches = 0
    all_generated_queries = []
    
    if run_sql_eval:
        tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    else:
        tokenizer = None
    
    with torch.no_grad():
        # Compute validation loss
        for batch in tqdm(dev_loader, desc="Evaluating Loss"):
            # The standard collate function returns five values
            encoder_input, encoder_mask, decoder_input, decoder_targets, _ = batch
            
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)

            with precision_context(args):
                outputs = model(
                    input_ids=encoder_input,
                    attention_mask=encoder_mask,
                    decoder_input_ids=decoder_input,
                    labels=decoder_targets
                )
                total_loss += outputs.loss.item()
            total_batches += 1
        
        avg_loss = total_loss / total_batches if total_batches > 0 else 0
        
        if not run_sql_eval:
            print("Skipping SQL query generation and metrics for this evaluation run.")
            return avg_loss, None, None, None, None
        
        # Generate SQL queries
        print("Generating SQL queries for evaluation...")
        for batch in tqdm(dev_loader, desc="Generating SQL"):
            # For generation we only need the first three items: encoder_input, encoder_mask, initial_decoder_input
            # normal_collate_fn returns five items, so unpack accordingly
            encoder_input, encoder_mask, _, _, initial_decoder_input = batch
            
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            
            # Generate using beam search
            with precision_context(args):
                generated_ids = model.generate(
                    input_ids=encoder_input,
                    attention_mask=encoder_mask,
                    max_length=256,
                    num_beams=10,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    decoder_start_token_id=tokenizer.pad_token_id,
                    no_repeat_ngram_size=4,
                    repetition_penalty=1.2,
                    renormalize_logits=True,
                )
            
            # Decode generated SQL
            for gen_ids in generated_ids:
                sql_query = tokenizer.decode(gen_ids, skip_special_tokens=True)
                all_generated_queries.append(sql_query)
    
    # Save outputs and compute metrics
    print("Saving generated queries and computing metrics...")
    save_queries_and_records(all_generated_queries, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_path, model_sql_path, gt_record_path, model_record_path
    )
    
    # Compute error rate
    error_rate = sum(1 for msg in error_msgs if msg and msg.strip()) / len(error_msgs) if error_msgs else 0
    
    return avg_loss, record_f1, record_em, sql_em, error_rate

def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    """Inference on the test set"""
    model.eval()
    all_generated_queries = []
    
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    print("Generating test set predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Inference"):
            # Test collate function returns three values
            encoder_input, encoder_mask, initial_decoder_input = batch
            
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            
            with precision_context(args):
                generated_ids = model.generate(
                    input_ids=encoder_input,
                    attention_mask=encoder_mask,
                    max_length=256,
                    num_beams=4,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    decoder_start_token_id=tokenizer.pad_token_id
                )
            
            for gen_ids in generated_ids:
                sql_query = tokenizer.decode(gen_ids, skip_special_tokens=True)
                all_generated_queries.append(sql_query)
    
    # Save test results
    save_queries_and_records(all_generated_queries, model_sql_path, model_record_path)
    print(f"Test predictions saved to:")
    print(f"  SQL queries: {model_sql_path}")
    print(f"  Records: {model_record_path}")

def main():
    """Main function"""
    args = get_args()
    
    print("=" * 60)
    print("T5 Text-to-SQL Training")
    print("=" * 60)
    print(f"Experiment: {args.experiment_name}")
    print(f"Mode: {'Fine-tuning' if args.finetune else 'Training from scratch'}")
    if args.test_only:
        print("Run type: Test-only inference (no training)")
    print(f"Max epochs: {args.max_n_epochs}")
    print(f"Batch size: {args.batch_size} (train), {args.test_batch_size} (eval)")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 60)
    
    if args.bf16:
        if is_bf16_supported():
            args.use_bf16 = True
            print("bfloat16 mixed precision ENABLED.")
        else:
            args.use_bf16 = False
            print("bfloat16 requested but not supported on this device, using float32.")
    else:
        args.use_bf16 = False
    
    # Set up experiment tracking
    if args.use_wandb:
        setup_wandb(args)
    
    try:
        # Ensure directories exist
        ensure_dirs()
        
        # Load data
        print("Loading data...")
        train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)

        if args.test_only:
            print("Test-only inference enabled. Skipping training and loading saved checkpoint...")
            model = load_model_from_checkpoint(args, best=True)
            model.eval()
            
            model_type = 'ft' if args.finetune else 'scr'
            model_sql_path = os.path.join('results', f't5_{model_type}_{args.experiment_name}_test.sql')
            model_record_path = os.path.join('records', f't5_{model_type}_{args.experiment_name}_test.pkl')
            
            test_inference(args, model, test_loader, model_sql_path, model_record_path)
            
            print("\n" + "=" * 50)
            print("TEST-ONLY INFERENCE COMPLETE")
            print("=" * 50)
            print(f"SQL queries: {model_sql_path}")
            print(f"Records: {model_record_path}")
            print("=" * 50)
            return 0
        
        # Initialize the model
        print("Initializing model...")
        model = initialize_model(args)
        
        # Initialize optimizer and scheduler
        print("Initializing optimizer and scheduler...")
        optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))
        
        # Train the model
        best_f1 = train(args, model, train_loader, dev_loader, optimizer, scheduler)
        
        # Load the best model for final evaluation
        print("\nLoading best model for final evaluation...")
        model = load_model_from_checkpoint(args, best=True)
        model.eval()
        
        # Final evaluation on the development set
        model_type = 'ft' if args.finetune else 'scr'
        gt_sql_path = os.path.join('data', 'dev.sql')
        gt_record_path = os.path.join('records', 'ground_truth_dev.pkl')
        model_sql_path = os.path.join('results', f't5_{model_type}_{args.experiment_name}_dev_final.sql')
        model_record_path = os.path.join('records', f't5_{model_type}_{args.experiment_name}_dev_final.pkl')
        
        print("Final evaluation on development set...")
        dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(
            args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path
        )
        
        print("\n" + "=" * 50)
        print("FINAL DEVELOPMENT SET RESULTS")
        print("=" * 50)
        print(f"Loss: {dev_loss:.4f}")
        print(f"Record F1: {dev_record_f1:.4f}")
        print(f"Record EM: {dev_record_em:.4f}") 
        print(f"SQL EM: {dev_sql_em:.4f}")
        print(f"Error Rate: {dev_error_rate*100:.2f}%")
        print("=" * 50)
        
        # Test set inference
        print("\nGenerating test set predictions...")
        model_sql_path = os.path.join('results', f't5_{model_type}_{args.experiment_name}_test.sql')
        model_record_path = os.path.join('records', f't5_{model_type}_{args.experiment_name}_test.pkl')
        test_inference(args, model, test_loader, model_sql_path, model_record_path)
        
        print("\n" + "=" * 50)
        print("SUBMISSION FILES GENERATED")
        print("=" * 50)
        print(f"SQL queries: {model_sql_path}")
        print(f"Records: {model_record_path}")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
