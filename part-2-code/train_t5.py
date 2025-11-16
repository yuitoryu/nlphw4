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
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def is_bf16_supported():
    """检查当前环境是否支持bfloat16"""
    if DEVICE.type != 'cuda':
        return False
    return hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()

def precision_context(args):
    """根据配置返回自动混合精度上下文"""
    if getattr(args, 'use_bf16', False) and DEVICE.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return nullcontext()

def get_args():
    '''训练参数配置'''
    parser = argparse.ArgumentParser(description='T5 training for Text-to-SQL')
    
    # 模型配置
    parser.add_argument('--finetune', action='store_true', 
                       help="Fine-tune pretrained T5 (otherwise train from scratch)")
    parser.add_argument('--bf16', action='store_true',
                       help="Use bfloat16 mixed precision when supported")
    
    # 训练超参数
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"])
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help="Learning rate (suggested: 1e-4 for finetune, 3e-4 for scratch)")
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--scheduler_type', type=str, default="linear", 
                       choices=["none", "cosine", "linear"])
    parser.add_argument('--num_warmup_epochs', type=int, default=1)
    parser.add_argument('--max_n_epochs', type=int, default=25)
    parser.add_argument('--patience_epochs', type=int, default=5)

    # 实验跟踪
    parser.add_argument('--use_wandb', action='store_true',
                       help="Use Weights & Biases for experiment tracking")
    parser.add_argument('--experiment_name', type=str, default='baseline',
                       help="Name for this experiment run")
    parser.add_argument('--force_sql_eval', action='store_true',
                       help="Force SQL-based evaluation each epoch even when not fine-tuning")

    # 数据超参数
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--test_batch_size', type=int, default=16)

    return parser.parse_args()

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    """训练主循环"""
    ensure_dirs()
    best_f1 = -1
    best_loss = float('inf')
    epochs_since_improvement = 0
    run_sql_eval = args.finetune or args.force_sql_eval
    
    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    args.checkpoint_dir = checkpoint_dir
    
    # 评估文件路径
    gt_sql_path = os.path.join('data', 'dev.sql')
    gt_record_path = os.path.join('records', 'ground_truth_dev.pkl')
    model_sql_path = os.path.join('results', f't5_{model_type}_{args.experiment_name}_dev.sql')
    model_record_path = os.path.join('records', f't5_{model_type}_{args.experiment_name}_dev.pkl')
    
    # 确保目录存在
    os.makedirs(os.path.dirname(model_sql_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_record_path), exist_ok=True)
    
    print(f"Starting training for {args.max_n_epochs} epochs...")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    print(f"Results will be saved with prefix: t5_{model_type}_{args.experiment_name}")
    
    for epoch in range(args.max_n_epochs):
        # 训练一个epoch
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch+1}/{args.max_n_epochs}: Train Loss = {tr_loss:.4f}")

        # 评估（非微调模式下每5个epoch运行一次完整SQL评估）
        should_run_sql_eval = run_sql_eval or ((epoch + 1) % 5 == 0)
        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path,
            run_sql_eval=should_run_sql_eval
        )
        
        if record_f1 is not None:
            print(f"Epoch {epoch+1}: Dev Loss = {eval_loss:.4f}, Record F1 = {record_f1:.4f}, "
                  f"Record EM = {record_em:.4f}, SQL EM = {sql_em:.4f}")
            print(f"Epoch {epoch+1}: Error Rate = {error_rate*100:.2f}%")
        else:
            print(f"Epoch {epoch+1}: Dev Loss = {eval_loss:.4f} (SQL metrics skipped)")

        # 记录到wandb
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

        # 早停逻辑
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

        # 保存最新模型
        save_model(checkpoint_dir, model, best=False)

        # 检查早停
        if epochs_since_improvement >= args.patience_epochs:
            print(f"Early stopping after {epoch + 1} epochs without improvement")
            break

    if run_sql_eval:
        print(f"Training completed. Best F1: {best_f1:.4f}")
    else:
        print(f"Training completed. Best Dev Loss: {best_loss:.4f}")
    return best_f1

def train_epoch(args, model, train_loader, optimizer, scheduler):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_batches = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch in progress_bar:
        # 正常collate函数返回5个值
        encoder_input, encoder_mask, decoder_input, decoder_targets, _ = batch
        
        # 移动到设备
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        # 前向传播
        with precision_context(args):
            outputs = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
                labels=decoder_targets
            )
            loss = outputs.loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # 统计
        total_loss += loss.item()
        total_batches += 1
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / total_batches if total_batches > 0 else 0

def eval_epoch(args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path,
               run_sql_eval=True):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_batches = 0
    all_generated_queries = []
    
    if run_sql_eval:
        tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    else:
        tokenizer = None
    
    with torch.no_grad():
        # 计算验证损失
        for batch in tqdm(dev_loader, desc="Evaluating Loss"):
            # 正常collate函数返回5个值
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
        
        # 生成SQL查询
        print("Generating SQL queries for evaluation...")
        for batch in tqdm(dev_loader, desc="Generating SQL"):
            # 对于生成，我们只需要前3个元素：encoder_input, encoder_mask, initial_decoder_input
            # 但normal_collate_fn返回5个元素，所以我们需要正确解包
            encoder_input, encoder_mask, _, _, initial_decoder_input = batch
            
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            
            # 使用beam search生成
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
            
            # 解码生成的SQL
            for gen_ids in generated_ids:
                sql_query = tokenizer.decode(gen_ids, skip_special_tokens=True)
                all_generated_queries.append(sql_query)
    
    # 保存并计算指标
    print("Saving generated queries and computing metrics...")
    save_queries_and_records(all_generated_queries, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_path, model_sql_path, gt_record_path, model_record_path
    )
    
    # 计算错误率
    error_rate = sum(1 for msg in error_msgs if msg and msg.strip()) / len(error_msgs) if error_msgs else 0
    
    return avg_loss, record_f1, record_em, sql_em, error_rate

def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    """测试集推理"""
    model.eval()
    all_generated_queries = []
    
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    print("Generating test set predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Inference"):
            # 测试collate函数返回3个值
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
    
    # 保存测试结果
    save_queries_and_records(all_generated_queries, model_sql_path, model_record_path)
    print(f"Test predictions saved to:")
    print(f"  SQL queries: {model_sql_path}")
    print(f"  Records: {model_record_path}")

def main():
    """主函数"""
    args = get_args()
    
    print("=" * 60)
    print("T5 Text-to-SQL Training")
    print("=" * 60)
    print(f"Experiment: {args.experiment_name}")
    print(f"Mode: {'Fine-tuning' if args.finetune else 'Training from scratch'}")
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
    
    # 设置实验跟踪
    if args.use_wandb:
        setup_wandb(args)
    
    try:
        # 确保目录存在
        ensure_dirs()
        
        # 加载数据
        print("Loading data...")
        train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
        
        # 初始化模型
        print("Initializing model...")
        model = initialize_model(args)
        
        # 初始化优化器和调度器
        print("Initializing optimizer and scheduler...")
        optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))
        
        # 训练模型
        best_f1 = train(args, model, train_loader, dev_loader, optimizer, scheduler)
        
        # 加载最佳模型进行最终评估
        print("\nLoading best model for final evaluation...")
        model = load_model_from_checkpoint(args, best=True)
        model.eval()
        
        # 开发集最终评估
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
        
        # 测试集推理
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
