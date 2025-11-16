import os
import torch
import transformers
from transformers import T5ForConditionalGeneration, T5Config, T5TokenizerFast
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {DEVICE}")

def setup_wandb(args):
    """设置Weights & Biases日志记录"""
    if args.use_wandb:
        wandb.init(project="text-to-sql-t5", name=args.experiment_name)
        wandb.config.update(args)

def initialize_model(args):
    """
    初始化T5模型
    - 如果args.finetune为True，则加载预训练权重进行微调
    - 否则，从配置初始化（从头训练）
    """
    try:
        if args.finetune:
            print("Loading pretrained T5-small model for fine-tuning...")
            model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
        else:
            print("Initializing T5 model from scratch...")
            config = T5Config.from_pretrained('google-t5/t5-small')
            model = T5ForConditionalGeneration(config)
        
        model.to(DEVICE)
        
        # 打印模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model initialized: {total_params:,} total parameters, {trainable_params:,} trainable")
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def ensure_dirs():
    """确保所有必要的目录都存在"""
    dirs = ['checkpoints', 'results', 'records']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("Required directories checked/created")

def save_model(checkpoint_dir, model, best=False):
    """保存模型检查点"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if best:
        model_path = os.path.join(checkpoint_dir, 'best_model.pt')
    else:
        model_path = os.path.join(checkpoint_dir, 'latest_model.pt')
    
    # 保存模型状态字典
    torch.save({
        'model_state_dict': model.state_dict(),
    }, model_path)
    print(f"Model saved to {model_path}")

def load_model_from_checkpoint(args, best=False):
    """从检查点加载模型"""
    model = initialize_model(args)
    
    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    
    if best:
        checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
    else:
        checkpoint_path = os.path.join(checkpoint_dir, 'latest_model.pt')
    
    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully")
    else:
        print(f"Checkpoint not found at {checkpoint_path}, using initial model")
    
    return model

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    """初始化优化器和学习率调度器"""
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    """初始化优化器，对不同的参数使用不同的权重衰减"""
    # 获取需要权重衰减的参数名（排除层归一化和偏置项）
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if (n in decay_parameters and p.requires_grad)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if (n not in decay_parameters and p.requires_grad)],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, 
            lr=args.learning_rate, 
            eps=1e-8, 
            betas=(0.9, 0.999)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer_type}")
    
    print(f"Optimizer initialized: {args.optimizer_type}, lr={args.learning_rate}, weight_decay={args.weight_decay}")
    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    """初始化学习率调度器"""
    if args.scheduler_type == "none":
        print("No learning rate scheduler used")
        return None
        
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "cosine":
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif args.scheduler_type == "linear":
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler_type}")
    
    print(f"Scheduler initialized: {args.scheduler_type}, warmup_epochs={args.num_warmup_epochs}")
    return scheduler

def get_parameter_names(model, forbidden_layer_types):
    """递归获取所有参数名"""
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # 添加模型特定参数
    result += list(model._parameters.keys())
    return result