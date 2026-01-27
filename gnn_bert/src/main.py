# 在导入 torch 之前设置 CuBLAS workspace 配置以支持确定性 CuBLAS（如果需要确定性运行）
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

# region 导入模块
import argparse
import warnings
import loguru
import torch

from datetime import datetime

from data_utils import load_and_preprocess_data, clean_data
from dataset import create_data_loaders
from models import get_model
from training import train_model, evaluate_model
from log_units import plot_training_curves, plot_predictions, save_predictions, save_training_history_and_dataset
from config import TASK_CONFIGS
# endregion

warnings.filterwarnings('ignore')


def set_global_seed(seed):
    import random
    import numpy as _np
    import torch as _torch
    random.seed(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)
    # 保证cuDNN确定性
    _torch.backends.cudnn.deterministic = True
    _torch.backends.cudnn.benchmark = False
    try:
        _torch.use_deterministic_algorithms(True)
    except Exception:
        # 某些老版本可能没有该函数
        pass
    print(f"Global seed set to {seed}")


def main():
    # region 解析命令行参数
    parser = argparse.ArgumentParser(description='GNN/BERT Retention Time Prediction')
    parser.add_argument('--task', type=str, default='default', help='Task name to use from config.py')
    # 动态补充所有 config 参数
    for k, v in TASK_CONFIGS['default'].items():
        if k == 'task':
            continue
        arg_type = type(v)
        if isinstance(v, bool):
            parser.add_argument(f'--{k}', action='store_true' if not v else 'store_false', default=v)
        else:
            parser.add_argument(f'--{k}', type=arg_type, default=v)
    args, unknown = parser.parse_known_args()

    # 读取任务参数 - 先读取default，再用特定任务覆盖
    task_name = args.task
    if 'default' not in TASK_CONFIGS:
        raise ValueError("Default task configuration 'default' not found in TASK_CONFIGS!")
    task_cfg = TASK_CONFIGS['default'].copy()
    if task_name != 'default':
        if task_name not in TASK_CONFIGS:
            raise ValueError(f"Task configuration '{task_name}' not found in TASK_CONFIGS!")
        specific_cfg = TASK_CONFIGS[task_name]
        task_cfg.update(specific_cfg)
        print(f"使用任务配置: {task_name} (基于default配置)")
        print(f"覆盖的参数: {list(specific_cfg.keys())}")
    else:
        print("使用默认配置: default")
    # 用最新参数覆盖 args
    for k, v in task_cfg.items():
        setattr(args, k, v)

    # 设置全局随机种子，确保可复现性
    set_global_seed(args.seed)

    # endregion

    # region 环境与目录初始化
    # 创建结果目录
    os.makedirs(args.save_dir, exist_ok=True)
    figs_dir = os.path.join(args.save_dir, 'figs')
    os.makedirs(figs_dir, exist_ok=True)

    # 选择设备：优先使用 args.gpu（如果在配置/命令行中指定），否则使用第一个可用 GPU（索引 0）
    selected_gpu = None
    if torch.cuda.is_available():
        try:
            gpu_count = torch.cuda.device_count()
        except Exception:
            gpu_count = 0

        if gpu_count == 0:
            print("CUDA reported available but found 0 devices. Falling back to CPU.")
            device = torch.device('cpu')
        else:
            # 如果配置中提供了 gpu 参数，则使用之；否则默认 0
            cfg_gpu = getattr(args, 'gpu', None)
            if cfg_gpu is None:
                gpu_idx = 0
            else:
                try:
                    gpu_idx = int(cfg_gpu)
                except Exception:
                    gpu_idx = 0

            # 限制索引在可用范围内
            gpu_idx = max(0, min(gpu_idx, gpu_count - 1))
            selected_gpu = gpu_idx
            device = torch.device(f'cuda:{selected_gpu}')

            # 安全获取设备信息
            try:
                gpu_name = torch.cuda.get_device_name(selected_gpu)
                gpu_memory = torch.cuda.get_device_properties(selected_gpu).total_memory / 1024**3
                print(f"GPU: {gpu_name} (id={selected_gpu}, {gpu_memory:.1f}GB)")
                print(f"CUDA Version: {torch.version.cuda}")
            except AssertionError:
                # 设备 id 无效（极少见），回退到默认设备 0
                print(f"Warning: invalid GPU id {selected_gpu}, falling back to cuda:0 if available")
                if gpu_count > 0:
                    selected_gpu = 0
                    device = torch.device('cuda:0')
                    try:
                        gpu_name = torch.cuda.get_device_name(0)
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        print(f"GPU: {gpu_name} (id=0, {gpu_memory:.1f}GB)")
                    except Exception:
                        pass
                else:
                    device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")
    # endregion

    # region 加载数据

    print("=" * 50)
    print("LOADING DATA")
    print("=" * 50)

    # ---------- 1. Load & clean main dataset ----------
    smiles, retention_times, numerical_features, cluster_ids = \
        load_and_preprocess_data(args.data_path)

    smiles, retention_times, numerical_features, cluster_ids = \
        clean_data(smiles, retention_times, numerical_features, cluster_ids)


    # ---------- 2. Case A: independent test set ----------
    if args.test_data_path is not None:
        print(f"Loading training set from: {args.data_path}")
        print(f"Loading independent test set from: {args.test_data_path}")

        test_smiles, test_retention_times, test_numerical_features, test_cluster_ids = \
            load_and_preprocess_data(args.test_data_path)

        test_smiles, test_retention_times, test_numerical_features, test_cluster_ids = \
            clean_data(
                test_smiles,
                test_retention_times,
                test_numerical_features,
                test_cluster_ids)

        train_loader, _, train_dataset, _ = create_data_loaders(
            smiles=smiles,
            y=retention_times,
            phys=numerical_features,
            model_type=args.model_type,
            batch_size=args.batch_size,
            random_state=args.seed,
            bert_model_name=args.bert_model_name,
            max_length=args.max_seq_length,
            augment=(args.model_type == "bert"))

        _, test_loader, _, test_dataset = create_data_loaders(
            smiles=test_smiles,
            y=test_retention_times,
            phys=test_numerical_features,
            model_type=args.model_type,
            batch_size=args.batch_size,
            shuffle=False,
            random_state=args.seed,
            bert_model_name=args.bert_model_name,
            max_length=args.max_seq_length,
            augment=False)

        print(
            f"Train set size: {len(train_loader.dataset)}, "
            f"Test set size: {len(test_loader.dataset)}")


    # ---------- 3. Case B: split train / test ----------
    else:
        print("No independent test set provided, splitting training data")

        train_loader, test_loader, train_dataset, test_dataset = create_data_loaders(
            smiles=smiles,
            y=retention_times,
            phys=numerical_features,
            model_type=args.model_type,
            train_ratio=args.train_ratio,
            batch_size=args.batch_size,
            random_state=args.seed,
            bert_model_name=args.bert_model_name,
            max_length=args.max_seq_length,
            augment=(args.model_type == "bert"))

        print(
            f"Train set size: {len(train_loader.dataset)}, "
            f"Test set size: {len(test_loader.dataset)}")

    # endregion

    # region 创建模型
    print("\n" + "="*50)
    print("CREATING MODEL")
    print("="*50)

    if args.model_type == 'bert':
        input_dim = None  # BERT不需要input_dim
        phys_dim = getattr(train_dataset, "phys_dim", 0)
        print(f"Physicochemical feature dimension: {phys_dim}")

        if args.use_physicochemical and phys_dim == 0:
            raise ValueError("BERT mode requires physicochemical features, but none were provided.")
    else:
        input_dim = train_dataset.get_feature_dim()
        phys_dim = 0
        print(f"Input feature dimension: {input_dim}")

    model = get_model(
        input_dim=input_dim,
        pretrained_model=args.model_dir,
        freeze_backbone=args.finetune,
        hidden_dim=args.hidden_dim,
        output_dim=1,
        num_layers=args.num_layers,
        dropout=args.dropout,
        gnn_type=args.gnn_type,
        model_type=args.model_type,
        bert_model_name=args.bert_model_name,
        phys_dim=phys_dim
    )

    model = model.to(device)
    print(f"Model: {model.__class__.__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    # endregion

    # region 模型训练与评估
    print("\n" + "="*50)
    print("TRAINING MODEL")
    print("="*50)
    model_save_path = os.path.join(args.save_dir, 'best_model.pth')

    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        device=device,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_path=model_save_path
    )

    # 可视化与保存结果
    plot_training_curves(
        train_losses, val_losses,
        save_path=os.path.join(figs_dir, 'training_curves.png'))
    
    print("\n" + "="*50)
    print("EVALUATING MODEL")
    print("="*50)

    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        model_path=model_save_path
    )

    plot_predictions(
        results,
        save_path=os.path.join(figs_dir, 'predictions.png'))

    save_predictions(
        results,
        save_path=os.path.join(args.save_dir, 'predictions.csv'))



    # 保存训练历史和config参数为两个文件
    history = {
        'final_results': {
            'mse': results.get('mse_orig', results['mse']),
            'rmse': results.get('rmse_orig', results['rmse']),
            'mae': results.get('mae_orig', results['mae']),
            'r2': results.get('r2_orig', results['r2'])
        },
        'model_config': (
            {
                'bert_model_name': args.bert_model_name,
                'max_seq_length': args.max_seq_length,
                'dropout': args.dropout,
                'phys_dim': phys_dim,
                'freeze_backbone': args.finetune
            }
            if args.model_type == 'bert' else
            {
                'input_dim': input_dim,
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
                'dropout': args.dropout,
                'gnn_type': args.gnn_type,
                'seed': args.seed
            }
        ),
        'training_config': {
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'train_ratio': args.train_ratio
        },
        'dataset_name': os.path.basename(args.data_path),
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
        import json
        json.dump(history, f, indent=2, ensure_ascii=False)
    config_dict = vars(args)
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        import json
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved to: {args.save_dir}")
    print("="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    # endregion

if __name__ == '__main__':
    main()

