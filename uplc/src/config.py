# -*- coding: utf-8 -*-


TASK_CONFIGS = {
    'default': {
        'data_path': None,
        'test_data_path': None,
        'use_stratified_sampling': False,
        'save_dir': '/home/huangzy/uplc/results/default',

        # 基本训练参数
        'model_type': 'gnn',
        'num_epochs': 3000,
        'batch_size': 512,
        'dropout': 0.5,
        'weight_decay': 1e-4,
        'learning_rate': 0.0001,
        'train_ratio': 0.9,

        'seed': 42,

        # GNN相关参数
        'gnn_type': 'GIN',
        'num_layers': 3,
        'hidden_dim': 128,
        'input_dim': None,  # 由数据集自动推断

        # BERT相关参数
        'bert_model_name': 'DeepChem/ChemBERTa-77M-MTR',
        'bert_model_dir': None,  # 可选，预训练模型目录
        'max_seq_length': 128,

        # 通用参数
        'model_dir': None,  # 预训练权重路径或模型保存路径
        'finetune': False,
        'gradient_clipping': 1.0,
        'scheduler': 'CosineAnnealingLR',
        'analyze_data': False,
        'gpu': None,  # 默认自动选择
        'use_physicochemical': True,
    },

    '20250801_GNN': {
        'data_path': '/home/huangzy/UPLC/datas/20250801/train/processed_dedup_filtered_add_fg_f-Default-2-90__cleaned_with_labels_k3_train.csv',
        'test_data_path': '/home/huangzy/UPLC/datas/20250801/test/processed_dedup_filtered_add_fg_f-Default-2-90__cleaned_with_labels_k3_test.csv',  # 新增：测试数据集路径，如果提供则不分割训练集
        'hidden_dim': 128,
        'num_layers': 3,
        'dropout': 0.2,
        'gnn_type': 'GIN',
        'batch_size': 512,
        'num_epochs': 3000,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'train_ratio': 0.9,
        'save_dir': '/home/huangzy/UPLC/论文数据/20250801/GNN/Default-2'},

    '20250808_GNN': {
        'data_path': '/home/huangzy/UPLC/datas/20250808/train/Default_Neutral_train.csv',
        'test_data_path': '/home/huangzy/UPLC/datas/20250808/test/Default_Neutral_test.csv',  
        'hidden_dim': 128,
        'num_layers': 3,
        'dropout': 0.2,
        'gnn_type': 'GIN',
        'batch_size': 512,
        'num_epochs': 3000,
        'learning_rate': 0.0001,
        'weight_decay': 1e-4,
        'save_dir': '/home/huangzy/UPLC/论文数据/20250808/GNN-Default-Neutral'},

    'test': {
        'data_path': '/home/huangzy/UPLC/datas/clustering_k0.csv',
        'save_dir': '/home/huangzy/UPLC/results/test',
        'num_epochs': 100},

    '20250811_GNN': {
        'data_path': '/home/huangzy/UPLC/datas/20250808/train/Default-2-BLANCE_train.csv',
        'test_data_path': '/home/huangzy/UPLC/datas/20250808/test/Default-2-BLANCE_test.csv',  
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'num_epochs': 3000,
        # 'model_dir': '/home/huangzy/UPLC/论文数据/20250811/GNN-Default-2/best_model.pth',
        'save_dir': '/home/huangzy/UPLC/论文数据/20250811/GNN-Default-2-BLANCE'},

    '20260127_GNN': {
        'data_path': '/home/huangzy/uplc/data/2.train_test_split/AM-I-filtered_with_labels_k4_train.csv',
        'test_data_path': '/home/huangzy/uplc/data/2.train_test_split/AM-I-filtered_with_labels_k4_test.csv',  
        'batch_size': 2048,
        'num_epochs': 5000,
        'num_layers': 2,
        'hidden_dim': 64,
        'learning_rate': 0.0001,
        'weight_decay': 1e-4,
        'dropout': 0.2, 
        'save_dir': '/home/huangzy/uplc-method-recommendation/BERT_GIN/论文数据/20260127/GNN-AM-I'},

    'test1': {
        'data_path': '/home/huangzy/uplc/data/2.train_test_split/AM-II-filtered_with_labels_k3_train.csv',
        'test_data_path': '/home/huangzy/uplc/data/2.train_test_split/AM-II-filtered_with_labels_k3_test.csv',  
        'batch_size': 128,
        'num_epochs': 500,
        'num_layers': 2,
        'hidden_dim': 64,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'dropout': 0.7, 
        'save_dir': '/home/huangzy/uplc/results/test'},
    'BERT_test': {
        'data_path': '/mnt/shared/uplc-method-recommendation/uplc/data/2.train_test_split/AM-I-filtered_with_labels_k4_train.csv',
        'test_data_path': '/mnt/shared/uplc-method-recommendation/uplc/data/2.train_test_split/AM-I-filtered_with_labels_k4_test.csv',  
        'model_type': 'bert',
        'max_seq_length': 128,
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'dropout': 0.2,
        'weight_decay': 0.0001,  
        'save_dir': '/mnt/shared/uplc-method-recommendation/uplc/results/BERT-AM-I'
    }

}

# ====== 绘图参数配置 ======
PLOT_CONFIG = {
    'default': {
        'FIGSIZE': (6, 6),
        'LINEWIDTH': 1,
        'LABELSIZE': 10,
        'TITLESIZE': 12,
        'SAVE_DPI': 600,
        'SAVE_BBOX': 'tight',
        'SCATTER_SIZE': 50,
        'ALPHA': 0.6,
        'TICK_LENGTH': 6,
        'TICK_WIDTH': 1.2,
        'EDGE_COLOR': 'k',
        'IPHONE_COLORS': {
            "scatter": "#007AFF",
            "line": "#FF3B30",
            "residual": "#34C759",
            "text": "#1C1C1E",
            "train": "#007AFF",
            "val": "#FF3B30"
        }
    },
    'training_curves': {
        'XLABEL': 'Epoch (every 10)',
        'YLABEL': 'Loss (log scale)',
        # 如需特殊配置可在此覆盖default，如：
        # 'FIGSIZE': (8, 6)
    },
    'prediction_plot': {
        'XLABEL': 'True Retention Time (s)',
        'YLABEL': 'Predicted Retention Time (s)',
        # 如需特殊配置可在此覆盖default
    },
    'residual_plot': {
        'XLABEL': 'Predicted Retention Time (s)',
        'YLABEL': 'Residuals (Predicted - True)',
        # 如需特殊配置可在此覆盖default
    },
    
    'iphone_style': {
        'FIGSIZE': (6, 6),
        'LINEWIDTH': 3,
        'LABELSIZE': 17,
        'TITLESIZE': 16,
        'SAVE_DPI': 600,
        'SAVE_BBOX': 'tight',
        'SCATTER_SIZE': 70,
        'ALPHA': 0.8,
        'TICK_LENGTH': 6,
        'TICK_WIDTH': 2,
        'EDGE_COLOR': 'k',
        'GRID': False,
        'SPINE_VISIBLE': True,
        'IPHONE_COLORS': {
            'scatter': '#007AFF',
            'line':    '#AEAEB2',
            'text':    '#000000'
        }
    },
    'iphone_scatter': {
        'XLABEL': 'True Retention Time (s)',
        'YLABEL': 'Predicted Retention Time (s)',
        'ANNOT_FONT': 16,
        'ANNOT_POS': (0.05, 0.95)
    },
    'iphone_residual': {
        'XLABEL': 'Predicted Retention Time (s)',
        'YLABEL': 'Residuals (Predicted - True)',
        'ANNOT_FONT': 16,
        'ANNOT_POS': (0.05, 0.95),
        'TITLE_POS': (0.5, -0.15),
    }
}