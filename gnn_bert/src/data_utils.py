import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from rdkit import Chem
import warnings
import loguru 

from sympy import N
warnings.filterwarnings('ignore')


def load_and_preprocess_data(
        data_path, cluster_col=None,
        smiles_col='SMILES', rt_col='UV_RT-s', 
        feature_col=[
            'MolWt', 'logP', 'TPSA', 'H_bond_donors', 'H_bond_acceptors']):
    """
    加载和预处理数据
    """
    # print(f"Loading data from {data_path}")

    data = pd.read_csv(data_path)
    
    # 提取需要的列
    smiles = data[smiles_col].tolist()
    retention_times = data[rt_col].tolist() if rt_col in data.columns else [0] * len(smiles)
    cluster_ids = data[cluster_col].tolist() if cluster_col in data.columns else None
    numerical_features = data[feature_col].values
    
    if cluster_ids is not None:
        unique_clusters = sorted(list(set(cluster_ids)))
        print(f"Found cluster information with {len(unique_clusters)} clusters: {unique_clusters}")
        cluster_counts = pd.Series(cluster_ids).value_counts().sort_index()
        print("Cluster distribution:")
        for cluster_id in unique_clusters:
            print(f"  Cluster {cluster_id}: {cluster_counts[cluster_id]} samples")
    
    return smiles, retention_times, numerical_features, cluster_ids

def analyze_data(smiles, retention_times, numerical_features, feature_names):
    """
    数据分析和可视化
    """
    print("\n=== Data Analysis ===")
    
    # 基本统计
    print(f"SMILES count: {len(smiles)}")
    print(f"Retention time range: [{min(retention_times):.2f}, {max(retention_times):.2f}]")
    print(f"Retention time mean±std: {np.mean(retention_times):.2f}±{np.std(retention_times):.2f}")
    
    # 检查缺失值
    print(f"\nMissing values:")
    print(f"SMILES: {sum(1 for s in smiles if not s or pd.isna(s))}")
    print(f"Retention times: {sum(1 for rt in retention_times if pd.isna(rt))}")
    print(f"Numerical features: {np.isnan(numerical_features).sum()}")
    
    # 数值特征统计
    print(f"\nNumerical features statistics:")
    for i, name in enumerate(feature_names):
        feat = numerical_features[:, i]
        print(f"{name}: {feat.min():.2f} - {feat.max():.2f} (mean: {feat.mean():.2f})")
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 保留时间分布
    axes[0, 0].hist(retention_times, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Retention Time (s)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Retention Time Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 数值特征分布
    for i, name in enumerate(feature_names):
        row = (i + 1) // 3
        col = (i + 1) % 3
        if row < 2 and col < 3:
            axes[row, col].hist(numerical_features[:, i], bins=30, alpha=0.7, edgecolor='black')
            axes[row, col].set_xlabel(name)
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].set_title(f'{name} Distribution')
            axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 相关性矩阵
    all_data = np.column_stack([numerical_features, np.array(retention_times).reshape(-1, 1)])
    all_names = feature_names + ['Retention_Time']
    
    correlation_matrix = np.corrcoef(all_data.T)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                xticklabels=all_names, yticklabels=all_names)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()

def clean_data(smiles, retention_times, numerical_features, cluster_ids=None):
    valid_indices = []
    for i, (smi, rt) in enumerate(zip(smiles, retention_times)):
        # 检查SMILES是否有效
        if smi and not pd.isna(smi):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                # 检查保留时间是否有效
                if not pd.isna(rt) and rt > 0:
                    # 检查数值特征是否有效
                    if not np.isnan(numerical_features[i]).any():
                        # 检查聚类信息是否有效（如果提供）
                        if cluster_ids is None or not pd.isna(cluster_ids[i]):
                            valid_indices.append(i)
    
    # print(f"Valid samples: {len(valid_indices)}/{len(smiles)}")
    
    # 过滤数据
    clean_smiles = [smiles[i] for i in valid_indices]
    clean_rt = [retention_times[i] for i in valid_indices]
    clean_features = numerical_features[valid_indices]
    clean_clusters = [cluster_ids[i] for i in valid_indices] if cluster_ids is not None else None
    
    return clean_smiles, clean_rt, clean_features, clean_clusters

def get_data_summary(smiles, retention_times, numerical_features, feature_names):
    """
    获取数据摘要
    """
    summary = {
        'total_samples': len(smiles),
        'valid_smiles': sum(1 for s in smiles if s and not pd.isna(s)),
        'rt_stats': {
            'min': min(retention_times),
            'max': max(retention_times),
            'mean': np.mean(retention_times),
            'std': np.std(retention_times)
        },
        'feature_stats': {}
    }
    
    for i, name in enumerate(feature_names):
        feat = numerical_features[:, i]
        summary['feature_stats'][name] = {
            'min': feat.min(),
            'max': feat.max(),
            'mean': feat.mean(),
            'std': feat.std()
        }
    
    return summary

def stratified_train_test_split(smiles, retention_times, numerical_features, cluster_ids, 
                               train_ratio=0.9, random_state=42):
    """
    基于聚类信息进行分层抽样，划分训练集和测试集
    
    Args:
        smiles: SMILES列表
        retention_times: 保留时间列表
        numerical_features: 数值特征数组
        cluster_ids: 聚类ID列表
        train_ratio: 训练集比例
        random_state: 随机种子
    
    Returns:
        训练集和测试集的索引
    """
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    print(f"Performing stratified sampling with train ratio: {train_ratio}")
    
    # 创建样本索引
    indices = np.arange(len(smiles))
    
    # 使用sklearn的分层抽样
    train_indices, test_indices = train_test_split(
        indices, 
        test_size=1-train_ratio,
        stratify=cluster_ids,
        random_state=random_state
    )
    
    # 打印分层抽样结果
    print("\nStratified sampling results:")
    print(f"Training set: {len(train_indices)} samples")
    print(f"Test set: {len(test_indices)} samples")
    
    # 检查每个聚类在训练集和测试集中的分布
    train_clusters = [cluster_ids[i] for i in train_indices]
    test_clusters = [cluster_ids[i] for i in test_indices]
    
    train_cluster_counts = pd.Series(train_clusters).value_counts().sort_index()
    test_cluster_counts = pd.Series(test_clusters).value_counts().sort_index()
    
    print("\nCluster distribution in training set:")
    for cluster_id in sorted(set(cluster_ids)):
        train_count = train_cluster_counts.get(cluster_id, 0)
        test_count = test_cluster_counts.get(cluster_id, 0)
        total_count = train_count + test_count
        train_pct = train_count / total_count * 100 if total_count > 0 else 0
        print(f"  Cluster {cluster_id}: {train_count}/{total_count} ({train_pct:.1f}%)")
    
    print("\nCluster distribution in test set:")
    for cluster_id in sorted(set(cluster_ids)):
        train_count = train_cluster_counts.get(cluster_id, 0)
        test_count = test_cluster_counts.get(cluster_id, 0)
        total_count = train_count + test_count
        test_pct = test_count / total_count * 100 if total_count > 0 else 0
        print(f"  Cluster {cluster_id}: {test_count}/{total_count} ({test_pct:.1f}%)")
    
    return train_indices, test_indices
