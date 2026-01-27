import torch
from sklearn.preprocessing import StandardScaler
from data_utils import load_and_preprocess_data
from dataset import create_data_loaders
from models import get_model
import pandas as pd

data_path = '../datas/预测/output_data_with_properties.csv'  
smiles, retention_times, numerical_features, feature_names, _ = load_and_preprocess_data(data_path)

# 标准化数值特征
scaler = StandardScaler()
numerical_features_scaled = scaler.fit_transform(numerical_features)


train_loader, test_loader, train_dataset, test_dataset = create_data_loaders(
    smiles, retention_times, numerical_features,
    cluster_ids=None,
    train_ratio=0.999,
    batch_size=1,
    shuffle=False,
    random_state=31
)
print(f"[INFO] Training samples: {len(train_dataset)}")
print(f"[INFO] Test samples: {len(test_dataset)}")

input_dim = train_dataset.get_feature_dim()
model = get_model(
    input_dim=input_dim,
    pretrained_model="/home/huangzy/deep-learning/UPLC/gnn_retention_prediction/results/GAT_k0/best_model.pth", 
    freeze_backbone=False,
    hidden_dim=128,
    output_dim=1,
    num_layers=3,
    dropout=0,
    gnn_type='GAT'
)

output = []
device = torch.device('cpu')
for batch_idx, batch in enumerate(train_loader):
    batch = batch.to(device)
    with torch.no_grad():
        output_batch = model(batch)
        # 保存smiles和预测结果
        for i in range(len(batch.smiles)):
            smiles_str = batch.smiles[i]
            retention_time = output_batch[i].item()
            output.append({'SMILE': smiles_str, 'Predicted_RT': retention_time})

# 保存预测结果

output_df = pd.DataFrame(output)
output_df.to_csv('predictions1.csv', index=False)
print("[INFO] Predictions saved to predictions.csv")
