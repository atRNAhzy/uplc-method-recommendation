import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import os

def forward_and_loss(model, batch, device, criterion=None):
    """
    统一 GNN / BERT 的 forward + loss 计算
    """

    # -------- BERT / ChemBERTa：DataLoader 默认 collate 为 dict --------
    if isinstance(batch, dict):
        labels = batch.get("labels")
        if labels is None:
            raise ValueError("BERT batch missing labels")

        # 不修改原 batch，复制一个送入设备
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = labels.to(device).view(-1).float()

        outputs = model(**inputs)
        preds = outputs.logits if hasattr(outputs, "logits") else outputs
        preds = preds.squeeze(-1)

        if criterion is None:
            raise ValueError("BERT training requires criterion when model output has no loss")

        loss = criterion(preds, labels)
        return loss, preds, labels

    # -------- BERT / ChemBERTa：手动 tuple 形式 (inputs_dict, labels) --------
    if isinstance(batch, (tuple, list)):
        inputs, labels = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device).view(-1).float()

        outputs = model(**inputs)
        preds = outputs.logits if hasattr(outputs, "logits") else outputs
        preds = preds.squeeze(-1)

        if criterion is None:
            raise ValueError("BERT training requires criterion when model output has no loss")

        loss = criterion(preds, labels)
        return loss, preds, labels

    # -------- GNN (PyG) 情况 --------
    # batch = torch_geometric.data.Data
    batch = batch.to(device)
    preds = model(batch)
    labels = batch.y.view(-1, 1)

    if criterion is None:
        raise ValueError("GNN training requires criterion")

    loss = criterion(preds, labels)
    return loss, preds.squeeze(-1), labels.squeeze(-1)

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=100,
    learning_rate=0.001,
    scheduler=None,
    weight_decay=1e-4,
    save_path="best_model.pth",
):
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # ---------- scheduler ----------
    if scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)
    elif scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    elif scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    elif scheduler == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif scheduler == "OneCycleLR":
        total_steps = len(train_loader) * num_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    print(f"Starting training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(num_epochs):

        # ===== train =====
        model.train()
        total_train_loss = 0
        train_preds, train_trues = [], []

        for batch in train_loader:
            optimizer.zero_grad()

            loss, preds, targets = forward_and_loss(
                model, batch, device, criterion
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

            total_train_loss += loss.item()
            train_preds.append(preds.detach().cpu().numpy())
            train_trues.append(targets.detach().cpu().numpy())

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        train_preds = np.concatenate(train_preds)
        train_trues = np.concatenate(train_trues)
        train_r2 = r2_score(train_trues, train_preds)

        # ===== validation =====
        model.eval()
        total_val_loss = 0
        val_preds, val_trues = [], []

        with torch.no_grad():
            for batch in val_loader:
                loss, preds, targets = forward_and_loss(
                    model, batch, device, criterion
                )

                total_val_loss += loss.item()
                val_preds.append(preds.cpu().numpy())
                val_trues.append(targets.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)
        val_r2 = r2_score(val_trues, val_preds)

        # scheduler step
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif scheduler is not None and not isinstance(
            scheduler, torch.optim.lr_scheduler.OneCycleLR
        ):
            scheduler.step()

        # save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "train_r2": train_r2,
                    "val_r2": val_r2,
                },
                save_path
            )

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {avg_train_loss:.6f}, R2: {train_r2:.4f}")
            print(f"  Val   Loss: {avg_val_loss:.6f}, R2: {val_r2:.4f}")
            print(f"  Best  Val  : {best_val_loss:.6f}")

    return train_losses, val_losses

def evaluate_model(model, test_loader, device, model_path=None):
    """
    通用评估函数：
    - 兼容 PyG GNN (batch.y, model(batch))
    - 兼容 BERT / Transformer (dict batch, model(**batch))
    """

    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {model_path}")

    model = model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []
    all_smiles = []

    with torch.no_grad():
        for batch in test_loader:

            # ---------- 情况 A：tuple / list ----------
            if isinstance(batch, (list, tuple)):
                batch_inputs, batch_labels = batch
            else:
                batch_inputs = batch
                batch_labels = None

            # 如果是 dict 且包含 labels，则补上标签（兼容 BERT DataLoader 默认行为）
            if batch_labels is None and isinstance(batch_inputs, dict):
                batch_labels = batch_inputs.get("labels")

            # ========== BERT ==========
            if isinstance(batch_inputs, dict):
                batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                outputs = model(**batch_inputs)

                if hasattr(outputs, "logits"):
                    preds = outputs.logits
                else:
                    preds = outputs

                all_predictions.extend(preds.cpu().numpy().flatten())

                if batch_labels is not None:
                    batch_labels = batch_labels.to(device)
                    all_targets.extend(batch_labels.cpu().numpy().flatten())

            # ========== GNN ==========
            else:
                batch_inputs = batch_inputs.to(device)
                preds = model(batch_inputs)
                targets = batch_inputs.y.view(-1, 1)

                all_predictions.extend(preds.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())

                if hasattr(batch_inputs, "smiles"):
                    all_smiles.extend(batch_inputs.smiles)

    # =========================
    # 数值指标（scaled）
    # =========================
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)

    # =========================
    # 反归一化（如果存在）
    # =========================
    mse_orig = rmse_orig = mae_orig = r2_orig = None
    preds_orig = targets_orig = None

    try:
        dataset = getattr(test_loader, "dataset", None)
        scaler = getattr(dataset, "rt_scaler", None)

        if scaler is not None:
            preds_arr = np.array(all_predictions).reshape(-1, 1)
            targets_arr = np.array(all_targets).reshape(-1, 1)

            preds_orig = scaler.inverse_transform(preds_arr).reshape(-1)
            targets_orig = scaler.inverse_transform(targets_arr).reshape(-1)

            mse_orig = mean_squared_error(targets_orig, preds_orig)
            rmse_orig = np.sqrt(mse_orig)
            mae_orig = mean_absolute_error(targets_orig, preds_orig)
            r2_orig = r2_score(targets_orig, preds_orig)

    except Exception as e:
        print(f"Warning: inverse transform failed: {e}")

    # =========================
    # 打印结果
    # =========================
    print("\n=== Test Results (scaled) ===")
    print(f"MSE : {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE : {mae:.6f}")
    print(f"R²  : {r2:.6f}")

    if preds_orig is not None:
        print("\n=== Test Results (original scale) ===")
        print(f"MSE : {mse_orig:.6f}")
        print(f"RMSE: {rmse_orig:.6f}")
        print(f"MAE : {mae_orig:.6f}")
        print(f"R²  : {r2_orig:.6f}")

    print("\n=== End Evaluation ===")

    return {
        "predictions": all_predictions,
        "targets": all_targets,
        "smiles": all_smiles,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "predictions_orig": preds_orig.tolist() if preds_orig is not None else None,
        "targets_orig": targets_orig.tolist() if targets_orig is not None else None,
        "mse_orig": mse_orig,
        "rmse_orig": rmse_orig,
        "mae_orig": mae_orig,
        "r2_orig": r2_orig,
    }

