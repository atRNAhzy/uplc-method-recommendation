import matplotlib.pyplot as plt
import numpy as np
from config import PLOT_CONFIG

def get_plot_params(plot_type='iphone_style', **overrides):
    """
    获取绘图参数：先读取default，然后读取特定图片类型参数，最后应用用户覆盖
    """
    # print("Using plot style:", plot_type)
    params = PLOT_CONFIG['default'].copy()
    if plot_type in PLOT_CONFIG:
        params.update(PLOT_CONFIG[plot_type])
    params.update(overrides)
    return params

def setup_axis_style(ax, params):
    """
    设置坐标轴样式（统一函数，避免重复代码）
    """
    ax.tick_params(axis='both', direction='out', length=params['TICK_LENGTH'], 
                   width=params['TICK_WIDTH'], color='black')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.grid(False)
    ax.set_facecolor('white')

def plot_training_curves(train_losses, val_losses, save_path=None, **overrides):
    """
    绘制训练曲线
    """
    params = get_plot_params('training_curves', **overrides)
    
    plt.figure(figsize=params['FIGSIZE'])
    ax = plt.gca()
    
    # 设置坐标轴样式
    setup_axis_style(ax, params)

    # 每10个epoch采样
    epochs = list(range(len(train_losses)))
    sampled_idx = [i for i in range(len(epochs)) if i % 10 == 0]
    sampled_epochs = [epochs[i] for i in sampled_idx]
    sampled_train = [train_losses[i] for i in sampled_idx]
    sampled_val = [val_losses[i] for i in sampled_idx]

    plt.plot(sampled_epochs, sampled_train, label='Training Loss', color=params['IPHONE_COLORS']['train'],
             linewidth=params['LINEWIDTH'])
    plt.plot(sampled_epochs, sampled_val, label='Validation Loss', color=params['IPHONE_COLORS']['val'],
             linewidth=params['LINEWIDTH'])
    plt.xlabel(params['XLABEL'], fontsize=params['LABELSIZE'])
    plt.ylabel(params['YLABEL'], fontsize=params['LABELSIZE'])
    plt.title("", fontsize=params['TITLESIZE'])
    plt.yscale('log')
    plt.legend(fontsize=params['LABELSIZE'])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, facecolor='white', edgecolor='none', dpi=params['SAVE_DPI'], bbox_inches=params['SAVE_BBOX'])


def plot_predictions(results, save_path=None, **overrides):
    """
    绘制三张风格统一的图：主散点图、残差图、拼图（两图一页），全部采用 iPhone 风格。
    """
    import numpy as np
    from sklearn.metrics import r2_score, mean_absolute_error
    import matplotlib.pyplot as plt

    # 取反归一化后的结果优先
    y_true = np.array(results.get('targets_orig', results['targets']))
    y_pred = np.array(results.get('predictions_orig', results['predictions']))

    IPHONE_COLORS = {
        'scatter': '#007AFF',  # iPhone 蓝
        'line': '#AEAEB2',     # iPhone 灰
        'text': '#000000'      # 黑色
    }

    # 计算指标
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    # 1. 主散点图
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.tick_params(axis='both', direction='out', length=6, width=2, labelsize=16)
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(2)
    plt.grid(False)
    plt.scatter(y_true, y_pred, alpha=0.8, s=70, color=IPHONE_COLORS['scatter'], edgecolors='none')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, linestyle='--', color=IPHONE_COLORS['line'], linewidth=3)
    plt.xlabel("True RT (s)", fontsize=18, fontweight='bold')
    plt.ylabel("Predicted RT (s)", fontsize=18, fontweight='bold')
    plt.text(0.05, 0.95, f"R² = {r2:.3g}\nMAE = {mae:.3g}", transform=ax.transAxes, verticalalignment='top', fontsize=16, color=IPHONE_COLORS['text'], bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=5))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace('.png', '_pred_vs_true.png') if save_path.endswith('.png') else save_path + '_pred_vs_true.png', dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()

    # 2. 残差图
    residuals = y_pred - y_true
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.tick_params(axis='both', direction='out', length=6, width=2, labelsize=16)
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(2)
    plt.grid(False)
    plt.scatter(y_true, residuals, alpha=0.8, s=70, color=IPHONE_COLORS['scatter'], edgecolors='none')
    plt.axhline(y=0, linestyle='--', color=IPHONE_COLORS['line'], linewidth=3)
    plt.xlabel("True RT (s)", fontsize=18, fontweight='bold')
    plt.ylabel("Residual (s)", fontsize=18, fontweight='bold')
    plt.text(0.05, 0.95, f"R² = {r2:.3g}\nMAE = {mae:.3g}", transform=ax.transAxes, verticalalignment='top', fontsize=16, color=IPHONE_COLORS['text'], bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=5))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace('.png', '_residual.png') if save_path.endswith('.png') else save_path + '_residual.png', dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()

    # 3. 拼图（两图一页）
    plt.figure(figsize=(12, 6))
    # 左：主散点
    plt.subplot(1, 2, 1)
    ax1 = plt.gca()
    ax1.tick_params(axis='both', direction='out', length=6, width=2, labelsize=16)
    for spine in ['top', 'right', 'bottom', 'left']:
        ax1.spines[spine].set_visible(True)
        ax1.spines[spine].set_linewidth(2)
    plt.grid(False)
    plt.scatter(y_true, y_pred, alpha=0.8, s=70, color=IPHONE_COLORS['scatter'], edgecolors='none')
    plt.plot(lims, lims, linestyle='--', color=IPHONE_COLORS['line'], linewidth=3)
    plt.xlabel("True RT (s)", fontsize=18, fontweight='bold')
    plt.ylabel("Predicted RT (s)", fontsize=18, fontweight='bold')
    plt.text(0.05, 0.95, f"R² = {r2:.3g}\nMAE = {mae:.3g}", transform=ax1.transAxes, verticalalignment='top', fontsize=16, color=IPHONE_COLORS['text'], bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=5))
    # 右：残差
    plt.subplot(1, 2, 2)
    ax2 = plt.gca()
    ax2.tick_params(axis='both', direction='out', length=6, width=2, labelsize=16)
    for spine in ['top', 'right', 'bottom', 'left']:
        ax2.spines[spine].set_visible(True)
        ax2.spines[spine].set_linewidth(2)
    plt.grid(False)
    plt.scatter(y_true, residuals, alpha=0.8, s=70, color=IPHONE_COLORS['scatter'], edgecolors='none')
    plt.axhline(y=0, linestyle='--', color=IPHONE_COLORS['line'], linewidth=3)
    plt.xlabel("True RT (s)", fontsize=18, fontweight='bold')
    plt.ylabel("Residual (s)", fontsize=18, fontweight='bold')
    plt.text(0.05, 0.95, f"R² = {r2:.3g}\nMAE = {mae:.3g}", transform=ax2.transAxes, verticalalignment='top', fontsize=16, color=IPHONE_COLORS['text'], bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=5))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace('.png', '_both.png') if save_path.endswith('.png') else save_path + '_both.png', dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()


def save_predictions(results, save_path):
    """
    保存预测结果到CSV文件
    """
    import pandas as pd
    
    df = pd.DataFrame({
        'SMILES': results['smiles'] if results['smiles'] else [''] * len(results['predictions']),
        'True_RT': results['targets'],
        'Predicted_RT': results['predictions'],
        'Residual': np.array(results['predictions']) - np.array(results['targets'])
    })
    
    df.to_csv(save_path, index=False)
    # print(f"Predictions saved to {save_path}")


def plot_scatter_and_residuals(y_true, y_pred, base_name, save_folder):
    """
    使用 iPhone 风格绘制预测散点图和残差图，完全复制用户提供的绘图样式
    参数从 config.py 中读取
    """
    import os
    from sklearn.metrics import r2_score, mean_absolute_error
    
    params = get_plot_params('iphone_style')
    scatter_params = get_plot_params('iphone_scatter')
    residual_params = get_plot_params('iphone_residual')
    
    # 1) 散点图
    plt.figure(figsize=params['FIGSIZE'])
    ax = plt.gca()
    ax.tick_params(axis='both', direction='out', length=params['TICK_LENGTH'], 
                   width=params['TICK_WIDTH'], labelsize=params['LABELSIZE'])
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)
    plt.grid(False)

    plt.scatter(
        y_true, y_pred,
        alpha=params['ALPHA'],
        s=params['SCATTER_SIZE'],
        color=params['IPHONE_COLORS']['scatter']
    )
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    plt.plot(lims, lims, linestyle='--', color=params['IPHONE_COLORS']['line'], 
             linewidth=params['LINEWIDTH'])

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    plt.xlabel(scatter_params['XLABEL'], fontsize=params['LABELSIZE'])
    plt.ylabel(scatter_params['YLABEL'], fontsize=params['LABELSIZE'])
    plt.text(
        scatter_params['ANNOT_POS'][0], scatter_params['ANNOT_POS'][1],
        f"R² = {r2:.3g}\nMAE = {mae:.3g}",
        transform=ax.transAxes, va='top',
        fontsize=scatter_params['ANNOT_FONT'], color=params['IPHONE_COLORS']['text']
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"scatter.png"), 
                facecolor='white', dpi=params['SAVE_DPI'], bbox_inches=params['SAVE_BBOX'])
    plt.close()

    # 2) 残差图
    residuals = y_pred - y_true
    plt.figure(figsize=params['FIGSIZE'])
    ax = plt.gca()
    ax.tick_params(axis='both', direction='out', length=params['TICK_LENGTH'], 
                   width=params['TICK_WIDTH'], labelsize=params['LABELSIZE'])
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)
    plt.grid(False)

    plt.scatter(
        y_pred, residuals,
        alpha=params['ALPHA'],
        s=params['SCATTER_SIZE'],
        color=params['IPHONE_COLORS']['scatter']
    )
    plt.axhline(y=0, linestyle='--', color=params['IPHONE_COLORS']['line'], 
                linewidth=params['LINEWIDTH'])

    plt.xlabel(residual_params['XLABEL'], fontsize=params['LABELSIZE'])
    plt.ylabel(residual_params['YLABEL'], fontsize=params['LABELSIZE'])
    plt.text(
        residual_params['ANNOT_POS'][0], residual_params['ANNOT_POS'][1],
        f"R² = {r2:.3g}\nMAE = {mae:.3g}",
        transform=ax.transAxes, va='top',
        fontsize=residual_params['ANNOT_FONT'], color=params['IPHONE_COLORS']['text']
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"residuals.png"), 
                facecolor='white', dpi=params['SAVE_DPI'], bbox_inches=params['SAVE_BBOX'])
    plt.close()


def save_training_history_and_dataset(history, data_path, save_dir):
    """
    保存训练历史到JSON，并检测/复制数据集到保存目录
    :param history: dict，训练历史和配置信息
    :param data_path: str，原始数据集路径
    :param save_dir: str，保存目录
    """
    import os
    import shutil
    import json
    dataset_name = os.path.basename(data_path)
    
    dataset_dir = "/home/huangzy/UPLC/论文数据/dataset"
    os.makedirs(dataset_dir, exist_ok=True)

    # 检查数据集是否存在，如果不存在则复制
    if not os.path.exists(os.path.join(dataset_dir, dataset_name)):
        shutil.copy(data_path, os.path.join(dataset_dir, dataset_name))

    # 保存训练历史
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
