#!/bin/bash

# 批量微调脚本
# 使用GAT_k2预训练模型对多个数据集进行微调

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置参数
SOURCE_MODEL="./results/GAT_k0/best_model.pth"
DATA_DIR="../datas/others"
RESULTS_BASE_DIR="./results/finetune_k0"
WORK_DIR="/home/huangzy/deep-learning/UPLC/gnn_retention_prediction"

# 数据集列表
DATASETS=(
    "Default-2-BLANCE"
    "Default_Neutral"
    "Default_90%_0807"
    "TEST-1116-07-01"
    "TEST-1014-04"
    "Default-2-3"
    "TEST-0625-3min-1"
    "TEST-0922-03-1"
    "TEST-0625-3min"
    "2MIN_100%_4"
    "Default-2-2"
    "TEST-0922-03"
)

# 微调参数
NUM_EPOCHS=3000
LEARNING_RATE=0.0001
BATCH_SIZE=512
HIDDEN_DIM=128
NUM_LAYERS=3
DROPOUT=0.2
GNN_TYPE="GAT"
TRAIN_RATIO=0.9

# 函数：打印带颜色的消息
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# 函数：检查文件是否存在
check_file() {
    if [ ! -f "$1" ]; then
        print_message $RED "❌ 文件不存在: $1"
        return 1
    fi
    return 0
}

# 函数：创建CSV结果文件头
create_csv_header() {
    local csv_file="$1"
    echo "dataset,status,mae,rmse,r2,train_loss,val_loss,epochs,training_time,error" > "$csv_file"
}

# 函数：追加结果到CSV
append_to_csv() {
    local csv_file="$1"
    local dataset="$2"
    local status="$3"
    local mae="$4"
    local rmse="$5"
    local r2="$6"
    local train_loss="$7"
    local val_loss="$8"
    local epochs="$9"
    local training_time="${10}"
    local error="${11}"
    
    echo "$dataset,$status,$mae,$rmse,$r2,$train_loss,$val_loss,$epochs,$training_time,$error" >> "$csv_file"
}

# 函数：解析training_history.json
parse_results() {
    local json_file="$1"
    if [ -f "$json_file" ]; then
        # 使用python解析JSON
        python3 -c "
import json
import sys
try:
    with open('$json_file', 'r') as f:
        data = json.load(f)
    final_results = data.get('final_results', {})
    train_losses = data.get('train_losses', [])
    val_losses = data.get('val_losses', [])
    
    mae = final_results.get('mae', '')
    rmse = final_results.get('rmse', '')
    r2 = final_results.get('r2', '')
    train_loss = train_losses[-1] if train_losses else ''
    val_loss = val_losses[-1] if val_losses else ''
    epochs = len(train_losses)
    
    print(f'{mae:.2f},{rmse:.2f},{r2:.2f},{train_loss:.2f},{val_loss:.2f},{epochs:.2f}')
except Exception as e:
    print(',,,,,,')
"
    else
        echo ",,,,,"
    fi
}

# 函数：微调单个数据集
finetune_dataset() {
    local dataset="$1"
    local csv_file="$2"
    
    print_message $BLUE "🚀 开始微调数据集: $dataset"
    
    # 检查数据文件
    local data_path="$DATA_DIR/${dataset}.csv"
    if ! check_file "$data_path"; then
        append_to_csv "$csv_file" "$dataset" "failed" "" "" "" "" "" "" "" "data_file_not_found"
        return 1
    fi
    
    # 设置保存目录
    local save_dir="$RESULTS_BASE_DIR/finetune_${dataset}"
    mkdir -p "$save_dir"
    
    # 记录开始时间
    local start_time=$(date +%s)
    
    # 构建命令
    local cmd="python main.py \
        --data_path '$data_path' \
        --save_dir '$save_dir' \
        --model_dir '$SOURCE_MODEL' \
        --finetune False\
        --num_epochs $NUM_EPOCHS \
        --learning_rate $LEARNING_RATE \
        --batch_size $BATCH_SIZE \
        --hidden_dim $HIDDEN_DIM \
        --num_layers $NUM_LAYERS \
        --dropout $DROPOUT \
        --gnn_type $GNN_TYPE \
        --train_ratio $TRAIN_RATIO"
    
    print_message $YELLOW "执行命令: $cmd"
    
    # 运行微调
    if eval $cmd > "$save_dir/training.log" 2>&1; then
        local end_time=$(date +%s)
        local training_time=$((end_time - start_time))
        
        print_message $GREEN "✅ 微调成功: $dataset (耗时: ${training_time}秒)"
        
        # 解析结果
        local results=$(parse_results "$save_dir/training_history.json")
        append_to_csv "$csv_file" "$dataset" "success" $results "$training_time" ""
        
        # 显示结果摘要
        local mae=$(echo $results | cut -d',' -f1)
        local r2=$(echo $results | cut -d',' -f3)
        if [ ! -z "$mae" ] && [ ! -z "$r2" ]; then
            print_message $GREEN "   📊 结果: MAE=$mae, R²=$r2"
        fi
    else
        local end_time=$(date +%s)
        local training_time=$((end_time - start_time))
        
        print_message $RED "❌ 微调失败: $dataset"
        
        # 获取错误信息
        local error_info=$(tail -n 5 "$save_dir/training.log" | tr '\n' ' ' | tr ',' ';')
        append_to_csv "$csv_file" "$dataset" "failed" "" "" "" "" "" "" "$training_time" "$error_info"
    fi
    
    echo ""
}

# 主函数
main() {
    print_message $BLUE "🎯 批量微调脚本启动"
    echo "=========================================="
    echo "源模型: $SOURCE_MODEL"
    echo "数据目录: $DATA_DIR"
    echo "结果目录: $RESULTS_BASE_DIR"
    echo "数据集数量: ${#DATASETS[@]}"
    echo "=========================================="
    
    # # 检查源模型
    # if ! check_file "$SOURCE_MODEL"; then
    #     print_message $RED "❌ 源模型文件不存在，退出"
    #     exit 1
    # fi
    
    # 检查数据目录
    if [ ! -d "$DATA_DIR" ]; then
        print_message $RED "❌ 数据目录不存在: $DATA_DIR"
        exit 1
    fi
    
    # 切换到工作目录
    cd "$WORK_DIR" || {
        print_message $RED "❌ 无法切换到工作目录: $WORK_DIR"
        exit 1
    }
    
    print_message $GREEN "✅ 工作目录: $(pwd)"
    
    # 创建结果目录
    mkdir -p "$RESULTS_BASE_DIR"
    
    # 创建CSV结果文件
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local csv_file="$RESULTS_BASE_DIR/finetune_results_${timestamp}.csv"
    create_csv_header "$csv_file"
    
    print_message $YELLOW "📝 结果将保存到: $csv_file"
    
    # 记录总开始时间
    local total_start_time=$(date +%s)
    
    # 微调每个数据集
    local success_count=0
    local total_count=${#DATASETS[@]}
    
    for i in "${!DATASETS[@]}"; do
        local dataset="${DATASETS[$i]}"
        local progress=$((i + 1))
        
        print_message $BLUE "进度: $progress/$total_count"
        
        if finetune_dataset "$dataset" "$csv_file"; then
            ((success_count++))
        fi
    done
    
    # 计算总耗时
    local total_end_time=$(date +%s)
    local total_training_time=$((total_end_time - total_start_time))
    local hours=$((total_training_time / 3600))
    local minutes=$(((total_training_time % 3600) / 60))
    local seconds=$((total_training_time % 60))
    
    # 输出汇总报告
    echo ""
    print_message $BLUE "📊 批量微调汇总报告"
    echo "=========================================="
    echo "总耗时: ${hours}小时${minutes}分钟${seconds}秒"
    echo "总数据集: $total_count"
    echo "成功数量: $success_count"
    echo "失败数量: $((total_count - success_count))"
    echo "成功率: $(( (success_count * 100) / total_count ))%"
    echo "结果文件: $csv_file"
    echo "=========================================="
    
    # 显示成功的结果
    if [ $success_count -gt 0 ]; then
        print_message $GREEN "✅ 成功微调的数据集:"
        echo ""
        printf "%-20s %-8s %-8s %-8s\n" "Dataset" "MAE" "R²" "RMSE"
        echo "----------------------------------------------------"
        
        # 解析CSV文件显示成功的结果
        tail -n +2 "$csv_file" | while IFS=',' read -r dataset status mae rmse r2 train_loss val_loss epochs training_time error; do
            if [ "$status" = "success" ] && [ ! -z "$mae" ]; then
                printf "%-20s %-8.3f %-8.3f %-8.3f\n" "$dataset" "$mae" "$r2" "$rmse"
            fi
        done
    fi
    
    print_message $BLUE "🎉 批量微调完成！"
}

# 执行主函数
main "$@"
