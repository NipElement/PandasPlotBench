import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

def collect_stats(base_dir: str) -> Dict[str, List[Dict]]:
    """收集所有库的统计信息"""
    base_path = Path(base_dir)
    lib_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    # 用于存储每个模型的统计信息
    model_stats = defaultdict(list)
    
    for lib_dir in lib_dirs:
        lib_name = lib_dir.name
        state_file = lib_dir / "benchmark_stat.jsonl"
        
        if not state_file.exists():
            print(f"警告: {lib_name} 目录下没有找到 state.jsonl")
            continue
            
        # 读取jsonl文件
        with open(state_file, "r") as f:
            for line in f:
                try:
                    stat = json.loads(line.strip())
                    # 添加库信息
                    stat['plotting_lib'] = lib_name
                    model_stats[stat['model']].append(stat)
                except json.JSONDecodeError:
                    print(f"警告: 解析 {lib_name}/state.jsonl 中的一行时出错")
                    continue
    
    return model_stats

def print_model_stats_markdown(model_stats: Dict[str, List[Dict]]):
    """使用Markdown表格格式打印统计信息"""
    # 打印表头
    headers = ["Model", "Library", "Vis Mean", "Task Mean", "Vis Good", "Task Good"]
    # 设置每列的宽度
    widths = [20, 12, 10, 10, 10, 10]
    
    # 打印表头
    header_row = "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths)) + " |"
    print(header_row)
    
    # 打印分隔行
    separator = "|" + "|".join(f"{'-'*w:^{w+2}}" for w in widths) + "|"
    print(separator)
    
    for model, stats_list in model_stats.items():
        # 获取简化的模型名称
        short_model = model.split('/')[-1]
        
        # 按库名称排序
        stats_list.sort(key=lambda x: x['plotting_lib'])
        
        # 打印每个库的统计信息
        for stat in stats_list:
            lib = stat['plotting_lib']
            scores = stat.get('scores', {})
            
            # 获取vis统计
            vis_stats = scores.get('vis', {})
            vis_mean = vis_stats.get('mean', 'N/A')
            vis_good = vis_stats.get('good', 'N/A')
            if vis_good != 'N/A':
                vis_good = f"{vis_good:.2f}"
            
            # 获取task统计
            task_stats = scores.get('task', {})
            task_mean = task_stats.get('mean', 'N/A')
            task_good = task_stats.get('good', 'N/A')
            if task_good != 'N/A':
                task_good = f"{task_good:.2f}"
            
            # 格式化每个字段
            row = [
                f"{short_model:<{widths[0]}}",
                f"{lib:<{widths[1]}}",
                f"{vis_mean:>{widths[2]-1}} ",
                f"{task_mean:>{widths[4]-1}} ",
                f"{vis_good:>{widths[3]-1}} ",
                f"{task_good:>{widths[5]-1}} "
            ]
            
            # 打印行
            print("| " + " | ".join(row) + " |")
        
        # 在不同模型之间添加分隔线
        print(separator)

def main():
    # 设置基础目录
    # base_dir = "/data/yuansheng/PandasPlotBench/eval_results/api_model"
    # base_dir = "/data/yuansheng/PandasPlotBench/eval_results/baseline_self_debug"
    # base_dir = "/data/yuansheng/PandasPlotBench/eval_results/qwen2_5_7b_coder_stage4_lr5e6_self_debug"
    # base_dir = "/data/yuansheng/PandasPlotBench/eval_results/baseline_self_debug_qwen"
    base_dir = "/data/yuansheng/PandasPlotBench/eval_results/qwen2_5_3b_coder_python_200K_lr5e6"
    
    model_stats = collect_stats(base_dir)
    
    # 打印统计信息
    if model_stats:
        print_model_stats_markdown(model_stats)
    else:
        print("没有找到任何统计信息")

if __name__ == "__main__":
    main()