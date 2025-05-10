import json
from pathlib import Path
import pandas as pd
from collections import defaultdict
import re

TARGET_LIB = "seaborn"  # matplotlib, seaborn, plotly 
ATTEMPT_TO_PRINT = 2  # None表示只打印初始结果，0/1/2表示打印对应attempt的修复结果
# error_rate_path = Path("eval_results/qwen2_5_7b_coder_stage4_lr5e6_self_debug.json")
error_rate_path = Path("eval_results/baseline_self_debug.json")

with open(error_rate_path, "r") as f:
    data = json.load(f)

results = defaultdict(dict)
for key, value in data.items():
    if isinstance(value, float):
        continue
    ckpt_path, lib = key.rsplit("_", 1)
    # 只处理指定的库
    if lib != TARGET_LIB:
        continue
        
    model_name = Path(ckpt_path).name
    total_cases = value["total_num"]
    
    # 原始错误率
    exec_errors = value["execution_error_num"]
    plot_errors = value["incorrect_plot_num"]
    results[model_name]["ExecErr"] = (exec_errors / total_cases) * 100
    results[model_name]["PlotErr"] = (plot_errors / total_cases) * 100
    
    # 如果需要打印某个attempt的结果
    if ATTEMPT_TO_PRINT is not None and "debug_attempts" in value:
        attempt_key = f"attempt_{ATTEMPT_TO_PRINT}"
        if attempt_key in value["debug_attempts"]:
            attempt_stats = value["debug_attempts"][attempt_key]
            
            if ATTEMPT_TO_PRINT == 0:
                # 第一轮attempt，对比initial
                prev_exec_errors = exec_errors
                prev_plot_errors = plot_errors
            else:
                # 其他轮次，对比上一轮
                prev_attempt_key = f"attempt_{ATTEMPT_TO_PRINT-1}"
                prev_stats = value["debug_attempts"][prev_attempt_key]
                prev_exec_errors = prev_stats["execution_error_num"]
                prev_plot_errors = prev_stats["incorrect_plot_num"]
                # 保存上一轮的post rate
                results[model_name]["PrevExec"] = (prev_stats["execution_error_num"] / total_cases) * 100
                results[model_name]["PrevPlot"] = (prev_stats["incorrect_plot_num"] / total_cases) * 100
            
            # 计算相对于上一轮的修复率
            if prev_exec_errors > 0:
                results[model_name]["FixExec"] = ((prev_exec_errors - attempt_stats["execution_error_num"]) / prev_exec_errors) * 100
            else:
                results[model_name]["FixExec"] = 100
                
            if prev_plot_errors > 0:
                results[model_name]["FixPlot"] = ((prev_plot_errors - attempt_stats["incorrect_plot_num"]) / prev_plot_errors) * 100
            else:
                results[model_name]["FixPlot"] = 100
            
            # 计算修复后的整体错误率
            results[model_name]["PostExec"] = (attempt_stats["execution_error_num"] / total_cases) * 100
            results[model_name]["PostPlot"] = (attempt_stats["incorrect_plot_num"] / total_cases) * 100

df = pd.DataFrame.from_dict(results, orient="index")
df.index.name = "Model"
df.reset_index(inplace=True)

def extract_sort_key(name):
    match = re.search(r"checkpoint-(\d+)", name)
    if match:
        return int(match.group(1))
    else:
        return float('inf')

df["sort_key"] = df["Model"].apply(extract_sort_key)
df = df.sort_values("sort_key").drop(columns="sort_key")

# 根据ATTEMPT_TO_PRINT选择列名和数据
if ATTEMPT_TO_PRINT is None:
    column_names = {
        "Model": "Model",
        "ExecErr": "Exec Error(%)",
        "PlotErr": "Plot Error(%)"
    }
else:
    if ATTEMPT_TO_PRINT == 0:
        column_names = {
            "Model": "Model",
            "ExecErr": "Init Exec(%)",
            "PlotErr": "Init Plot(%)",
            "FixExec": "A0 Fix Exec(%)",
            "FixPlot": "A0 Fix Plot(%)",
            "PostExec": "A0 Post Exec(%)",
            "PostPlot": "A0 Post Plot(%)"
        }
    else:
        # 其他轮次只显示上一轮的post rate和当前轮次的修复情况
        column_names = {
            "Model": "Model",
            "PrevExec": f"A{ATTEMPT_TO_PRINT-1} Post Exec(%)",
            "PrevPlot": f"A{ATTEMPT_TO_PRINT-1} Post Plot(%)",
            "FixExec": f"A{ATTEMPT_TO_PRINT} Fix Exec(%)",
            "FixPlot": f"A{ATTEMPT_TO_PRINT} Fix Plot(%)",
            "PostExec": f"A{ATTEMPT_TO_PRINT} Post Exec(%)",
            "PostPlot": f"A{ATTEMPT_TO_PRINT} Post Plot(%)"
        }

# 只选择需要显示的列
df = df[list(column_names.keys())]
df = df.rename(columns=column_names)
df = df.round(2)

# 打印时强制显示所有列
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(f"\nResults for {TARGET_LIB}:")
print(df.to_markdown(index=False))