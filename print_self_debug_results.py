import json
from pathlib import Path
import pandas as pd
from collections import defaultdict
import re

TARGET_LIB = "plotly"  # "seaborn", "plotly" 

# error_rate_path = Path("/data/yuansheng/PandasPlotBench/eval_results/baseline_self_debug.json")
error_rate_path = Path("/data/yuansheng/PandasPlotBench/eval_results/qwen2_5_7b_coder_stage4_lr5e6_self_debug.json")

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
    plot_errors = value["incorrect_code_num"]
    results[model_name]["ExecErr"] = (exec_errors / total_cases) * 100
    results[model_name]["PlotErr"] = (plot_errors / total_cases) * 100
    
    # Debug相关统计
    if "debug_attempts" in value and "attempt_0" in value["debug_attempts"]:
        debug_stats = value["debug_attempts"]["attempt_0"]
        
        # Debug后仍存在的错误数
        remaining_exec = debug_stats["execution_error_num"]
        remaining_plot = debug_stats["incorrect_plot_num"]
        
        # Debug修复成功率计算 (原始错误-修复后仍存在错误)/原始错误
        results[model_name]["FixExec"] = ((exec_errors - remaining_exec) / exec_errors * 100) if exec_errors > 0 else 0
        results[model_name]["FixPlot"] = ((plot_errors - remaining_plot) / plot_errors * 100) if plot_errors > 0 else 0
        
        # Debug后的错误率：修复后仍存在的错误/总样本数
        results[model_name]["PostExecErr"] = (remaining_exec / total_cases * 100)
        results[model_name]["PostPlotErr"] = (remaining_plot / total_cases * 100)

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

# 重命名列以使其更清晰
column_names = {
    "Model": "Model",
    "ExecErr": "Exec Error",
    "PlotErr": "Plot Error",
    "FixExec": "Fixed Exec",
    "FixPlot": "Fixed Plot",
    "PostExecErr": "Post-Debug Exec",
    "PostPlotErr": "Post-Debug Plot"
}

df = df.rename(columns=column_names)
df = df.round(2)

# 打印时强制显示所有列
pd.set_option('display.max_columns', None)
print(f"\nResults for {TARGET_LIB}:")
print(df.to_markdown(index=False))