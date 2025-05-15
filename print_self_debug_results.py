import json
from pathlib import Path
import pandas as pd
from collections import defaultdict
import re

TARGET_LIB = "matplotlib"  # matplotlib, seaborn, plotly 
# Remove the original ATTEMPT_TO_PRINT variable since we'll iterate through all attempts
# error_rate_path = Path("eval_results/qwen2_5_7b_coder_stage4_lr5e6_self_debug.json")
# error_rate_path = Path("eval_results/baseline_self_debug.json")
# error_rate_path = Path("eval_results/baseline_self_debug_supp.json")
# error_rate_path = Path("eval_results/api_model_self_debug.json")
# error_rate_path = Path("eval_results/qwen2_5_3b_coder_python_200K_lr5e6.json")
# error_rate_path = Path("eval_results/llama3_1_8b_instruct_python_200K_lr5e6.json")
# error_rate_path = Path("eval_results/llama3_1_8b_instruct_verify.json")
error_rate_path = Path("eval_results/baseline_self_debug_qwen.json")

with open(error_rate_path, "r") as f:
    data = json.load(f)

# Function to process results for a specific attempt
def process_attempt(data, attempt_num):
    results = defaultdict(dict)
    for key, value in data.items():
        if isinstance(value, float):
            continue
        ckpt_path, lib = key.rsplit("_", 1)
        if lib != TARGET_LIB:
            continue
            
        model_name = Path(ckpt_path).name
        total_cases = value["total_num"]
        
        # Initial error rates
        exec_errors = value["execution_error_num"]
        plot_errors = value["incorrect_plot_num"]
        results[model_name]["ExecErr"] = (exec_errors / total_cases) * 100
        results[model_name]["PlotErr"] = (plot_errors / total_cases) * 100
        
        if "debug_attempts" in value:
            attempt_key = f"attempt_{attempt_num}"
            if attempt_key in value["debug_attempts"]:
                attempt_stats = value["debug_attempts"][attempt_key]
                
                if attempt_num == 0:
                    prev_exec_errors = exec_errors
                    prev_plot_errors = plot_errors
                else:
                    prev_attempt_key = f"attempt_{attempt_num-1}"
                    prev_stats = value["debug_attempts"][prev_attempt_key]
                    prev_exec_errors = prev_stats["execution_error_num"]
                    prev_plot_errors = prev_stats["incorrect_plot_num"]
                    results[model_name]["PrevExec"] = (prev_stats["execution_error_num"] / total_cases) * 100
                    results[model_name]["PrevPlot"] = (prev_stats["incorrect_plot_num"] / total_cases) * 100
                
                if prev_exec_errors > 0:
                    results[model_name]["FixExec"] = ((prev_exec_errors - attempt_stats["execution_error_num"]) / prev_exec_errors) * 100
                else:
                    results[model_name]["FixExec"] = 100
                    
                if prev_plot_errors > 0:
                    results[model_name]["FixPlot"] = ((prev_plot_errors - attempt_stats["incorrect_plot_num"]) / prev_plot_errors) * 100
                else:
                    results[model_name]["FixPlot"] = 100
                
                results[model_name]["PostExec"] = (attempt_stats["execution_error_num"] / total_cases) * 100
                results[model_name]["PostPlot"] = (attempt_stats["incorrect_plot_num"] / total_cases) * 100
    
    return results

# Print results for each attempt
for attempt in range(3):  # 0, 1, 2
    print(f"\n=== Processing Attempt {attempt} ===")
    results = process_attempt(data, attempt)
    
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
    
    # Define column names based on attempt number
    if attempt == 0:
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
        column_names = {
            "Model": "Model",
            "PrevExec": f"A{attempt-1} Post Exec(%)",
            "PrevPlot": f"A{attempt-1} Post Plot(%)",
            "FixExec": f"A{attempt} Fix Exec(%)",
            "FixPlot": f"A{attempt} Fix Plot(%)",
            "PostExec": f"A{attempt} Post Exec(%)",
            "PostPlot": f"A{attempt} Post Plot(%)"
        }
    
    # Select and rename columns
    df = df[list(column_names.keys())]
    df = df.rename(columns=column_names)
    df = df.round(2)
    
    # Print results
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(f"\nResults for {TARGET_LIB} - Attempt {attempt}:")
    print(df.to_markdown(index=False))