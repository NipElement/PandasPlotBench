import json
from pathlib import Path
import pandas as pd
from collections import defaultdict
import re

# Define all libraries to process
LIBRARIES = ["matplotlib", "seaborn", "plotly"]

# Use your preferred error_rate_path
# Remove the original ATTEMPT_TO_PRINT variable since we'll iterate through all attempts
# error_rate_path = Path("eval_results/qwen2_5_7b_coder_stage4_lr5e6_self_debug.json")
# error_rate_path = Path("eval_results/baseline_self_debug.json")
# error_rate_path = Path("eval_results/api_model_self_debug.json")
# error_rate_path = Path("eval_results/qwen2_5_3b_coder_python_200K_lr5e6.json")
error_rate_path = Path("eval_results/qwen2_5_7b_coder_ablation.json")
# error_rate_path = Path("eval_results/llama3_1_8b_instruct_python_200K_lr5e6.json")
# error_rate_path = Path("eval_results/llama3_1_8b_instruct_verify.json")
# error_rate_path = Path("eval_results/baseline_self_debug_qwen.json")

def calculate_success_rates(data):
    # Dictionary to store results for all libraries
    all_results = defaultdict(lambda: defaultdict(dict))
    
    for key, value in data.items():
        if isinstance(value, float):
            continue
            
        ckpt_path, lib = key.rsplit("_", 1)
        if lib not in LIBRARIES:
            continue
            
        model_name = Path(ckpt_path).name
        total_cases = value["total_num"]
        
        # Calculate initial success rate
        init_success = ((total_cases - value["execution_error_num"]) / total_cases) * 100
        all_results[lib][model_name]["Init"] = init_success
        
        # Calculate success rates for each attempt
        if "debug_attempts" in value:
            for attempt in range(3):  # A0, A1, A2
                attempt_key = f"attempt_{attempt}"
                if attempt_key in value["debug_attempts"]:
                    attempt_stats = value["debug_attempts"][attempt_key]
                    success_rate = ((total_cases - attempt_stats["execution_error_num"]) / total_cases) * 100
                    all_results[lib][model_name][f"Post A{attempt}"] = success_rate
                else:
                    all_results[lib][model_name][f"Post A{attempt}"] = None
    
    return all_results

def create_formatted_table(results, lib):
    df = pd.DataFrame.from_dict(results[lib], orient="index")
    
    # Sort by checkpoint number if available
    def extract_sort_key(name):
        match = re.search(r"checkpoint-(\d+)", name)
        return int(match.group(1)) if match else float('inf')
    
    df.index.name = "Model"
    df.reset_index(inplace=True)
    df["sort_key"] = df["Model"].apply(extract_sort_key)
    df = df.sort_values("sort_key").drop(columns="sort_key")
    
    # Rename columns
    columns = ["Model", "Init", "Post A0", "Post A1", "Post A2"]
    df = df[columns]
    
    # Round values
    for col in df.columns[1:]:
        df[col] = df[col].round(2)
    
    return df

def main():
    # Load data
    with open(error_rate_path, "r") as f:
        data = json.load(f)
    
    # Calculate success rates for all libraries
    all_results = calculate_success_rates(data)
    
    # Print results for each library
    for lib in LIBRARIES:
        print(f"\n=== {lib.upper()} Execution Success Rates (%) ===")
        df = create_formatted_table(all_results, lib)
        print(df.to_markdown(index=False))
        print("\n")

if __name__ == "__main__":
    main()
