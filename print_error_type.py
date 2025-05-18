import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Set

def extract_error_type(error_text: str) -> str:
    if not error_text or error_text == "":
        return ""
        
    if "Traceback" in error_text:
        return error_text.split('Traceback')[0].replace('-', '').replace('\n', '').strip()
    
    if "SyntaxError" in error_text:
        return "SyntaxError"
    
    return error_text.strip()

def get_available_models(base_path: Path) -> list:
    # 从matplotlib目录下获取所有结果文件
    result_files = list((base_path / "matplotlib").glob("results_*_matplotlib_head_0.json"))
    # 提取模型名称: results_MODEL_matplotlib_head_0.json -> MODEL
    models = [f.name.split('_')[1] for f in result_files]
    return models

def analyze_model_errors():
    # base_path = Path("/data/yuansheng/PandasPlotBench/eval_results/api_model_self_debug")
    # # base_path = Path("/data/yuansheng/PandasPlotBench/eval_results/baseline_self_debug")
    # base_path = Path("/data/yuansheng/PandasPlotBench/eval_results/baseline_self_debug_qwen")
    # base_path = Path("/data/yuansheng/PandasPlotBench/eval_results/qwen2_5_3b_coder_python_200K_lr5e6")
    base_path = Path("/data/yuansheng/PandasPlotBench/eval_results/qwen2_5_7b_coder_stage4_lr5e6_self_debug")
    libs = ["matplotlib", "plotly", "seaborn"]
    
    # Get all available models
    models = get_available_models(base_path)
    
    for model_name in models:
        print(f"\n{model_name}")
        
        lib_errors: Dict[str, Counter] = {}
        all_error_types: Set[str] = set()
        
        # Collect error statistics for each library
        for lib in libs:
            file_path = base_path / lib / f"results_{model_name}_{lib}_head_0.json"
            if not file_path.exists():
                print(f"Warning: {file_path} does not exist")
                continue
                
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            error_counter = Counter()
            for idx in data['id'].keys():
                if data['error'].get(idx) and data['error'][idx] != "":
                    error_type = extract_error_type(data['error'][idx])
                    error_counter[error_type] += 1
                    all_error_types.add(error_type)
            
            lib_errors[lib] = error_counter
        
        # Sort error types alphabetically
        sorted_error_types = sorted(all_error_types)
        
        # Print table header
        print("\n{:<30} | {:<10} | {:<10} | {:<10}".format(
            "Error Type", "Matplotlib", "Plotly", "Seaborn"))
        print("-" * 70)
        
        # Print statistics for each error type
        for error_type in sorted_error_types:
            counts = []
            for lib in libs:
                count = lib_errors.get(lib, Counter()).get(error_type, 0)
                counts.append(str(count) if count > 0 else "0")
            
            print("{:<30} | {:<10} | {:<10} | {:<10}".format(
                error_type[:30],
                counts[0],
                counts[1],
                counts[2]
            ))
        
        # Print total errors
        print("-" * 70)
        totals = []
        for lib in libs:
            total = sum(lib_errors.get(lib, Counter()).values())
            totals.append(str(total) if total > 0 else "0")
        
        print("{:<30} | {:<10} | {:<10} | {:<10}".format(
            "Total Errors",
            totals[0],
            totals[1],
            totals[2]
        ))

def inspect_specific_errors():
    file_path = Path("/data/yuansheng/PandasPlotBench/eval_results/api_model_self_debug/plotly/results_gpt-4o_plotly_head_0.json")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    for idx in data['id'].keys():
        if data['error'].get(idx):
            error_text = data['error'][idx]
            # 如果error中包含"Cell In"，打印完整信息
            if "Cell In[86], line 42" in error_text:
                print("\n" + "="*50)
                print(f"Sample ID: {data['id'][idx]}")
                print("完整的error信息:")
                print(error_text)
                print("="*50)

def inspect_error_type():
    file_path = Path("/data/yuansheng/PandasPlotBench/eval_results/api_model_self_debug/plotly/results_gpt-4o_plotly_head_0.json")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print("Checking samples with error type 'Error':")
    
    for idx in data['id'].keys():
        if data['error'].get(idx) and data['error'][idx] != "":
            error_type = extract_error_type(data['error'][idx])
            if error_type == "Error":
                print("\n" + "="*50)
                print(f"Sample ID: {data['id'][idx]}")
                print("Complete error message:")
                print(data['error'][idx])
                print("="*50)

if __name__ == "__main__":
    analyze_model_errors()
    # inspect_specific_errors()
    # inspect_error_type()