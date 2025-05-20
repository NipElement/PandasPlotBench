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
    # base_path = Path("/data/yuansheng/PandasPlotBench/eval_results/qwen2_5_7b_coder_stage4_lr5e6_self_debug")
    base_path = Path("/data/yuansheng/PandasPlotBench/eval_results/VisCoder-7B")
    libs = ["matplotlib", "plotly", "seaborn"]
    
    # Get all available models
    models = get_available_models(base_path)
    
    for model_name in models:
        print(f"\nAnalyzing model: {model_name}")
        
        # Process each library separately
        for lib in libs:
            lib_errors = Counter()
            attempt_errors = {i: Counter() for i in range(3)}
            error_types = set()
            
            file_path = base_path / lib / f"results_{model_name}_{lib}_head_0.json"
            if not file_path.exists():
                print(f"Warning: {file_path} does not exist")
                continue
            
            print(f"\n{lib.upper()} Results:")
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Process each sample
            for idx in data['id'].keys():
                # Count initial errors
                if data['error'].get(idx):
                    error_type = extract_error_type(data['error'][idx])
                    lib_errors[error_type] += 1
                    error_types.add(error_type)
                
                # Count debug attempt errors
                if 'debug_info' in data and data['debug_info'].get(idx):
                    debug_info = data['debug_info'][idx]
                    for attempt_id in range(3):
                        if str(attempt_id) not in debug_info['attempts']:
                            break
                        
                        attempt = debug_info['attempts'][str(attempt_id)]
                        if attempt.get('error'):
                            error_type = extract_error_type(attempt['error'])
                            attempt_errors[attempt_id][error_type] += 1
                            error_types.add(error_type)
            
            # Sort error types alphabetically
            sorted_error_types = sorted(error_types)
            
            # Create header format
            header_format = "{:<30} | {:<10} | {:<10} | {:<10} | {:<10}"
            
            # Print table header
            print("\n" + header_format.format(
                "Error Type", "Initial", "Attempt 0", "Attempt 1", "Attempt 2"))
            print("-" * 80)
            
            # Print statistics for each error type
            for error_type in sorted_error_types:
                counts = [
                    str(lib_errors[error_type]),
                    str(attempt_errors[0][error_type]),
                    str(attempt_errors[1][error_type]),
                    str(attempt_errors[2][error_type])
                ]
                print(header_format.format(error_type[:30], *counts))
            
            # Print total errors
            print("-" * 80)
            totals = [
                str(sum(lib_errors.values())),
                str(sum(attempt_errors[0].values())),
                str(sum(attempt_errors[1].values())),
                str(sum(attempt_errors[2].values()))
            ]
            print(header_format.format("Total Errors", *totals))

def analyze_model_errors_all():
    # base_path = Path("/data/yuansheng/PandasPlotBench/eval_results/api_model_self_debug")
    base_path = Path("/data/yuansheng/PandasPlotBench/eval_results/baseline_self_debug")
    # base_path = Path("/data/yuansheng/PandasPlotBench/eval_results/baseline_self_debug_qwen")
    # base_path = Path("/data/yuansheng/PandasPlotBench/eval_results/qwen2_5_3b_coder_python_200K_lr5e6")
    # base_path = Path("/data/yuansheng/PandasPlotBench/eval_results/qwen2_5_7b_coder_stage4_lr5e6_self_debug")
    # base_path = Path("/data/yuansheng/PandasPlotBench/eval_results/VisCoder-7B")
    libs = ["matplotlib", "plotly", "seaborn"]
    
    models = get_available_models(base_path)
    
    for model_name in models:
        print(f"\n{model_name}")
        
        lib_errors = {
            lib: {
                'initial': Counter(),
                'attempts': {
                    0: Counter(),
                    1: Counter(),
                    2: Counter()
                }
            } for lib in libs
        }
        all_error_types = set()
        
        # Collect error statistics
        for lib in libs:
            file_path = base_path / lib / f"results_{model_name}_{lib}_head_0.json"
            if not file_path.exists():
                print(f"Warning: {file_path} does not exist")
                continue
                
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            for idx in data['id'].keys():
                # Initial errors
                if data['error'].get(idx):
                    error_type = extract_error_type(data['error'][idx])
                    lib_errors[lib]['initial'][error_type] += 1
                    all_error_types.add(error_type)
                
                # Debug attempts errors
                if 'debug_info' in data and data['debug_info'].get(idx):
                    debug_info = data['debug_info'][idx]
                    for attempt_id in range(3):
                        if str(attempt_id) not in debug_info['attempts']:
                            break
                        
                        attempt = debug_info['attempts'][str(attempt_id)]
                        if attempt.get('error'):
                            error_type = extract_error_type(attempt['error'])
                            lib_errors[lib]['attempts'][attempt_id][error_type] += 1
                            all_error_types.add(error_type)
        
        # Sort error types alphabetically
        sorted_error_types = sorted(all_error_types)
        
        # Print table header
        print("\n{:<30} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10}".format(
            "Error Type", 
            "MPL Init", "MPL A0", "MPL A1", "MPL A2",
            "PLT Init", "PLT A0", "PLT A1", "PLT A2",
            "SNS Init", "SNS A0", "SNS A1", "SNS A2"))
        print("-" * 160)
        
        # Print statistics for each error type
        for error_type in sorted_error_types:
            counts = []
            for lib in libs:
                counts.extend([
                    str(lib_errors[lib]['initial'][error_type]),
                    str(lib_errors[lib]['attempts'][0][error_type]),
                    str(lib_errors[lib]['attempts'][1][error_type]),
                    str(lib_errors[lib]['attempts'][2][error_type])
                ])
            
            print("{:<30} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10}".format(
                error_type[:30], *counts))
        
        # Print total errors
        print("-" * 160)
        totals = []
        for lib in libs:
            totals.extend([
                str(sum(lib_errors[lib]['initial'].values())),
                str(sum(lib_errors[lib]['attempts'][0].values())),
                str(sum(lib_errors[lib]['attempts'][1].values())),
                str(sum(lib_errors[lib]['attempts'][2].values()))
            ])
        
        print("{:<30} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10}".format(
            "Total Errors", *totals))

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
    analyze_model_errors_all()
    # inspect_specific_errors()
    # inspect_error_type()