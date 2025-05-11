import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

def check_json_file(json_path: Path) -> List[Tuple[str, str]]:
    issues = []
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 直接访问 debug_info 字段
        debug_info = data.get('debug_info', {})
        
        if not debug_info:
            issues.append(('file_error', "debug_info is None or empty"))
            return issues
        
        # Check each debug round
        for round_id, round_data in debug_info.items():
            if not round_data:  # Skip if round_data is None
                continue
                
            if 'attempts' not in round_data:
                continue
                
            # Check each attempt in this round
            for attempt_id, attempt_data in round_data['attempts'].items():
                # Check for empty model_response
                # if 'model_response' in attempt_data and not attempt_data['model_response']:
                #     issues.append((
                #         round_id,
                #         f"Empty model_response in attempt {attempt_id}"
                #     ))
                
                # Check for empty error but has_plot is False
                if ('error' in attempt_data and attempt_data['error'] == "" and
                    'has_plot' in attempt_data and attempt_data['has_plot'] is False):
                    issues.append((
                        round_id,
                        f"Empty error but has_plot=False in attempt {attempt_id}"
                    ))
                        
    except Exception as e:
        print(f"Error processing {json_path}: {str(e)}")
        issues.append(('file_error', f"Error processing {json_path}: {str(e)}"))
        
    return issues

def verify_eval_results(eval_dir: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Verify all json files in the evaluation directory
    
    Returns:
        Dict mapping file paths to lists of (task_id, issue_description) tuples
    """
    eval_path = Path(eval_dir)
    all_issues = {}
    
    # Find all json files recursively
    for json_path in eval_path.rglob('*.json'):
        # print(f"Checking {json_path}...")
        issues = check_json_file(json_path)
        if issues:
            all_issues[str(json_path)] = issues
            
    return all_issues

def main():
    # 指定评测结果目录
    # eval_dir = "/data/yuansheng/PandasPlotBench/eval_results/qwen2_5_7b_coder_stage4_lr5e6_self_debug"
    eval_dir = "/data/yuansheng/PandasPlotBench/eval_results/baseline_self_debug"
    
    print("开始验证评测结果...")
    issues = verify_eval_results(eval_dir)
    
    # 输出结果
    if not issues:
        print("✅ 没有发现任何问题！")
    else:
        print("\n发现以下问题：")
        for file_path, file_issues in issues.items():
            print(f"\n文件: {file_path}")
            for task_id, issue in file_issues:
                print(f"  - Task {task_id}: {issue}")
                
    # 输出统计信息
    total_files = sum(1 for _ in Path(eval_dir).rglob('*.json'))
    files_with_issues = len(issues)
    print(f"\n统计信息:")
    print(f"- 检查的文件总数: {total_files}")
    print(f"- 有问题的文件数: {files_with_issues}")

if __name__ == "__main__":
    main()
