import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
import json

def collect_failed_cells(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["error"] != "") | (~df["has_plot"])].copy()


class DebugSession:
    def __init__(self, model_name: str, output_dir: Path, max_attempts: int = 1):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.max_attempts = max_attempts
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_debug_conversation(
        self,
        row: pd.Series,
        attempt_id: int,
        previous_attempts: List[dict] = None,
    ) -> List[dict]:
        if attempt_id == 0:
            return self._generate_single_turn_conversation(row)
        else:
            return self._generate_multi_turn_conversation(row, previous_attempts or [])

    @staticmethod
    def _generate_single_turn_conversation(row: pd.Series) -> List[dict]:
        task = row.get('task', '') or '\n'.join(
            str(row[f]) for f in row.index if f.startswith("task__") and pd.notna(row[f])
        )
        return [
            {"content": task},
            {"content": f"```python\n{row['code']}\n```", "is_assistant": True},
            {"content": f"The above code failed with the following error:\n{row['error']}\nPlease provide a corrected version enclosed in ```python ... ```."}
        ]

    @staticmethod
    @staticmethod
    def _generate_multi_turn_conversation(row: pd.Series, history: List[dict]) -> List[dict]:
        conversation = history[-1]['debug_conversation'].copy()
        
        for attempt in history:
            conversation.append({"content": f"```python\n{attempt['code']}\n```", "is_assistant": True})
            conversation.append({"content": f"The above code still failed with the following error:\n{attempt['error']}\nPlease provide a corrected version enclosed in ```python ... ```."})
        
        return conversation

def update_error_rate_statistics(
    error_rate_file: Path,
    model_name: str,
    plotting_lib: str,
    dataset_df: pd.DataFrame,
) -> None:
    """更新错误率统计信息并写入文件"""
    if error_rate_file.exists():
        with open(error_rate_file, "r") as f:
            error_rates = json.load(f)
    else:
        error_rates = {}
    
    # 获取debug案例
    debug_cases = dataset_df[dataset_df['debug_info'].notna()]
    if len(debug_cases) == 0:
        print("[DEBUG] No debug cases found")
        return
        
    # 记录原始评估数据
    record_key = f"{model_name}_{plotting_lib.split(' ')[0]}"
    error_rates[record_key] = {
        "total_num": len(dataset_df),
        "execution_error_num": len(dataset_df[dataset_df["error"] != ""]),
        "incorrect_plot_num": len(dataset_df[~dataset_df["has_plot"]]),
    }
    
    # 获取最大尝试次数
    max_attempts = max(
        max(int(attempt) for attempt in row['debug_info']['attempts'].keys())
        for _, row in debug_cases.iterrows()
    ) + 1
    
    # 记录每次尝试的结果
    debug_attempts = {}
    remaining_cases = len(debug_cases)  # 初始需要调试的案例数
    
    for attempt_idx in range(max_attempts):
        execution_error_num = 0
        incorrect_plot_num = 0
        
        # 统计当前attempt的错误数
        for _, row in debug_cases.iterrows():
            debug_info = row['debug_info']
            attempt = str(attempt_idx)
            if attempt not in debug_info['attempts']:
                continue
                
            attempt_info = debug_info['attempts'][attempt]
            if attempt_info['error'] != "":
                execution_error_num += 1
            if not attempt_info['has_plot']:
                incorrect_plot_num += 1
        
        debug_attempts[f"attempt_{attempt_idx}"] = {
            "total_num": remaining_cases,  # 使用当前还需要调试的案例数
            "execution_error_num": execution_error_num,
            "incorrect_plot_num": incorrect_plot_num
        }
        
        # 更新下一轮需要调试的案例数
        remaining_cases = execution_error_num  # 因为下一轮只需要处理这一轮未修复的案例
    
    error_rates[record_key]["debug_attempts"] = debug_attempts
    
    with open(error_rate_file, "w") as f:
        json.dump(error_rates, f, indent=4)
    
    print(f"[DEBUG] Error rate statistics saved to {error_rate_file}")

