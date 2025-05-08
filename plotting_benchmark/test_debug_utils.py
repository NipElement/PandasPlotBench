from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd
import json
from omegaconf import DictConfig
import os

@dataclass
class DebugAttempt:
    """调试尝试的信息"""
    attempt_id: int
    code: str
    error: str
    has_plot: bool
    plots_generated: List[str]
    model_response: str
    conversation: List[Dict]

class DebugSession:
    """调试会话，管理整个调试过程"""
    def __init__(self, 
                 model_name: str,
                 output_dir: Path,
                 max_attempts: int = 1):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.max_attempts = max_attempts
        self.current_attempt = 0
        self.debug_info = {}  # item_id -> List[DebugAttempt]
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_attempt_notebook_path(self, attempt_id: int) -> Path:
        """获取特定尝试的notebook保存路径"""
        return self.output_dir / f"debug_attempt_{attempt_id}.ipynb"
    
    def get_attempt_result_path(self, attempt_id: int) -> Path:
        """获取特定尝试的结果保存路径"""
        return self.output_dir / f"debug_results_attempt_{attempt_id}.json"
    
    def load_previous_attempt(self, attempt_id: int) -> Optional[pd.DataFrame]:
        """加载之前尝试的结果"""
        result_path = self.get_attempt_result_path(attempt_id)
        if result_path.exists():
            try:
                return pd.read_json(result_path)
            except Exception as e:
                print(f"[WARNING] Failed to load previous attempt {attempt_id}: {e}")
                return None
        return None
    
    def generate_debug_conversation(self, 
                                  row: pd.Series, 
                                  attempt_id: int,
                                  previous_attempts: List[DebugAttempt] = None) -> List[Dict]:
        """根据尝试次数生成对应的调试对话"""
        if attempt_id == 0:
            return self._generate_initial_conversation(row)
        else:
            return self._generate_follow_up_conversation(row, previous_attempts)
    
    @staticmethod
    def _generate_initial_conversation(row: pd.Series) -> List[Dict]:
        """生成初始调试对话"""
        task = row.get('task', '')
        if not task:
            task_fields = [f for f in row.index if f.startswith('task__')]
            task = '\n'.join(str(row[f]) for f in task_fields if pd.notna(row[f]))
            
        return [
            {"content": task},
            {"content": f"```python\n{row['code']}\n```", "is_assistant": True},
            {"content": f"The above code failed with the following error:\n{row['error']}\nPlease provide a corrected version enclosed in ```python ... ```"}
        ]
    
    @staticmethod
    def _generate_follow_up_conversation(row: pd.Series, 
                                       previous_attempts: List[DebugAttempt]) -> List[Dict]:
        """生成后续调试对话"""
        task = row.get('task', '')
        if not task:
            task_fields = [f for f in row.index if f.startswith('task__')]
            task = '\n'.join(str(row[f]) for f in task_fields if pd.notna(row[f]))
            
        conversation = [{"content": task}]
        
        for attempt in previous_attempts:
            conversation.extend([
                {"content": f"```python\n{attempt.code}\n```", "is_assistant": True},
                {"content": f"The above code still failed with the following error:\n{attempt.error}\nPlease provide a corrected version enclosed in ```python ... ```"}
            ])
        
        return conversation
