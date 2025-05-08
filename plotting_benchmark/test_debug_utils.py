from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd
import json
from omegaconf import DictConfig
import os

from plotting_benchmark.benchmark import PlottingBenchmark
from plotting_benchmark.debug_utils import collect_failed_cells
from plotting_benchmark.code_plot_generator import CodePlotGenerator

@dataclass
class DebugAttempt:
    """单次调试尝试的信息"""
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
            # 第一次尝试，使用基础对话
            return self._generate_initial_conversation(row)
        else:
            # 后续尝试，包含历史信息
            return self._generate_follow_up_conversation(row, previous_attempts)
    
    @staticmethod
    def _generate_initial_conversation(row: pd.Series) -> List[Dict]:
        """生成初始调试对话"""
        task = row.get('task', '')  # 获取task，如果不存在则使用空字符串
        if not task:
            # 尝试从其他可能的字段构建task
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
        # 获取task的方式与_generate_initial_conversation相同
        task = row.get('task', '')
        if not task:
            task_fields = [f for f in row.index if f.startswith('task__')]
            task = '\n'.join(str(row[f]) for f in task_fields if pd.notna(row[f]))
            
        conversation = [{"content": task}]
        
        # 添加之前所有尝试的历史
        for attempt in previous_attempts:
            conversation.extend([
                {"content": f"```python\n{attempt.code}\n```", "is_assistant": True},
                {"content": f"The above code still failed with the following error:\n{attempt.error}\nPlease provide a corrected version enclosed in ```python ... ```"}
            ])
        
        return conversation

def execute_debug_attempt(benchmark: PlottingBenchmark,
                         conversations: List[Tuple[str, List[Dict]]],
                         notebook_path: Path) -> pd.DataFrame:
    """执行单次调试尝试"""
    # 准备模型请求
    all_messages = [conv for _, conv in conversations]
    id_to_conv = {item_id: conv for item_id, conv in conversations}
    
    # 发送请求获取响应
    responses = benchmark.model_plot.make_debug_request(all_messages)
    
    # 处理响应
    debug_rows = []
    for i, (item_id, _) in enumerate(conversations):
        response = responses["response"][i]
        
        # 创建新的数据行
        debug_row = {
            "id": int(item_id),
            "code": CodePlotGenerator.gather_code(response),
            "model_response": response,
            "error": "",
            "has_plot": False,
            "plots_generated": []
        }
        debug_rows.append(debug_row)
    
    # 创建DataFrame并执行代码生成图片
    debug_df = pd.DataFrame(debug_rows)
    debug_df = benchmark.plot_generator.draw_debug_plots(debug_df)
    
    return debug_df

def run_debug_session(benchmark: PlottingBenchmark,
                     dataset_df: pd.DataFrame,
                     model_name: str,
                     max_attempts: int = 1) -> Tuple[pd.DataFrame, Dict]:
    """运行调试会话的主函数"""
    
    debug_session = DebugSession(
        model_name=model_name,
        output_dir=benchmark.config.debug.output_dir,
        max_attempts=max_attempts
    )
    
    # 对于每次尝试
    for attempt_id in range(max_attempts):
        print(f"[DEBUG] Starting attempt {attempt_id + 1}/{max_attempts}")
        
        # 尝试加载之前的结果
        previous_results = debug_session.load_previous_attempt(attempt_id - 1) if attempt_id > 0 else None
        
        if attempt_id == 0 or previous_results is None:
            # 第一次尝试或无法加载之前的结果
            current_failed_df = collect_failed_cells(dataset_df)
        else:
            # 基于之前的结果继续调试
            current_failed_df = collect_failed_cells(previous_results)
        
        if len(current_failed_df) == 0:
            print(f"[DEBUG] No failed cases to fix in attempt {attempt_id + 1}")
            break
            
        print(f"[DEBUG] Found {len(current_failed_df)} cases to fix in attempt {attempt_id + 1}")
        
        # 生成调试对话
        debug_conversations = []
        for _, row in current_failed_df.iterrows():
            item_id = str(row['id'])
            previous_attempts = debug_session.debug_info.get(item_id, [])
            conversation = debug_session.generate_debug_conversation(
                row, attempt_id, previous_attempts
            )
            debug_conversations.append((item_id, conversation))
        
        # 执行调试
        debug_df = execute_debug_attempt(
            benchmark=benchmark,
            conversations=debug_conversations,
            notebook_path=debug_session.get_attempt_notebook_path(attempt_id)
        )
        
        # 保存结果
        debug_df.to_json(debug_session.get_attempt_result_path(attempt_id))
        
        # 更新调试信息
        for _, row in debug_df.iterrows():
            item_id = str(row['id'])
            attempt = DebugAttempt(
                attempt_id=attempt_id,
                code=row['code'],
                error=row['error'],
                has_plot=row['has_plot'],
                plots_generated=row.get('plots_generated', []),
                model_response=row['model_response'],
                conversation=next(conv for id_, conv in debug_conversations if id_ == item_id)
            )
            if item_id not in debug_session.debug_info:
                debug_session.debug_info[item_id] = []
            debug_session.debug_info[item_id].append(attempt)
    
    # 更新最终结果
    dataset_df['debug_info'] = dataset_df['id'].apply(
        lambda x: debug_session.debug_info.get(str(x))
    )
    
    return dataset_df, {}

def test_debug_session():
    """测试函数"""
    # 加载配置
    config_path = "configs/test.yaml"  # 使用测试配置
    benchmark = PlottingBenchmark(config_path=config_path)
    
    # 运行一个小规模的测试
    test_ids = [8, 14]  # 只测试前三个样例
    dataset_df = benchmark.dataset.select(test_ids).to_pandas()
    
    # 运行调试会话
    result_df, _ = run_debug_session(
        benchmark=benchmark,
        dataset_df=dataset_df,
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        max_attempts=2  # 测试两次尝试
    )
    
    # 打印结果
    print("\nDebug Session Results:")
    for idx, row in result_df.iterrows():
        if row.get('debug_info'):
            print(f"\nCase {row['id']}:")
            for attempt in row['debug_info']:
                print(f"  Attempt {attempt.attempt_id}:")
                print(f"    Error: {attempt.error}")
                print(f"    Has Plot: {attempt.has_plot}")

if __name__ == "__main__":
    test_debug_session()
