import pandas as pd
from typing import List
from pathlib import Path

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

