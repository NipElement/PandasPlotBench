import pandas as pd

def collect_failed_cells(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["error"] != "") | (~df["has_plot"])].copy()

def generate_self_debug_conversation(failed_df: pd.DataFrame) -> list[tuple[str, list[dict]]]:
    """Generate debug conversations for all failed cases
    
    Args:
        failed_df: DataFrame containing failed cases
        
    Returns:
        List of tuples (id, conversation), where conversation is a list of message dicts
    """
    conversations = []
    for idx, row in failed_df.iterrows():
        item_id = str(row['id'])
        task = row['task']
        original_code = row['code']
        error = row['error']
        
        conversation = [
            {"content": task},
            {"content": f"```python\n{original_code}\n```", "is_assistant": True},
            {"content": f"The above code failed with the following error:\n{error}\nPlease provide a corrected version enclosed in ```python ... ```"}
        ]
        conversations.append((item_id, conversation))
    
    return conversations


