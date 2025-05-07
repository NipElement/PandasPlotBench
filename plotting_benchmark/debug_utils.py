import pandas as pd
import uuid
import tempfile
import traceback
import nbformat
from pathlib import Path
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

def get_debug_filename(base_dir: str | Path, prefix: str, model_name: str, plotting_lib: str, data_descriptor: str, suffix: str = "jsonl") -> Path:
    """
    统一构造 debug 输出文件路径

    Args:
        base_dir: 输出根目录
        prefix: 文件前缀（如 "debug_trials", "debug_result"）
        model_name: 模型名称路径，将 / 替换为 __
        plotting_lib: 使用的绘图库（matplotlib / seaborn / plotly）
        data_descriptor: 数据描述（如 "head", "describe"）
        suffix: 文件后缀（默认 jsonl）

    Returns:
        Path 对象：完整的输出文件路径
    """
    model_tag = model_name.replace("/", "__").replace("\\", "__")
    file_name = f"{prefix}_{model_tag}_{plotting_lib}_{data_descriptor}.{suffix}"
    return Path(base_dir) / file_name

def collect_failed_cells(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["error"] != "") | (~df["has_plot"])].copy()

def generate_debug_prompts(failed_df: pd.DataFrame) -> list[dict]:
    prompts = []
    for idx, row in failed_df.iterrows():
        prompt = (
            f"You are given a visualization task and a piece of Python code that failed to run.\n"
            f"---\nTask Description:\n{row['task__data_description']}\n"
            f"---\nOriginal Code:\n```python\n{row['code']}\n```\n"
            f"---\nError Message:\n{row['error']}\n"
            f"Please provide a corrected version of the code. Only return valid Python code enclosed in triple backticks like this: ```python ... ```."
        )
        prompts.append({"index": idx, "prompt": prompt})
    return prompts

def run_debug_attempts(model, prompts: list[dict], top_k: int = 3, output_dir: Path = None) -> pd.DataFrame:
    if output_dir is None:
        output_dir = Path(tempfile.gettempdir())
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for item in prompts:
        idx = item["index"]
        prompt = item["prompt"]

        for attempt in range(top_k):
            response = model.make_request(task=prompt, system_prompt="You are a code debugging assistant.")
            fixed_code = extract_code_from_response(response["response"])
            error_msg = ""
            has_plot = False
            success = False

            # notebook path
            nb_path = output_dir / f"debug_attempt_{idx}_{attempt}.ipynb"

            # Create notebook
            nb = nbformat.v4.new_notebook()
            nb["cells"] = [nbformat.v4.new_code_cell(f"# id = {idx}\n%matplotlib inline\n{fixed_code}")]

            try:
                ep = ExecutePreprocessor(timeout=10, interrupt_on_timeout=True, allow_errors=True, kernel_name="python3")
                ep.preprocess(nb, {"metadata": {"path": str(output_dir)}})
                for cell in nb["cells"]:
                    for out in cell.get("outputs", []):
                        if out.output_type == "error":
                            error_msg = out.ename + ": " + out.evalue
                    has_plot = any(
                        out.output_type == "display_data" and "image/png" in out.data
                        for out in cell.get("outputs", [])
                    )
                success = error_msg == ""
            except Exception as e:
                error_msg = f"ExecutionError: {str(e)}\n{traceback.format_exc()}"

            # Save notebook
            with open(nb_path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)

            # Append result
            all_results.append({
                "id": idx,
                "attempt": attempt,
                "fixed_code": fixed_code,
                "debug_success": success,
                "has_plot": has_plot,
                "error_message": error_msg,
                "notebook_path": str(nb_path),
            })

            if success:
                break

    return pd.DataFrame(all_results)

def extract_code_from_response(response: str) -> str:
    # Naive extraction assuming code in ```python ... ```
    import re
    match = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    return match.group(1).strip() if match else response.strip()

def check_code_validity(code: str) -> bool:
    try:
        compile(code, "<string>", "exec")
        return True
    except Exception:
        return False
