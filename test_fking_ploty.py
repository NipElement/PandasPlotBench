import json
import pandas as pd
from pathlib import Path

def verify_results(result_file: str):
    """
    验证结果文件中的异常情况：error为空但has_plot为False的cases
    """
    print(f"\n检查文件: {result_file}")
    
    # 读取json文件
    with open(result_file, 'r') as f:
        df = pd.read_json(f)
    
    # 找出异常cases
    abnormal_cases = df[
        (df['error'] == "") & 
        (df['has_plot'] == False)
    ]
    
    if len(abnormal_cases) > 0:
        print(f"发现 {len(abnormal_cases)} 个异常cases:")
        for _, row in abnormal_cases.iterrows():
            print(f"  - Task {row['id']}: Empty error but has_plot=False")
            # 如果需要更详细的信息，可以打印其他字段
            if 'model_response' in row:
                print(f"    Model response length: {len(str(row['model_response']))}")
            if 'code' in row:
                print(f"    Code length: {len(str(row['code']))}")
    else:
        print("未发现异常cases")

def main():
    # 指定结果文件目录
    eval_dir = Path("eval_results/baseline_self_debug")
    
    # 遍历所有结果文件
    for lib_dir in ['matplotlib', 'plotly', 'seaborn']:
        lib_path = eval_dir / lib_dir
        if not lib_path.exists():
            continue
            
        for result_file in lib_path.glob("results_*.json"):
            verify_results(str(result_file))

if __name__ == "__main__":
    main()

# import re

# def gather_code(answer: str) -> str:
#     """
#     Gather all python code blocks in a response to a single code
#     """
#     if "```python" in answer:
#         first_block = answer.split("```python", 1)[1]
#         code = first_block.split("```", 1)[0].strip()
#     else:
#         for stop_token in ["<|eot_id|>", "<|endoftext|>", "<EOS>", "<|start_header_id|>", "<|end_header_id|>"]:
#             answer = answer.replace(stop_token, "")
#         code = answer.strip()

#     # Replace reading CSV with a placeholder
#     code = code.replace('df = pd.read_csv("data.csv")', "#")
#     code = code.replace("df = pd.read_csv('data.csv')", "#")
#     code = code.replace('df=pd.read_csv("data.csv")', "#")
#     code = code.replace("df=pd.read_csv('data.csv')", "#")

#     # Ensure numpy is imported if used
#     if "np." in code:
#         code = "import numpy as np\n" + code

#     # Use regex to replace .write_image(...) and .savefig(...) with .show()
#     code = re.sub(r'\.write_image\([^\)]*\)', '.show()', code)
#     code = re.sub(r'\.savefig\([^\)]*\)', '.show()', code)

#     # Replace try-except blocks containing .show() with just the .show() call
#     try_pattern = re.compile(r'try:\s*(.*?\.show\(\).*?)except.*?pass', re.DOTALL)
#     code = try_pattern.sub(r'\1', code)

#     return code

# # Test code
# code = """```python
# import pandas as pd
# import plotly.graph_objects as go

# # Load the dataframe
# #

# # Extract x, y, z values from the dataframe
# x = df['x'].values
# y = df['y'].values
# z = df['z'].values

# # Create a figure with a contour plot
# fig = go.Figure(data=go.Contour(
#     x=x, y=y, z=z,
#     contours=dict(start=0, end=1, size=0.1),
#     colorscale='Viridis',  # Corrected to a valid colorscale name
#     line_width=1,
#     showscale=False
# ))

# # Set the aspect ratio to be equal
# fig.update_layout(
#     title='Triangulation and Contour Plot',
#     xaxis=dict(title='X'),
#     yaxis=dict(title='Y'),
#     width=800,
#     height=800,
#     margin=dict(l=50, r=50, t=50, b=50),
#     plot_bgcolor='white'
# )

# # Show the plot
# try:
#     asdfasd.show()
# except KeyboardInterrupt:
#     pass
# ```<|eot_id|>"""

# print("处理后的代码：")
# print(gather_code(code))