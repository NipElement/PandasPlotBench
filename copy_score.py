import json
import pandas as pd

def copy_scores():
    # 文件路径
    normal_file = "/data/yuansheng/PandasPlotBench/eval_results/api_model/plotly/results_gpt-4o-mini_plotly_head_0.json"
    debug_file = "/data/yuansheng/PandasPlotBench/eval_results/api_model_self_debug/plotly/results_gpt-4o-mini_plotly_head_0.json"
    
    # 读取文件
    normal_df = pd.read_json(normal_file)
    debug_df = pd.read_json(debug_file)
    
    # 需要复制的score相关列
    score_columns = ['score_vis', 'score_task', 'scoring_response', 'wrong_libs']
    
    # 复制分数信息
    for col in score_columns:
        if col in normal_df.columns:
            debug_df[col] = normal_df[col]
    
    # 保存更新后的文件
    debug_df.to_json(debug_file)
    print(f"已成功将评分信息从 {normal_file} 复制到 {debug_file}")

if __name__ == "__main__":
    copy_scores()