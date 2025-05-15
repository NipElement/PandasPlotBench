import re
import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd
from tqdm import tqdm


class VisJudge:
    """
    Class for running visual benchmark over the plotted plots comparing with golden truth datapoints
    Visual benchmarking is asking model to compare two images and return a score
    """

    def __init__(
        self,
        vis_judge_model,
        instructs: dict,
        benchmark_types: list[str],
        plot_lib: str,
    ) -> None:
        self.vis_judge_model = vis_judge_model

        self.instructs = instructs
        if "system_prompt" in self.instructs:
            self.system_prompt = self.instructs["system_prompt"]
        else:
            print("No system prompt is given. One for basic model will be used:")
            print(vis_judge_model.system_prompt)
        self.eligible_bench_types = ["vis", "task"]
        self.bench_types = benchmark_types
        self.plot_lib = plot_lib
        if "codebert" in self.bench_types:
            # That import is here temporary to prevent import of cuda-libraries if they are not needed.
            from plotting_benchmark.code_bert_scorer import \
                calc_code_bert_score

    @staticmethod
    def gen_task_judge_request(base_instruct: str, item: NamedTuple) -> str:
        # For benchmarking we take original task and style descriptions
        if hasattr(item, "old_task__plot_description"):
            plot_descr = item.old_task__plot_description
        else:
            plot_descr = item.task__plot_description
        if hasattr(item, "old_task__plot_style"):
            plot_style = item.old_task__plot_style
        else:
            plot_style = item.task__plot_style
        instruct = [base_instruct, "[PLOT TASK]:", plot_descr, plot_style]

        return "\n".join(instruct)

    def score_by_type(self, dataset: pd.DataFrame, bench_type: str) -> pd.DataFrame:
        if bench_type not in self.eligible_bench_types:
            raise ValueError(f"Unknown benchmark type {bench_type}")

        instruct_name = f"judge_instruct_{bench_type}"
        if instruct_name not in self.instructs:
            raise ValueError(f"You should have {instruct_name} key in instructs")
        bench_instruct = self.instructs[instruct_name]

        if "plots_generated" not in dataset.columns:
            raise ValueError(
                "Dataset does not contain images, please generate them first"
            )

        print(f"{bench_type} benchmarking.")
        scoring_responses = []
        scores = []
        wrong_libs = []
        for item in tqdm(dataset.itertuples(), total=len(dataset)):
            gen_plots = item.plots_generated
            code = item.code
            wrong_lib = 0
            score_response = ""
            if self.plot_lib not in code:
                wrong_lib = 1
            if gen_plots is np.nan:
                score = None
            # If there is no target plotting lib, we score plot as 0
            elif (len(gen_plots) == 0) or (self.plot_lib not in code):
                score = 0
            else:
                if bench_type == "vis":
                    plots = [gen_plots[0], item.plots_gt[0]]
                elif bench_type == "task":
                    bench_instruct = self.gen_task_judge_request(bench_instruct, item)
                    plots = gen_plots
                response = self.vis_judge_model.make_request(
                    request=bench_instruct,
                    images=plots,
                    image_detail="auto",
                )

                if response is not None:
                    score_response = response["response"]
                    score = self.parse_bench_response(response["response"])

            scoring_responses.append(score_response)
            scores.append(score)
            wrong_libs.append(wrong_lib)

        dataset[f"score_{bench_type}"] = scores
        dataset["wrong_libs"] = wrong_libs
        dataset["scoring_response"] = scoring_responses

        return dataset

    def score(self, dataset: pd.DataFrame) -> pd.DataFrame:
        print("Scoring the plot results.")
        for bench_type in self.bench_types:
            if bench_type == "codebert":
                dataset = calc_code_bert_score(dataset)
            else:
                dataset = self.score_by_type(dataset, bench_type)

        return dataset

    @staticmethod
    def parse_bench_response(response: str) -> int | None:
        try:
            return int(response)

        except ValueError:
            # match = re.search(r"[FINAL SCORE]:? ?(\d+)", response)
            match = re.search(r".*\[FINAL SCORE]:? ?(\d+)", response, re.DOTALL)
            if match:
                return int(match.group(1))
            else:
                return None

    @staticmethod
    def calculate_stats_by_type(dataset_orig: pd.DataFrame, bench_type: str) -> dict:
        dataset = dataset_orig.copy()
        """
        Calculate statistics of the scores
        """
        score_name = f"score_{bench_type}"

        total_items = len(dataset)
        model_name = dataset["model"][0]
        if len(dataset["model"].unique()) > 1:
            warnings.warn(
                f"There are {len(dataset['model'].unique())} model names in results. Only first one would be used: {model_name}"
            )

        dataset[score_name] = dataset[score_name].fillna(0)
        scored_items = len(dataset)
        # number of unscored items - either LLM error or unformated response
        # score stats would be calculated among scored results
        num_unparsed = total_items - scored_items
        scores = dataset[score_name].to_numpy()
        scores_good = np.sum(scores >= 75) / scored_items
        scores_bad = np.sum(scores <= 25) / scored_items

        if bench_type in ["vis", "task"]:
            statistics = {
                "mean": int(np.mean(scores)),
                "median": int(np.median(scores)),
                "good": round(scores_good, 2),
                "bad": round(scores_bad, 2),
                "min": int(min(scores)),
                "max": int(max(scores)),
                "num_scored_items": int(scored_items),
                "unparsed": int(num_unparsed),
            }
        else:
            statistics = {
                "mean": round(np.mean(scores), 4),
                "median": round(np.median(scores), 4),
                "good": round(scores_good, 4),
                "bad": round(scores_bad, 4),
                "min": round(min(scores), 4),
                "max": round(max(scores), 4),
                "num_scored_items": int(scored_items),
                "unparsed": int(num_unparsed),
            }

        return statistics

    def calculate_stats(self, dataset: pd.DataFrame) -> dict:
        model_name = dataset["model"][0]
        data_descriptor = dataset["data_descriptor"][0]
        start_time = dataset["start_time"][0] if "start_time" in dataset else "UNK"
        if len(dataset["model"].unique()) > 1:
            warnings.warn(
                f"There are {len(dataset['model'].unique())} model names in results. Only first one would be used: {model_name}"
            )

        no_plots = (~dataset["has_plot"]).sum()
        err_num = (dataset["error"] != "").sum()
        wrong_libs = dataset["wrong_libs"].sum()
        dataset["response_length"] = dataset["raw_response"].apply(
            lambda x: len(x["response"])
        )
        dataset["task_length"] = dataset["task"].str.len()
        mean_task_length = dataset["task_length"].mean()
        mean_response_length = dataset["response_length"].mean()
        time_used_per_item = (
            round(dataset["time_used_gen"].mean(), 1)
            if "time_used_gen" in dataset
            else "UNK"
        )

        bench_types = [
            col.removeprefix("score_")
            for col in dataset.columns
            if col.startswith("score_")
        ]

        stats = dict()
        scored_items = 0
        for bench_type in bench_types:
            stat_type = self.calculate_stats_by_type(dataset, bench_type)
            stats[bench_type] = stat_type
            scored_items += stat_type["num_scored_items"]

        scored_items /= len(bench_types)
        statistics = {
            "model": model_name,
            "plotting_lib": self.plot_lib,
            "data_descriptor": data_descriptor,
            "no_plots": int(no_plots),
            "wrong_libs": int(wrong_libs),
            "error_number": int(err_num),
            "error_rate": round(err_num / scored_items, 3),  # among mean scored items
            "response_mean_length_symb": round(mean_response_length, 0),
            "task_mean_length_symb": round(mean_task_length, 0),
            "time_used_per_item": time_used_per_item,
            "start_time": start_time,
            "scores": stats,
            "instructs": self.instructs,
        }

        return statistics

    def score_debug(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        专门用于debug模式的评分函数
        """
        print("Scoring debug attempts...")
        
        # 收集需要评分的cases
        debug_cases = {}
        for idx, row in dataset.iterrows():
            if row.get('debug_info') is not None:
                debug_cases[idx] = row['debug_info']
        
        # 对每种评分类型进行评分
        for bench_type in self.bench_types:
            if bench_type == "codebert":
                continue  # 暂不支持codebert评分
            
            print(f"Scoring {bench_type} for debug attempts...")
            
            # 收集所有需要评分的attempts
            scoring_rows = []
            case_attempt_map = []  # 用于记录每行对应的case和attempt
            
            for case_id, debug_info in debug_cases.items():
                for attempt_id, attempt_info in debug_info['attempts'].items():
                    # 跳过无图片生成的attempts
                    if not attempt_info.get('has_plot') or not attempt_info.get('plots_generated'):
                        continue
                        
                    scoring_rows.append({
                        'plots_generated': attempt_info['plots_generated'],
                        'code': attempt_info['code'],
                        'plots_gt': dataset.loc[case_id, 'plots_gt'],
                        'task__plot_description': dataset.loc[case_id, 'task__plot_description'],
                        'task__plot_style': dataset.loc[case_id, 'task__plot_style']
                    })
                    case_attempt_map.append((case_id, attempt_id))
            
            if not scoring_rows:
                continue
            
            # 一次性评分所有attempts
            temp_df = pd.DataFrame(scoring_rows)
            scored_df = self.score_by_type(temp_df, bench_type)
            
            # 将评分结果存回debug_info
            for idx, (case_id, attempt_id) in enumerate(case_attempt_map):
                attempt_info = debug_cases[case_id]['attempts'][attempt_id]
                attempt_info[f'score_{bench_type}'] = scored_df[f'score_{bench_type}'].iloc[idx]
                attempt_info['scoring_response'] = scored_df['scoring_response'].iloc[idx]
                attempt_info['wrong_libs'] = scored_df['wrong_libs'].iloc[idx]
        
        return dataset

    def calculate_debug_stats(self, dataset: pd.DataFrame) -> dict:
        # 获取基本信息
        model_name = dataset["model"][0]
        data_descriptor = dataset["data_descriptor"][0]
        
        stats = {
            "model": model_name,
            "plotting_lib": self.plot_lib,  # 从self.plot_lib获取
            "data_descriptor": data_descriptor
        }
        
        # 1. 首先计算normal模式的统计（直接使用原始分数）
        normal_stats = {}
        normal_stats['vis'] = self.calculate_stats_by_type(dataset, 'vis')
        normal_stats['task'] = self.calculate_stats_by_type(dataset, 'task')
        stats['normal'] = normal_stats
        
        # 2. 获取最大attempt数
        max_attempts = max(
            max(int(attempt_id) 
                for attempt_id in debug_info['attempts'].keys())
            for debug_info in dataset['debug_info']
            if debug_info and 'attempts' in debug_info
        )
        
        # 3. 对每个attempt进行统计
        for attempt_id in range(max_attempts + 1):
            # 创建当前attempt的DataFrame
            current_rows = []
            
            for _, row in dataset.iterrows():
                case_id = row['id']
                debug_info = row.get('debug_info', {})
                
                if not debug_info or 'attempts' not in debug_info:
                    # 如果没有debug信息，使用原始分数
                    current_rows.append({
                        'id': case_id,
                        'model': model_name,
                        'score_vis': row.get('score_vis', 0),
                        'score_task': row.get('score_task', 0)
                    })
                    continue
                
                # 找到不超过当前attempt_id的最大attempt
                available_attempts = sorted([int(aid) for aid in debug_info['attempts'].keys()])
                target_attempt = None
                for aid in available_attempts:
                    if aid <= attempt_id:
                        target_attempt = str(aid)
                    else:
                        break
                
                if target_attempt is not None:
                    attempt_info = debug_info['attempts'][target_attempt]
                    current_rows.append({
                        'id': case_id,
                        'model': model_name,
                        'score_vis': attempt_info.get('score_vis', 0),
                        'score_task': attempt_info.get('score_task', 0)
                    })
                else:
                    current_rows.append({
                        'id': case_id,
                        'model': model_name,
                        'score_vis': row.get('score_vis', 0),
                        'score_task': row.get('score_task', 0)
                    })
            
            # 创建临时DataFrame并计算统计
            temp_df = pd.DataFrame(current_rows)
            attempt_stats = {}
            attempt_stats['vis'] = self.calculate_stats_by_type(temp_df, 'vis')
            attempt_stats['task'] = self.calculate_stats_by_type(temp_df, 'task')
            stats[f'attempt_{attempt_id}'] = attempt_stats
        
        return stats
