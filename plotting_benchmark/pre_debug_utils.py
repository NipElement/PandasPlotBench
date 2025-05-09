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



    def pre_run_self_debug(self, dataset_df: pd.DataFrame, model_name: str):
        """Run self debug mode while maintaining original eval structure"""
        print(f"[DEBUG] Running self debug mode for {model_name}")
        failed_df = collect_failed_cells(dataset_df)
        print(f"[DEBUG] Found {len(failed_df)} failed cases")
        
        debug_conversations = generate_self_debug_conversation(failed_df)
        
        debug_info = {}
        all_messages = []
        id_to_attempts = {}
        
        for item_id, conversation in debug_conversations:
            original_row = failed_df[failed_df['id'] == int(item_id)].iloc[0]
            debug_info[item_id] = {
                "original_error": original_row["error"],
                "original_has_plot": original_row["has_plot"],
                "debug_conversation": conversation,
                "attempts": {}
            }
            
            for attempt in range(self.config.debug.top_k):
                all_messages.append(conversation)
                id_to_attempts[len(all_messages)-1] = (item_id, str(attempt))
        
        if not hasattr(self, 'model_plot'):
            print("[DEBUG] Model not initialized. Initializing now...")
            self.init_gen_model(model_name)
            
        if all_messages:
            responses = self.model_plot.make_debug_request(all_messages)
            
            for i, response in enumerate(responses["response"]):
                item_id, attempt = id_to_attempts[i]
                debug_info[item_id]["attempts"][attempt] = {
                    "model_response": response,
                    "error": "",
                    "has_plot": False,
                    "plots_generated": []  # 初始化plots_generated字段
                }
        
        debug_rows = []
        for item_id, info in debug_info.items():
            original_row = failed_df[failed_df['id'] == int(item_id)].iloc[0]
            for attempt, attempt_info in info["attempts"].items():
                debug_row = original_row.copy()
                debug_row["code"] = CodePlotGenerator.gather_code(attempt_info["model_response"])
                debug_rows.append(debug_row)
        
        if debug_rows:
            debug_df = pd.DataFrame(debug_rows)
            debug_df = self.plot_generator.draw_debug_plots(debug_df)
            
            for idx, row in debug_df.iterrows():
                item_id = str(row['id'])
                attempt = str(idx % self.config.debug.top_k)
                debug_info[item_id]["attempts"][attempt].update({
                    "error": row["error"],
                    "has_plot": row["has_plot"],
                    "plots_generated": row.get("plots_generated", [])  # 保存每次尝试生成的图片
                })
        
        dataset_df["debug_info"] = dataset_df["id"].apply(
            lambda x: debug_info.get(str(x), None)
        )
        
        self.dump_results(dataset_df)
        
        error_rate_record_file = self.error_rate_file
        if error_rate_record_file.exists():
            with open(error_rate_record_file, "r") as f:
                error_rates = json.load(f)
        else:
            error_rates = {}
        
        # 统计debug修复情况
        total_debug_cases = len(debug_info)
        attempt_stats = {}
        
        # 初始化每个attempt的统计
        for k in range(self.config.debug.top_k):
            attempt_stats[k] = {
                "total_num": total_debug_cases,
                "execution_error_num": 0,
                "execution_error_rate": 0,
                "incorrect_plot_num": 0,
                "incorrect_plot_rate": 0
            }
        
        # 统计每个attempt的情况
        for item_id, info in debug_info.items():
            for attempt_idx in range(self.config.debug.top_k):
                attempt = str(attempt_idx)
                if attempt not in info["attempts"]:
                    continue
                
                # 统计执行错误
                if info["attempts"][attempt]["error"] != "":
                    attempt_stats[attempt_idx]["execution_error_num"] += 1
                
                # 统计图像生成情况
                if not info["attempts"][attempt]["has_plot"]:
                    attempt_stats[attempt_idx]["incorrect_plot_num"] += 1
        
        # 计算每个attempt的错误率
        for k in range(self.config.debug.top_k):
            stats = attempt_stats[k]
            total = stats["total_num"]
            if total > 0:
                stats["execution_error_rate"] = round(stats["execution_error_num"] / total, 4)
                stats["incorrect_plot_rate"] = round(stats["incorrect_plot_num"] / total, 4)
        
        # 更新error_rates记录
        record_key = f"{model_name}_{self.config.plotting_lib.split(' ')[0]}"
        if record_key in error_rates:
            error_rates[record_key].update({
                "debug_total_cases": int(total_debug_cases),
                "debug_attempts": {
                    f"attempt_{k}": {
                        "total_num": int(stats["total_num"]),
                        "execution_error_num": int(stats["execution_error_num"]),
                        "execution_error_rate": float(stats["execution_error_rate"]),
                        "incorrect_plot_num": int(stats["incorrect_plot_num"]),
                        "incorrect_plot_rate": float(stats["incorrect_plot_rate"])
                    }
                    for k, stats in attempt_stats.items()
                }
            })
            
            with open(error_rate_record_file, "w") as f:
                json.dump(error_rates, f, indent=4)
        
        return dataset_df, {}
