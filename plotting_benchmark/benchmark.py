import gc
import json
import os
import shutil
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from plotting_benchmark.generation_engines.get_model import get_model_by_name

from .code_plot_generator import CodePlotGenerator
from .task_changer import TaskChanger
from .vis_generator import VisGenerator, add_index_to_filename
from .vis_judge import VisJudge
from .debug_utils import collect_failed_cells, generate_debug_prompts, run_debug_attempts, get_debug_filename

load_dotenv()


def get_config_template(config_folder: str | Path) -> None:
    config_folder = Path(config_folder)
    os.makedirs(config_folder, exist_ok=True)
    resource_folder = Path(__file__).parent.resolve() / "resources"
    config_file = resource_folder / "config_template.yaml"
    shutil.copyfile(config_file, config_folder / "config_template.yaml")


def get_instructs(instruct_folder: str | Path) -> None:
    os.makedirs(instruct_folder, exist_ok=True)
    instruct_folder = Path(instruct_folder)
    resource_folder = Path(__file__).parent.resolve() / "resources"
    config_file = resource_folder / "instructs.json"
    shutil.copyfile(config_file, instruct_folder / "instructs.json")


class PlottingBenchmark:
    def __init__(
        self,
        config_path: str | Path | None = None,
        config: DictConfig | None = None,
        task_changer: TaskChanger | None = None,
    ):
        self.resource_folder = Path(__file__).parent.resolve() / "resources"
        if task_changer is None:
            task_changer = TaskChanger()
        if config_path is not None:
            config = OmegaConf.load(config_path)
        elif config is None:
            raise ValueError("Provide either config or config path")
        self.config = config
        paths = self.config.paths
        self.error_rate_file = Path(paths.error_rate_file)
        benchmark_types = config.benchmark_types
        self.model_names = self.config.model_plot_gen.names

        out_folder = Path(paths.out_folder)
        # results filename is amended by model name
        out_folder.mkdir(exist_ok=True, parents=True)
        self.output_file = self.get_unique_filename(out_folder, "current_results.jsonl")
        self.bench_stat_file = out_folder / paths.bench_stat_filename

        if ("instructs_file" not in paths) or (paths.instructs_file is None):
            paths.instructs_file = self.resource_folder / "instructs.json"

        with open(paths.instructs_file, "r") as f:
            self.instructs = json.load(f)
        self.system_prompt = self.instructs["system_prompt"]
        self.dataset = load_dataset("JetBrains-Research/PandasPlotBench", split="test")
        self.model_judge = get_model_by_name(
            self.config.model_judge.name,
            dict(self.config.model_judge.parameters),
            self.system_prompt,
        )

        setup_instruct = self.instructs["setup_instruct"].replace(
            "[PLOTLIB]", self.config.plotting_lib
        )
        if ("matplotlib" not in self.config.plotting_lib) and (
            "seaborn" not in self.config.plotting_lib
        ):
            setup_instruct += f"DO NOT use matplotlib for plotting. Only {self.config.plotting_lib} library. "
        if "matplotlib" in self.config.plotting_lib:
            setup_instruct += "DO NOT use seaborn for plotting. Only pure matplotlib. "

        self.task_changer = task_changer
        self.task_changer.init_task_changer(
            data_instruct=self.instructs["data_instruct"],
            setup_instruct=setup_instruct,
            data_descriptor_name=self.config.data_descriptor,
        )

        dataset_folder = self.unpack_csv(paths.dataset_folder, self.dataset)

        self.plot_generator = VisGenerator(
            output_folder=out_folder,
            dataset=self.dataset,
            csv_folder=dataset_folder,
            config=self.config,
        )

        self.judge = VisJudge(
            vis_judge_model=self.model_judge,
            instructs=self.instructs,
            benchmark_types=benchmark_types,
            plot_lib=self.config.plotting_lib,
        )

        self.responses = None
        self.plot_responses = None

    def init_gen_model(self, model_name: str):
        # print plot gen parameters
        print(f"Plotting model parameters: {self.config.model_plot_gen.parameters}")
        self.model_plot = get_model_by_name(
            model_name,
            dict(self.config.model_plot_gen.parameters),
            self.system_prompt,
        )

        self.code_generator = CodePlotGenerator(
            model=self.model_plot,
            output_file=self.output_file,
            plotting_prompt=self.instructs["plot_instruct"],
            system_prompt=self.system_prompt,
        )

    @staticmethod
    def unpack_csv(csv_folder: str, dataset: Dataset) -> Path:
        csv_folder = Path(csv_folder)
        csv_folder.mkdir(parents=True, exist_ok=True)

        for item in dataset:
            data_csv_content = item["data_csv"]
            csv_path = csv_folder / f"data-{item['id']}.csv"

            if not csv_path.exists():
                with open(csv_path, "w") as file:
                    file.write(data_csv_content)
        print(f"CSV files unpacked to {csv_folder}")

        return csv_folder.resolve()

    @staticmethod
    def get_unique_filename(folder: Path, filename: str) -> Path:
        base, extension = os.path.splitext(filename)
        i = 1
        while (folder / filename).exists():
            filename = f"{base}_{i}{extension}"
            i += 1

        return folder / filename

    def kill_vllm(self):
        del self.model_plot.llm
        del self.model_plot
        gc.collect()
        # That import is here temporary to prevent import of cuda-libraries if they are not needed.
        import torch

        torch.cuda.empty_cache()
        print("Killed vLLM instance")

    def dump_results(self, dataset: pd.DataFrame) -> None:
        if os.path.exists(self.output_file):
            os.remove(self.output_file)
        dataset.to_json(self.results_file)
        print(f"results are dumped in {self.results_file}")

    def load_results(self, ids: list[int] | None = None) -> Dataset:
        dataset_df = pd.read_json(self.results_file)
        if isinstance(ids, list):
            dataset_df = dataset_df.loc[dataset_df["id"].isin(ids)]
        elif isinstance(ids, int):
            dataset_df = dataset_df.sample(n=ids, random_state=42)

        return dataset_df

    def run_self_debug(self, dataset_df: pd.DataFrame, model_name: str):
        plot_lib = self.config.plotting_lib.split(" ")[0]
        data_descriptor = self.config.data_descriptor
        output_dir = Path(self.config.debug.output_dir)

        # 1. collect failed items
        failed_df = collect_failed_cells(dataset_df)
        print(f"[DEBUG] Collected {len(failed_df)} failed samples for debugging.")

        # 2. build prompts
        prompts = generate_debug_prompts(failed_df)

        # 3. call model Top-K attempts
        debug_outputs = run_debug_attempts(
            self.model_plot,
            prompts,
            self.config.debug.top_k,
            output_dir=output_dir
        )

        # 4. save all attempts jsonl
        debug_trials_path = get_debug_filename(
            output_dir, "debug_trials", model_name, plot_lib, data_descriptor
        )
        debug_outputs.to_json(debug_trials_path, orient="records", lines=True, force_ascii=False)
        print(f"[DEBUG] Debug attempts (Top-K) saved to {debug_trials_path}")

        # 5. update working dataset code with first success
        for idx, row in debug_outputs.iterrows():
            if row["debug_success"]:
                dataset_df.loc[row["id"], "code"] = row["fixed_code"]

        # 6. build new debug notebook
        new_debug_nb_path = self.plot_generator.build_debug_plots(dataset_df)
        print(f"[DEBUG] Rebuilt debug notebook: {new_debug_nb_path}")

        # 7. re-execute and get result
        dataset_df = self.plot_generator.draw_plots(dataset_df)
        dataset_df["debug_success"] = dataset_df["error"] == ""

        # 8. save debugged final result jsonl
        debug_result_path = get_debug_filename(
            output_dir, "debug_result", model_name, plot_lib, data_descriptor
        )
        dataset_df.to_json(debug_result_path, orient="records", lines=True, force_ascii=False)
        print(f"[DEBUG] Final debug results saved to {debug_result_path}")

        return dataset_df

    def run_benchmark_model(
        self,
        model_name: str,
        ids: list[int] | int | None = None,
        reuse_results: bool = False,
        load_intermediate: bool = False,
        only_stats: bool = False,
        skip_plot: bool = False,
    ) -> tuple[pd.DataFrame, dict]:

        if self.config.get("run_mode", "normal") == "self_debug":
            plot_lib = (self.config.plotting_lib).split(" ")[0]
            gen_model_name = "_" + model_name.split("/")[-1]
            results_file_spostfix = (
                gen_model_name + "_" + plot_lib + "_" + self.config.data_descriptor
            )
            _, old_results_file = add_index_to_filename(
                self.config.paths.out_folder,
                self.config.paths.results_filename,
                results_file_spostfix,
            )
            
            if os.path.exists(old_results_file):
                dataset_df = self.load_results(ids)
                return self.run_self_debug(dataset_df, model_name=model_name)
            else:
                self.config.run_mode = "normal"
                dataset_df = self.run_benchmark_model(model_name, ids, reuse_results=False, 
                                                    load_intermediate=False, only_stats=False, 
                                                    skip_plot=False)
                self.config.run_mode = "self_debug"
                return self.run_self_debug(dataset_df, model_name=model_name)

        print(20 * "-")
        print(f"Benchmarking {model_name} model")
        print(20 * "-")
        gen_model_name = "_" + model_name.split("/")[-1]
        plot_lib = (self.config.plotting_lib).split(" ")[0]
        results_file_spostfix = (
            gen_model_name + "_" + plot_lib + "_" + self.config.data_descriptor
        )
        new_results_file, old_results_file = add_index_to_filename(
            self.config.paths.out_folder,
            self.config.paths.results_filename,
            results_file_spostfix,
        )
        print(f"[DEBUG]: Reuse: {reuse_results}, Load: {load_intermediate}, Only stats: {only_stats}")
        print(f"[DEBUG] New results file: {new_results_file}")
        print(f"[DEBUG] Old results file: {old_results_file}")
        if reuse_results or only_stats:
            self.results_file = old_results_file
            print(f"loading {self.results_file}")
            dataset_df = self.load_results(ids)
        else:
            self.init_gen_model(model_name)
            # Run the model
            self.results_file = new_results_file
            if isinstance(ids, list):
                self.dataset = self.dataset.select(ids)
            elif isinstance(ids, int):
                self.dataset = self.dataset.shuffle(seed=42).select(range(ids))
            dataset_df = self.dataset.to_pandas()
            dataset_df = self.task_changer.change_task(dataset_df)
            self.dataset = Dataset.from_pandas(dataset_df)
            dataset_df = self.code_generator.generate_codeplot_datapoints(
                self.dataset, load_intermediate
            )
            self.dump_results(dataset_df)
            # Kill the vllm model after running to
            if self.model_plot.__class__.__name__ == "VllmEngine":
                self.kill_vllm()

        # if not only_stats:
        #     dataset_df = self.plot_generator.draw_plots(dataset_df)
        #     dataset_df = self.judge.score(dataset_df)
        #     self.dump_results(dataset_df)
        if not only_stats:
            if not skip_plot:
                print("[DEBUG] Drawing plots...")
                dataset_df = self.plot_generator.draw_plots(dataset_df)
            else:
                print("[DEBUG] Skipping plot rendering.")
                model_name = dataset_df["model"].iloc[0].replace("/", "__")
                data_descriptor = dataset_df["data_descriptor"].iloc[0]
                plot_lib = self.config.plotting_lib.split(" ")[0]
                json_filename = Path(self.results_file).name
                notebook_index = json_filename.split("_")[-1].replace(".json", "")
                expected_nb_path = (
                    Path(self.config.paths.out_folder) /
                    f"plots_{data_descriptor}_{model_name}_{plot_lib}_{notebook_index}.ipynb"
                )

                if not expected_nb_path.exists():
                    raise FileNotFoundError(
                        f"Notebook {expected_nb_path} not found. Cannot extract plots!"
                    )

                print(f"[DEBUG] Loading plots from notebook {expected_nb_path}")
                parsed_df = self.plot_generator.parse_plots_notebook(expected_nb_path)

                # Remove any old columns from dataset that will be overwritten
                common_cols = dataset_df.columns.intersection(parsed_df.columns).drop("id")
                dataset_df = dataset_df.drop(columns=common_cols)
                dataset_df = dataset_df.merge(parsed_df, on="id", how="left")
            print("[DEBUG] Skip Score")
            total_items = len(dataset_df)

            # 统计 Execution Error Rate（Cell 有错误）
            execution_error_num = (dataset_df["error"] != "").sum()
            execution_error_rate = round(execution_error_num / total_items, 4)

            # 统计 Incorrect Code Rate（没有图像生成）
            incorrect_code_num = (dataset_df["has_plot"] == False).sum()
            incorrect_code_rate = round(incorrect_code_num / total_items, 4)

            print(f"[DEBUG] Execution error rate (cell error): {execution_error_rate:.4f}")
            print(f"[DEBUG] Incorrect code rate (no plot): {incorrect_code_rate:.4f}")

            error_rate_record_file = self.error_rate_file
            if error_rate_record_file.exists():
                with open(error_rate_record_file, "r") as f:
                    error_rates = json.load(f)
            else:
                error_rates = {}

            record_key = f"{model_name}_{plot_lib}"
            error_rates[record_key] = {
                "execution_error_rate": execution_error_rate,
                "incorrect_code_rate": incorrect_code_rate
            }

            with open(error_rate_record_file, "w") as f:
                json.dump(error_rates, f, indent=4)

            # exit(0)
            return dataset_df, {}

            dataset_df = self.judge.score(dataset_df)
            self.dump_results(dataset_df)
        bench_stats = self.judge.calculate_stats(dataset_df)
        with open(self.bench_stat_file, "a") as f:
            json.dump(bench_stats, f)
            f.write("\n")
        print(f"Benchmark stats saved in {self.bench_stat_file}")

        return dataset_df, bench_stats

    def run_benchmark(
        self,
        ids: list[int] | int | None = None,
        reuse_results: bool = False,
        load_intermediate: bool = False,
        only_stats: bool = False,
        skip_plot: bool = False,
    ) -> None:
        for model_name in self.model_names:
            self.run_benchmark_model(
                model_name,
                ids,
                reuse_results,
                load_intermediate,
                only_stats,
                skip_plot,
            )
