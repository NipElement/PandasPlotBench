import time

import fire

from plotting_benchmark.benchmark import PlottingBenchmark
from plotting_benchmark.custom_task_changer import TaskShortner


def main(limit: int | list[int] | None = 10):
    # You can limit ids like this:
    # 10, [0,1,2,3,4,5,6,7], [0,1], None, [3,4,8,14,21]

    # task_changer = TaskShortner(shorten_type="short")
    task_changer = None

    benchmark = PlottingBenchmark(
        config_path="configs/self_debug_test.yaml", task_changer=task_changer
    )

    for i in range(1):
        benchmark.run_benchmark(
            limit, reuse_results=False, load_intermediate=False, only_stats=False, skip_plot=False
        )
        time.sleep(1)

    # run_benchmark's flags:
    # reuse_results - if True, does not generate plots, reuses results saved in results_filename.
    # load_intermediate - if True, does not generate plots, loads intermediate results from current_results.jsonl
    # that is stores intermediate results for the case of crush.
    # only_stats - if True does not run benchmarking just calculates stats from the file results_filename.
    
    # python run_benchmark.py --limit=2

if __name__ == "__main__":
    fire.Fire(main)
