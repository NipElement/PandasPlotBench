import time

import fire

from plotting_benchmark.benchmark import PlottingBenchmark
from plotting_benchmark.custom_task_changer import TaskShortner


def main(limit: int | list[int] | None = None):
    # You can limit ids like this:
    # 10, [0,1,2,3,4,5,6,7], [0,1], None, [3,4,8,14,21]

    # task_changer = TaskShortner(shorten_type="short")
    task_changer = None

    benchmark = PlottingBenchmark(
        config_path="configs/stat_debug_only.yaml", task_changer=task_changer
    )
    for i in range(1):
        benchmark.run_benchmark(
            limit, score_debug_only=True, only_stats=True
        )
        time.sleep(1)

if __name__ == "__main__":
    fire.Fire(main)
