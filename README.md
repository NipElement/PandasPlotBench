# Plotting benchmark

This is the benchmark to assess the capability of models in writing the code for visualizations given the description of the Pandas DataFrame.

🛠️ **Task**. Given the plotting task and the description of a Pandas DataFrame, write the code to build a plot.

The dataset can be found on [HuggingFace page](https://huggingface.co/datasets/JetBrains-Research/plot_bench). It is based on the [MatPlotLib gallery](https://matplotlib.org/stable/gallery/index.html).

📩 If you have any questions or requests concerning this dataset, please contact the author at [timur.galimzyanov@jetbrains.com](mailto:timur.galimzyanov@jetbrains.com).

# Install

1. Clone repo `git clone https://github.com/JetBrains-Research/plotting-benchmark.git`
2. `cd plotting-benchmark`
3. Run `poetry install`
   1. If you're going to use benchmarking on the local machine (includes using `code_bert_score`), instead run `poetry install --extras "local_gpu"`
4. Edit config if needed (`configs/config.yaml`).
5. Setup environment variables for the proprietary model keys if necessary (see details in [Usage section](#usage)).

6. Run the benchmark (see details in [Usage section](#usage)):
`poetry run python run_benchmark.py`

You can run the benchmark on some subset of the datapoints passing the `--limit` parameter either the number of datapoints to run or list of IDs:

`poetry run python run_benchmark.py --limit=2`

# Dataset

Dataset contains plotting task, tiny data csv to be plotted and ground truth images. 
Each datapoint is stored in separate folder. The task is divided into 2 parts:
1. **Plot Description**. Main part, describing the target plot.
2. **Plot Style Description**. General guidelines for plot styling.

Tasks can be changed dynamically using `TaskChanger` class (see **Usage** section).

Dataset can be loaded from via [`load_dataset`](https://huggingface.co/docs/datasets/v3.1.0/en/package_reference/loading_methods#datasets.load_dataset):

```
from datasets import load_dataset
dataset = load_dataset("JetBrains-Research/plot_bench", split="test")
```

# Usage

For the code generation models you can use four options:

1. VLLM. Just pass HuggingFace model name in the model_plot_gen.names list.
2. OpenAI models. Add "openai/" prefix to the OpenAI model name to select this option. In this case you should set `OPENAI_KEY` environment variable with corresponding token. 
3. TogetherAI models. Add "together/" prefix to the TogetherAI model name to select this option. In this case you should set `TOGETHERAI_KEY` environment variable with corresponding token.

For image-based scoring we use OpenAI GPT4-v model (default is `gpt-4o-2024-05-13`). So, you have to set `OPENAI_KEY` environment variable with corresponding token.
You can provide keys in .env file in the root of the repo, that would be loaded automatically.

## Basic usage
```
from plotting_benchmark.benchmark import PlottingBenchmark

benchmark = PlottingBenchmark(config_path="configs/config.yaml")

benchmark.run_benchmark()
```

### Method's arguments:

- `ids` - Limits datapoints ids to be benchmarked: i.e `ids = [3, 5, 7]`
- `reuse_results` - if `True`, does not generate plots, reuses results saved in results_filename.
- `load_intermediate` - if `True`, does not generate plots, loads intermediate results from current_results.jsonl
that stores intermediate results for the case of crush.
- `only_stats` - if `True` does not run benchmarking just calculates stats from the file results_filename.

### Resources

Config template and LLM instructs can be found in `plotting_benchmark/resources` folder


## Results

Results are saved in the `out_folder` that is set up in config.
For each benchmarking model following files are saved:

- `results_{modelname}_{plottinglib}_{df_descriptor}.json` - dataset with results for each datapoint (plots in encoded png, scores, generated code)
- `all_plotsall_plots_{modelname}_{plottinglib}_{df_descriptor}.ipynb` - notebook with all plots (code, figures, possible errors) of the dataset.
- `benchmark_stat.jsonl` - statistics for the benchmark scores. Each model result starts from new line.
 

## Custom task changer

You can experiment with tasks wording. I.e. change data description or setup part to control plotting libraries.
To do that, create a `CustomTaskChanger` inherited from `TaskChanger`. This is a template for custom task changer:

```
import pandas as pd
from plotting_benchmark.benchmark import PlottingBenchmark
from plotting_benchmark.task_changer import TaskChanger

class MyTaskChanger(TaskChanger):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
   
    def setup_changer(self, task_text: str, df: pd.DataFrame) -> str:
        return "Use assembler language and [PLOLIB] library to draw a plot"

    def data_descr_changer(self, task_text: str, df: pd.DataFrame) -> str:
        return generate_custon_dataframe_description(task_text, df)
        
    def plot_descr_changer(self, task_text: str, df: pd.DataFrame) -> str:
        # Be carefull with that - it is the main task, describing the plot.
        return task_text

    def style_changer(self, task_text: str, df: pd.DataFrame) -> str:
        return "Draw a beautiful plot"

benchmark = PlottingBenchmark(
    config_path="configs/config.yaml", task_changer_class=MyTaskChanger
)

benchmark.run_benchmark()
```

# Important notes:

1. Our approach relies on the run of the LLM-generated code. Apart from the safety, there is an issue of installed libraries. Generated code could use uninstalled libs and this will lead to not plotted plot. May be in the prompt we should list installed graphical libraries.
2. time_used_per_item in stats includes waiting time in case of time-out.
