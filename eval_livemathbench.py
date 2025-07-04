# livemathbench_task.py
from lighteval.tasks.lighteval_task import LightevalTaskConfig

LiveMathBench = LightevalTaskConfig(
    name="livemathbench",
    dataset_path="path/to/LiveMathBench",
    metric=["accuracy"],
    prompt_function="livemathbench_prompt",
)