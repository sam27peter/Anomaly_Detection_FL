import subprocess
import sys

from config.federated_config import DATASETS

PIPELINE = [

    "train.preprocessor_v2",
    "client.partitioner",

]


def run_module(
        module_name: str
):

    print("\n" + "=" * 60)

    print(f"Running {module_name}")

    print("=" * 60)

    subprocess.run(

        [
            sys.executable,
            "-m",
            module_name
        ],

        check=True
    )


if __name__ == "__main__":

    for module in PIPELINE:

        run_module(module)

    print("\nPipeline Completed Successfully")