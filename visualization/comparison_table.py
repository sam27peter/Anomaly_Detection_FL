from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def generate_table():

    csv_path = (

        Path("results")
        / "comparison"
        / "experiment_summary.csv"

    )

    if not csv_path.exists():

        print(
            "Run comparison engine first."
        )

        return

    df = pd.read_csv(
        csv_path
    )

    plt.figure(
        figsize=(16, 8)
    )

    plt.axis("off")

    table = plt.table(

        cellText=df.round(4).values,

        colLabels=df.columns,

        loc="center"

    )

    table.auto_set_font_size(
        False
    )

    table.set_fontsize(8)

    table.scale(
        1.2,
        1.5
    )

    save_path = (

        Path("results")
        / "comparison"
        / "comparison_table.png"

    )

    plt.savefig(

        save_path,

        bbox_inches="tight"

    )

    plt.close()

    print(
        f"Saved {save_path}"
    )


if __name__ == "__main__":

    generate_table()