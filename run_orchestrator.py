import csv
import json
import os

import matplotlib.pyplot as plt

from server.fl_common import get_default_config
from server.fl_server_avg import run_fedavg
from server.fl_server_prox import run_fedprox


def _save_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)


def _plot_final_accuracy_all(rows, out_path):
    labels = [f"{r['algorithm']}-{r['dataset']}-{r['partition']}" for r in rows]
    values = [r["final_global_accuracy"] for r in rows]

    plt.figure(figsize=(12, 5))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Final Global Accuracy")
    plt.title("System Convergence Benchmark Comparison")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_iid_vs_noniid(rows, algorithm, dataset, out_path):
    scoped = [r for r in rows if r["algorithm"] == algorithm and r["dataset"] == dataset]
    scoped = sorted(scoped, key=lambda x: x["partition"])

    partitions = [r["partition"] for r in scoped]
    values = [r["final_global_accuracy"] for r in scoped]

    plt.figure(figsize=(6, 4))
    plt.bar(partitions, values)
    plt.ylim(0, 1)
    plt.ylabel("Final Global Accuracy")
    plt.xlabel("Partition Type")
    plt.title(f"{algorithm.upper()} IID vs Non-IID (Dataset {dataset})")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_all_experiments():
    cfg = get_default_config()

    comparisons_dir = "results/federated/comparisons"
    os.makedirs(comparisons_dir, exist_ok=True)

    run_manifest = []
    final_accuracy_rows = []

    for algorithm in ["fedavg", "fedprox"]:
        runner = run_fedavg if algorithm == "fedavg" else run_fedprox

        for dataset in cfg.datasets:
            for partition in cfg.partitions:
                status = "success"
                result = None
                error_message = ""

                try:
                    result = runner(dataset, partition, cfg)
                    final_accuracy_rows.append(
                        {
                            "algorithm": algorithm,
                            "dataset": dataset,
                            "partition": partition,
                            "final_global_accuracy": result["final_global_accuracy"],
                            "final_global_precision": result["final_global_precision"],
                            "final_global_recall": result["final_global_recall"],
                            "final_global_f1_score": result["final_global_f1_score"],
                            "output_root": result["output_root"],
                        }
                    )
                except Exception as exc:
                    status = "failed"
                    error_message = str(exc)

                run_manifest.append(
                    {
                        "algorithm": algorithm,
                        "dataset": dataset,
                        "partition": partition,
                        "status": status,
                        "output_root": result["output_root"] if result else "",
                        "error": error_message,
                    }
                )

    _save_csv(
        f"{comparisons_dir}/run_manifest.csv",
        run_manifest,
        ["algorithm", "dataset", "partition", "status", "output_root", "error"],
    )
    _save_json(f"{comparisons_dir}/run_manifest.json", {"runs": run_manifest})

    if final_accuracy_rows:
        _save_csv(
            f"{comparisons_dir}/final_global_accuracy.csv",
            final_accuracy_rows,
            [
                "algorithm",
                "dataset",
                "partition",
                "final_global_accuracy",
                "final_global_precision",
                "final_global_recall",
                "final_global_f1_score",
                "output_root",
            ],
        )

        _save_json(
            f"{comparisons_dir}/final_global_accuracy.json",
            {"rows": final_accuracy_rows},
        )

        _plot_final_accuracy_all(
            final_accuracy_rows,
            f"{comparisons_dir}/system_convergence_benchmark.png",
        )

        for algorithm in ["fedavg", "fedprox"]:
            for dataset in cfg.datasets:
                pair_rows = [
                    r for r in final_accuracy_rows
                    if r["algorithm"] == algorithm and r["dataset"] == dataset
                ]
                if len(pair_rows) == 2:
                    _save_csv(
                        f"{comparisons_dir}/{algorithm}_{dataset}_iid_vs_non_iid.csv",
                        pair_rows,
                        [
                            "algorithm",
                            "dataset",
                            "partition",
                            "final_global_accuracy",
                            "final_global_precision",
                            "final_global_recall",
                            "final_global_f1_score",
                            "output_root",
                        ],
                    )
                    _plot_iid_vs_noniid(
                        final_accuracy_rows,
                        algorithm,
                        dataset,
                        f"{comparisons_dir}/{algorithm}_{dataset}_iid_vs_non_iid.png",
                    )

    return {
        "manifest_path": f"{comparisons_dir}/run_manifest.csv",
        "final_accuracy_path": f"{comparisons_dir}/final_global_accuracy.csv",
    }


if __name__ == "__main__":
    output = run_all_experiments()
    print(json.dumps(output, indent=2))
