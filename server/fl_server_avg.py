import json

from server.fl_common import get_default_config, run_federated_experiment


def run_fedavg(dataset_type: str, partition_type: str, config=None):
    return run_federated_experiment(
        algorithm="fedavg",
        dataset_type=dataset_type,
        partition_type=partition_type,
        config=config or get_default_config(),
    )


if __name__ == "__main__":
    cfg = get_default_config()
    summary = []

    for dataset in cfg.datasets:
        for partition in cfg.partitions:
            result = run_fedavg(dataset, partition, cfg)
            summary.append(result)
            print(
                f"[FedAvg] dataset={dataset} partition={partition} "
                f"final_global_accuracy={result['final_global_accuracy']:.4f}"
            )

    print(json.dumps(summary, indent=2))
