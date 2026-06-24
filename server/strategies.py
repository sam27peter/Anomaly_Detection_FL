import copy
import torch


def federated_average(
        local_weights: list
) -> dict:
    """
    FedAvg aggregation.
    """

    avg_weights = copy.deepcopy(
        local_weights[0]
    )

    for key in avg_weights.keys():

        avg_weights[key] = torch.stack(
            [
                w[key].float()
                for w in local_weights
            ],
            dim=0
        ).mean(dim=0)

    return avg_weights


def get_prox_mu(
        algorithm: str
) -> float:
    """
    Return proximal coefficient.
    """

    if algorithm.lower() == "fedprox":
        return 0.01

    return 0.0