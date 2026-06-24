import numpy as np


def compute_fairness(
        client_accuracies: list
) -> dict:
    """
    Compute federated fairness metrics.
    """

    client_accuracies = np.array(
        client_accuracies
    )

    return {

        "mean_accuracy":

            float(
                np.mean(
                    client_accuracies
                )
            ),

        "std_accuracy":

            float(
                np.std(
                    client_accuracies
                )
            ),

        "best_client_accuracy":

            float(
                np.max(
                    client_accuracies
                )
            ),

        "worst_client_accuracy":

            float(
                np.min(
                    client_accuracies
                )
            ),

        "accuracy_gap":

            float(

                np.max(
                    client_accuracies
                )

                -

                np.min(
                    client_accuracies
                )

            )

    }