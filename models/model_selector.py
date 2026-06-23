from models.cnn_model import (
    CNNModel
)


AVAILABLE_MODELS = {

    "cnn": CNNModel

}


def get_model(
        model_name: str,
        num_features: int
):
    """
    Model factory.

    Parameters
    ----------
    model_name : str
        Model architecture name.

    num_features : int
        Number of input features.

    Returns
    -------
    nn.Module
        Instantiated model.
    """

    model_name = (
        model_name
        .lower()
        .strip()
    )

    if model_name not in AVAILABLE_MODELS:

        raise ValueError(

            f"Unknown model '{model_name}'.\n"
            f"Available models: "
            f"{list(AVAILABLE_MODELS.keys())}"

        )

    return AVAILABLE_MODELS[
        model_name
    ](

        num_features=num_features

    )