from models.cnn_model import CNNModel


def get_model(
    model_name,
    num_features
):

    model_name = model_name.lower()

    if model_name == "cnn":

        return CNNModel(
            num_features=num_features
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )