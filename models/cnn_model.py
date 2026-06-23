import torch
import torch.nn as nn

from pathlib import Path

from models.architecture import BaseModel


class CNNModel(
    nn.Module,
    BaseModel
):
    """
    1D CNN model for multivariate time-series
    anomaly detection.
    """

    def __init__(
            self,
            num_features: int
    ) -> None:
        """
        Parameters
        ----------
        num_features : int
            Number of input sensor features.
        """

        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=32,
            kernel_size=5,
            padding=2
        )

        self.bn1 = nn.BatchNorm1d(
            32
        )

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm1d(
            64
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool1d(
            kernel_size=2
        )

        self.global_pool = (
            nn.AdaptiveAvgPool1d(1)
        )

        self.dropout = nn.Dropout(
            p=0.3
        )

        self.fc1 = nn.Linear(
            64,
            128
        )

        self.fc2 = nn.Linear(
            128,
            1
        )

        self.sigmoid = nn.Sigmoid()

    # ==================================================
    # FORWARD PASS
    # ==================================================

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape:
            (batch, sequence_length, features)

        Returns
        -------
        torch.Tensor
            Predicted probabilities.
        """

        x = x.permute(
            0,
            2,
            1
        )

        x = self.pool(
            self.relu(
                self.bn1(
                    self.conv1(x)
                )
            )
        )

        x = self.pool(
            self.relu(
                self.bn2(
                    self.conv2(x)
                )
            )
        )

        x = self.global_pool(x)

        x = x.squeeze(-1)

        x = self.relu(
            self.fc1(x)
        )

        x = self.dropout(x)

        x = self.sigmoid(
            self.fc2(x)
        )

        return x

    # ==================================================
    # PLACEHOLDER METHODS
    # ==================================================

    def train_model(
            self,
            train_loader,
            val_loader=None
    ):
        """
        Reserved for future standalone training.
        """
        raise NotImplementedError(
            "Training handled externally."
        )

    def evaluate(
            self,
            data_loader
    ):
        """
        Reserved for future evaluation API.
        """
        raise NotImplementedError(
            "Evaluation handled externally."
        )

    def predict(
            self,
            data_loader
    ):
        """
        Reserved for future prediction API.
        """
        raise NotImplementedError(
            "Prediction handled externally."
        )

    # ==================================================
    # SAVE / LOAD
    # ==================================================

    def save(
            self,
            path: str | Path
    ) -> None:
        """
        Save model weights.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        torch.save(
            self.state_dict(),
            path
        )

    def load(
            self,
            path: str | Path,
            device: str = "cpu"
    ) -> None:
        """
        Load model weights.

        Parameters
        ----------
        path : str | Path
            Model file path.

        device : str
            Device for loading.
        """

        self.load_state_dict(
            torch.load(
                path,
                map_location=device
            )
        )