import torch
import torch.nn as nn

from models.architecture import BaseModel


class CNNModel(
    nn.Module,
    BaseModel
):

    def __init__(
        self,
        num_features
    ):

        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=32,
            kernel_size=5,
            padding=2
        )

        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm1d(64)

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool1d(2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(
            64,
            128
        )

        self.fc2 = nn.Linear(
            128,
            1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x
    ):

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

    def train_model(
        self,
        train_loader,
        val_loader=None
    ):
        pass

    def evaluate(
        self,
        data_loader
    ):
        pass

    def predict(
        self,
        data_loader
    ):
        pass

    def save(
        self,
        path
    ):
        torch.save(
            self.state_dict(),
            path
        )

    def load(
        self,
        path
    ):
        self.load_state_dict(
            torch.load(path)
        )