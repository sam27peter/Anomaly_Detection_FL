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

        self.bn1 = nn.BatchNorm1d(
            32
        )

        self.pool1 = nn.MaxPool1d(
            2
        )

        self.conv2 = nn.Conv1d(
            32,
            64,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm1d(
            64
        )

        self.pool2 = nn.MaxPool1d(
            2
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            64 * 25,
            128
        )

        self.dropout = nn.Dropout(
            0.3
        )

        self.fc2 = nn.Linear(
            128,
            1
        )

        self.relu = nn.ReLU()

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

        x = self.pool1(
            self.relu(
                self.bn1(
                    self.conv1(x)
                )
            )
        )

        x = self.pool2(
            self.relu(
                self.bn2(
                    self.conv2(x)
                )
            )
        )

        x = self.flatten(x)

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