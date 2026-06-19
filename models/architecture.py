from abc import ABC, abstractmethod


class BaseModel(ABC):

    @abstractmethod
    def train_model(
        self,
        train_loader,
        val_loader=None
    ):
        pass

    @abstractmethod
    def evaluate(
        self,
        data_loader
    ):
        pass

    @abstractmethod
    def predict(
        self,
        data_loader
    ):
        pass

    @abstractmethod
    def save(
        self,
        path
    ):
        pass

    @abstractmethod
    def load(
        self,
        path
    ):
        pass