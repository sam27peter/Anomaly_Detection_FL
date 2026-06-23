# Federated Learning Configuration

NUM_CLIENTS = 5

NUM_ROUNDS = 3

LOCAL_EPOCHS = 20

BATCH_SIZE = 64

LEARNING_RATE = 0.001

ALGORITHM = "fedavg"

PARTITION = "iid"

WINDOW_SIZE = 100
STRIDE = 10

DATASETS = [
    "SMAP",
    "MSL"
]