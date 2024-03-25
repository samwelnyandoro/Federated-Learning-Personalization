import flwr as fl

from fedpfl.federated_learning.example.model_cnn import CNNModelManager
from fedpfl.federated_learning.utils import get_client_cls, load_config

if __name__ == '__main__':

    config = load_config("../../data/config.json")

    client_cls = get_client_cls(algorithm=config.get("algorithm", "FedAvg"))

    client = client_cls(
        config=config,
        client_id=0,
        model_manager_class=CNNModelManager
    )

    fl.client.start_numpy_client("127.0.0.1:8080", client=client)
