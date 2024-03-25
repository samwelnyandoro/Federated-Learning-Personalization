from typing import Any, Dict

import flwr as fl

from fedpfl.federated_learning.constants import Algorithms
from fedpfl.federated_learning.utils import load_config
from fedpfl.federated_learning.example.model_cnn import DEVICE, CNNModelSplit, CNNNet
from fedpfl.federated_learning.utils import get_server_strategy

if __name__ == '__main__':

    config = load_config("../../data/config.json")

    algorithm = config.get("algorithm", Algorithms.FEDAVG.value)

    # Get strategy
    strategy = get_server_strategy(algorithm=algorithm)

    def create_model(config: Dict[str, Any]) -> CNNNet:
        """Create initial CNN model."""
        return CNNNet().to(DEVICE)

    strategy = strategy(
        config=config,
        algorithm=algorithm,
        create_model=create_model,
        model_split_class=CNNModelSplit,
        min_fit_clients=2,
        min_eval_clients=2,
        min_available_clients=2
    )

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config={"num_rounds": config.get("num_rounds", 3)},
        strategy=strategy
    )
