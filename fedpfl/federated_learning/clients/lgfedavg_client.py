from collections import OrderedDict
from typing import Dict, List, Union

import numpy as np
import torch

from fedpfl.federated_learning.constants import Algorithms
from fedpfl.federated_learning.clients.base_client import BaseClient


class LGFedAvgClient(BaseClient):
    """Implementation of Local Global Federated Averaging (LG-FedAvg) Client."""

    def get_parameters(self) -> List[np.ndarray]:
        """Return the current local body parameters."""
        return [val.cpu().numpy() for _, val in self.model_manager.model.head.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set the local head parameters to the received parameters.
        In the first train round the body parameters are also set to the global body parameters,
        to ensure every client body is initialized equally.

        Args:
            parameters: parameters to set the body to.
        """
        model_keys = [k for k in self.model_manager.model.state_dict().keys() if k.startswith("_head")]

        if self.train_id == 1:
            # Only update client's local body if it hasn't trained yet
            model_keys = [k for k in self.model_manager.model.state_dict().keys() if k.startswith("_body")]\
                 + model_keys
        else:
            # only consider the head parameters from all the received parameters
            parameters = [parameters[k] for k in range(-len(model_keys), 0)]

        params_dict = zip(model_keys, parameters)

        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        self.model_manager.model.set_parameters(state_dict)

    def perform_train(self, tag: str = None) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """
        Perform local training to the whole model.

        Args:
            tag: str of the form <Algorithm>_<model_train_part>.
                <Algorithm> - indicates the federated algorithm that is being performed\
                              (FedAvg, FedPer, FedRep, FedBABU or FedHybridAvgLGDual).
                              In the case of FedHybridAvgLGDual the tag also includes which part of the algorithm\
                                is being performed, either FedHybridAvgLGDual_FedAvg or FedHybridAvgLGDual_LG-FedAvg.
                <model_train_part> - indicates the part of the model that is being trained (full, body, head).
                This tag can be ignored if no difference in train behaviour is desired between federated algortihms.
        Returns:
            Dict with the train metrics.
        """
        return super().perform_train(tag=f"{Algorithms.LG_FEDAVG.value}_full" if tag is None else tag)
