from collections import OrderedDict
from typing import Dict, List, Union

import numpy as np
import torch

from fedpfl.federated_learning.constants import DEFAULT_TRAIN_EP, Algorithms
from fedpfl.federated_learning.clients.base_client import BaseClient


class FedRepClient(BaseClient):
    """Implementation of Federated Representation Learning (FedRep) Client."""

    def get_parameters(self) -> List[np.ndarray]:
        """Return the current local head parameters."""
        return [val.cpu().numpy() for _, val in self.model_manager.model.body.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set the local body parameters to the received parameters.
        In the first train round the head parameters are also set to the global head parameters,
        to ensure every client head is initialized equally.

        Args:
            parameters: parameters to set the body to.
        """
        model_keys = [k for k in self.model_manager.model.state_dict().keys() if k.startswith("_body")]

        if self.train_id == 1:
            # Only update client's local head if it hasn't trained yet
            model_keys.extend([k for k in self.model_manager.model.state_dict().keys() if k.startswith("_head")])

        params_dict = zip(model_keys, parameters)

        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        self.model_manager.model.set_parameters(state_dict)

    def perform_train(self, tag: str = None) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """
        Perform local training to the head using the global body and then to the body using the previously trained head.

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
        epochs = self.config.get("epochs", {"body": DEFAULT_TRAIN_EP, "head": DEFAULT_TRAIN_EP})
        self.model_manager.model.enable_head()
        self.model_manager.model.disable_body()
        train_results = self.model_manager.train(
            train_id=self.train_id,
            epochs=epochs.get("head", DEFAULT_TRAIN_EP),
            tag=f"{Algorithms.FEDREP.value}_head" if tag is None else tag
        )
        self.model_manager.model.enable_body()
        self.model_manager.model.disable_head()
        train_results.update(
            self.model_manager.train(
                train_id=self.train_id,
                epochs=epochs.get("body", DEFAULT_TRAIN_EP),
                tag=f"{Algorithms.FEDREP.value}_body" if tag is None else tag
            )
        )
        return train_results
