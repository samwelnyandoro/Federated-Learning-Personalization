from typing import Dict, List, Union

import numpy as np

from fedpfl.federated_learning.constants import Algorithms
from fedpfl.federated_learning.clients.base_client import BaseClient
from fedpfl.federated_learning.clients.hybrid_client import HybridClient
from fedpfl.federated_learning.clients.lgfedavg_client import LGFedAvgClient


class HybridAvgLGClient(HybridClient, LGFedAvgClient):
    """Implementation of Federated Hybrid FedAvg LG-FedAvg (FedHybridAvgLG) Client."""

    def get_parameters(self) -> List[np.ndarray]:
        """Return the current local model parameters."""
        return BaseClient.get_parameters(self)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set the local model parameters to the received parameters.
        Smaller clients set the whole local model. Larger clients only set the local head.

        Args:
            parameters: parameters to set the model to.
        """
        if self.smaller_client:
            BaseClient.set_parameters(self, parameters=parameters)
        else:
            LGFedAvgClient.set_parameters(self, parameters=parameters)

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
        return super().perform_train(tag=f"{Algorithms.PROPOSAL_HYBRID_AVGLG.value}_full" if tag is None else tag)
