import copy
from collections import OrderedDict
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from flwr.common import Scalar

from fedpfl.federated_learning.constants import Algorithms
from fedpfl.federated_learning.clients.base_client import BaseClient
from fedpfl.federated_learning.clients.hybrid_client import HybridClient
from fedpfl.federated_learning.clients.lgfedavg_client import LGFedAvgClient


class HybridAvgLGDualClient(HybridClient, LGFedAvgClient):
    """Implementation of Federated Hybrid FedAvg LG-FedAvg Dual Model (FedHybridAvgLGDual) Client."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_body_parameters = None

    def get_parameters(self) -> List[np.ndarray]:
        """Return the current local model parameters. Smaller clients return their local model parameters, \
            larger clients return the trained server body parameters and the local head parameters trained \
            with the local body."""
        if self.smaller_client:
            return BaseClient.get_parameters(self)
        return [val.cpu().numpy() for _, val in self.server_body_parameters.items()] +\
            LGFedAvgClient.get_parameters(self)

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
            body_keys = [k for k in self.model_manager.model.body.state_dict().keys()]
            body_params_dict = zip(body_keys, parameters)
            self.server_body_parameters = OrderedDict({k: torch.tensor(v) for k, v in body_params_dict})

            LGFedAvgClient.set_parameters(self, parameters=parameters)

    def perform_train(self, tag: str = None) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """
        Perform local training to the whole model for smaller clients. \
            Larger clients train the global model and then train the local body with the global head.

        Args:
            tag: str of the form <Algorithm>_<model_train_part>.
                <Algorithm> - indicates the federated algorithm that is being performed\
                              (FedAvg, FedPer, FedRep, FedBABU or FedHybridAvgLGDual).
                              In the case of FedHybridAvgLGDual the tag also includes which part of the algorithm\
                                is being performed, either FedHybridAvgLGDual_FedAvg or FedHybridAvgLGDual_LG-FedAvg.
                <model_train_part> - indicates the part of the model that is being trained (full, body, head).
                This tag can be ignored if no difference in train behaviour is desired between federated algorithms.
        Returns:
            Dict with the train metrics.
        """
        if self.smaller_client:
            return BaseClient.perform_train(
                self,
                tag=f"{Algorithms.PROPOSAL_HYBRID_AVGLG_DUAL.value}_{Algorithms.FEDAVG.value}_full" if tag is None else tag
            )

        local_model = copy.deepcopy(self.model_manager.model.state_dict())

        self.model_manager.model.body = self.server_body_parameters

        trn_results = BaseClient.perform_train(
            self,
            tag=f"{Algorithms.PROPOSAL_HYBRID_AVGLG_DUAL.value}_{Algorithms.FEDAVG.value}_full" if tag is None else tag
        )

        self.server_body_parameters = copy.deepcopy(self.model_manager.model.body.state_dict())

        # Reset model to the original local model
        self.model_manager.model.set_parameters(local_model)
        trn_results.update(LGFedAvgClient.perform_train(
            self,
            tag=f"{Algorithms.PROPOSAL_HYBRID_AVGLG_DUAL.value}_{Algorithms.LG_FEDAVG.value}_full" if tag is None else tag
        ))

        return trn_results

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar]
    ) -> Union[
        Tuple[float, int, Dict[str, Scalar]],
        Tuple[int, float, float],
        Tuple[int, float, float, Dict[str, Scalar]],
    ]:
        """
        Evaluate the provided global parameters using the locally held dataset.

        Args:
            parameters: The current (global) model parameters.
            config: configuration parameters for training sent by the server.

        Returns:
        Tuple containing the test loss, \
                the number of examples used for evaluation and \
                the evaluation metrics.
        """
        if self.smaller_client:
            return BaseClient.evaluate(self, parameters=parameters, config=config)
        else:
            return LGFedAvgClient.evaluate(self, parameters=parameters, config=config)
