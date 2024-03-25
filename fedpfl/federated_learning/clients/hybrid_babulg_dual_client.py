import copy
from collections import OrderedDict
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from flwr.common import Scalar

from fedpfl.federated_learning.constants import Algorithms
from fedpfl.federated_learning.clients.fedbabu_client import FedBABUClient
from fedpfl.federated_learning.clients.hybrid_client import HybridClient
from fedpfl.federated_learning.clients.lgfedavg_client import LGFedAvgClient


class HybridBABULGDualClient(HybridClient, FedBABUClient, LGFedAvgClient):
    """Implementation of Federated Hybrid FedBABU LG-FedAvg Dual Model (FedHybridBABULGDual) Client."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.babu_body_parameters = None

    def get_parameters(self) -> List[np.ndarray]:
        """Return the current local model parameters. Smaller clients return their local body parameters, \
            larger clients return the trained server body parameters and the local head parameters trained \
            with the local body."""
        if self.smaller_client:
            return FedBABUClient.get_parameters(self)
        return [val.cpu().numpy() for _, val in self.babu_body_parameters.items()] + LGFedAvgClient.get_parameters(self)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set the local model parameters to the received parameters.
        Smaller clients set the local body. Larger clients set the local head.

        Args:
            parameters: parameters to set the model to.
        """
        if self.smaller_client:
            FedBABUClient.set_parameters(
                self,
                parameters=[
                    parameters[k] for k in range(
                        -len(self.model_manager.model.state_dict().keys()),
                        -len(self.model_manager.model.head.state_dict().keys())
                    )
                ]
            )
        else:
            body_keys = [k for k in self.model_manager.model.body.state_dict().keys()]
            body_params_dict = zip(body_keys, parameters)
            self.babu_body_parameters = OrderedDict({k: torch.tensor(v) for k, v in body_params_dict})

            if self.train_id == 1:
                # Set Fixed Head only the first time
                fixed_head_keys = [k for k in self.model_manager.model.fixed_head.state_dict().keys()]
                fixed_head_params = [
                    parameters[k] for k in range(
                        len(self.model_manager.model.body.state_dict().keys()),
                        (len(self.model_manager.model.body.state_dict().keys()) +
                         len(self.model_manager.model.fixed_head.state_dict().keys()))
                    )
                ]
                fixed_head_tuple = zip(fixed_head_keys, fixed_head_params)
                self.model_manager.model.fixed_head = OrderedDict({k: torch.tensor(v) for k, v in fixed_head_tuple})

            LGFedAvgClient.set_parameters(
                self,
                parameters=(
                    [parameters[k] for k in range(len(self.model_manager.model.body.state_dict().keys()))] +
                    [parameters[k] for k in range(-len(self.model_manager.model.head.state_dict().keys()), 0)]
                )
            )

    def perform_train(self, tag: str = None) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """
        Perform local training to the global body with the fixed head for smaller clients. \
            Larger clients train the global body with the fixed head and then train the local body with the global head.

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
            return FedBABUClient.perform_train(
                self,
                tag=f"{Algorithms.PROPOSAL_HYBRID_BABULG_DUAL.value}_{Algorithms.FEDBABU.value}_body" if tag is None else tag
            )
        self.model_manager.model.use_fixed_head(False)
        trn_results = LGFedAvgClient.perform_train(
            self,
            tag=f"{Algorithms.PROPOSAL_HYBRID_BABULG_DUAL.value}_{Algorithms.LG_FEDAVG.value}_full" if tag is None else tag
        )

        local_body = copy.deepcopy(self.model_manager.model.body.state_dict())

        # Update body with the BABU body parameters to perform BABU's training
        self.model_manager.model.body = self.babu_body_parameters
        trn_results.update(
            FedBABUClient.perform_train(
                self,
                tag=f"{Algorithms.PROPOSAL_HYBRID_BABULG_DUAL.value}_{Algorithms.FEDBABU.value}_body" if tag is None else tag
            )
        )

        self.babu_body_parameters = copy.deepcopy(self.model_manager.model.body.state_dict())

        # Reset body to the original body
        self.model_manager.model.body = local_body

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
        Evaluate the provided global parameters using the locally held dataset. \
            Smaller clients perform fine-tuning before evaluation, larger clients do not.

        Args:
            parameters: The current (global) model parameters.
            config: configuration parameters for training sent by the server.

        Returns:
        Tuple containing the test loss, \
                the number of examples used for evaluation and \
                the evaluation metrics.
        """
        if self.smaller_client:
            return FedBABUClient.evaluate(self, parameters=parameters, config=config)
        else:
            self.model_manager.model.use_fixed_head(False)
            return LGFedAvgClient.evaluate(self, parameters=parameters, config=config)
