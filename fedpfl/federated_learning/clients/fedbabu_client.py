import copy
from collections import OrderedDict
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from flwr.common import Scalar

from fedpfl.federated_learning.constants import DEFAULT_FT_EP, DEFAULT_TRAIN_EP, Algorithms
from fedpfl.federated_learning.clients.base_client import BaseClient


class FedBABUClient(BaseClient):
    """Implementation of Federated Averaging with Body Aggregation and Body Update (FedBABU) Client."""

    def __init__(self, *args, **kwargs):
        super().__init__(has_fixed_head=True, *args, **kwargs)

    def get_parameters(self) -> List[np.ndarray]:
        """Return the current local body parameters."""
        return [val.cpu().numpy() for _, val in self.model_manager.model.body.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set the local body and fixed head parameters to the received parameters.
        In the first train round the body parameters are also set to the global body parameters,
        to ensure every client body is initialized equally.

        Args:
            parameters: parameters to set the model to.
        """
        model_keys = [k for k in self.model_manager.model.state_dict().keys()
                      if k.startswith("_body") or k.startswith("_fixed_head")]

        if self.train_id == 1:
            # Only update client's local head if it hasn't trained yet
            model_keys.extend([k for k in self.model_manager.model.state_dict().keys() if k.startswith("_head")])
            parameters.extend([
                parameters[k] for k in range(-len(self.model_manager.model.head.state_dict().keys()), 0)
            ])

        params_dict = zip(model_keys, parameters)

        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        self.model_manager.model.set_parameters(state_dict)

    def perform_train(self, tag: str = None) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """
        Perform local training to the body using the fixed head.

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
        epochs = self.config.get("epochs", {"body": DEFAULT_TRAIN_EP})
        self.model_manager.model.use_fixed_head(True)
        self.model_manager.model.enable_body()
        self.model_manager.model.disable_fixed_head()
        return self.model_manager.train(
            train_id=self.train_id,
            epochs=epochs.get("body", DEFAULT_TRAIN_EP),
            tag=f"{Algorithms.FEDBABU.value}_body" if tag is None else tag
        )

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
        Evaluate the provided global parameters using the locally held dataset. Performs fine-tuning before evaluation.

        Args:
            parameters: The current (global) model parameters.
            config: configuration parameters for training sent by the server.

        Returns:
        Tuple containing the test loss, \
                the number of examples used for evaluation and \
                the evaluation metrics.
        """
        self.set_parameters(parameters)

        # Set the head as the fixed_head and set to use the head for fine-tuning
        fixed_head_state_dict = copy.deepcopy(self.model_manager.model.fixed_head.state_dict())
        self.model_manager.model.head = fixed_head_state_dict
        self.model_manager.model.use_fixed_head(False)

        # Save the model parameters before fine-tuning
        model_state_dict = copy.deepcopy(self.model_manager.model.state_dict())

        self.model_manager.model.enable_body()
        self.model_manager.model.enable_head()

        # Fine-tune the model before testing
        epochs = self.config.get("epochs", {"fine-tuning": DEFAULT_FT_EP})
        ft_trn_results = self.model_manager.train(
            train_id=self.test_id,
            epochs=epochs.get("fine-tuning", DEFAULT_FT_EP),
            fine_tuning=True,
            tag=f"{self.config.get('algorithm', Algorithms.FEDBABU.value)}_full"
        )

        tst_results = self.model_manager.test(test_id=self.test_id)

        self.hist[str(self.test_id)] = {**self.hist[str(self.test_id)], "tst": tst_results}

        # Set the model parameters as they were before fine-tuning
        self.model_manager.model.set_parameters(model_state_dict)

        # Update the history with the ft_trn results
        self.hist[str(self.test_id)]["trn"] = {**(self.hist[str(self.test_id)].get("trn", {})), **ft_trn_results}

        self.test_id += 1

        return tst_results.get('loss', 0.0),\
            self.model_manager.test_dataset_size(),\
            {k: v for k, v in tst_results.items() if not isinstance(v, (dict, list))}
