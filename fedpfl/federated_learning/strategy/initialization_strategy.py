from typing import Any, Callable, Dict, Optional, Type

import torch.nn as nn
from flwr.common import Parameters, weights_to_parameters
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.strategy import Strategy

from fedpfl.federated_learning.constants import Algorithms
from fedpfl.model.model_split import ModelSplit


class ServerInitializationStrategy(Strategy):
    """Server FL Parameter Initialization strategy implementation."""

    def __init__(
        self,
        model_split_class: Type[ModelSplit],
        create_model: Callable[[Dict[str, Any]], nn.Module],
        config: Dict[str, Any] = {},
        algorithm: str = Algorithms.FEDAVG.value,
        has_fixed_head: bool = False,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.algorithm = algorithm
        self.model = model_split_class(model=create_model(config), has_fixed_head=has_fixed_head)

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize the (global) model parameters.

        Args:
            client_manager: ClientManager. The client manager which holds all currently
                connected clients.

        Returns:
            If parameters are returned, then the server will treat these as the
            initial global model parameters.
        """
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        if initial_parameters is None and self.model is not None:
            if self.algorithm == Algorithms.LG_FEDAVG.value:
                initial_parameters = [val.cpu().numpy() for _, val in self.model.head.state_dict().items()]
            elif self.algorithm in [Algorithms.FEDPER.value, Algorithms.FEDREP.value, Algorithms.FEDBABU.value]:
                initial_parameters = [val.cpu().numpy() for _, val in self.model.body.state_dict().items()]
            elif self.algorithm in [Algorithms.PROPOSAL_HYBRID_BABULG_DUAL.value]:
                initial_parameters = [val.cpu().numpy() for _, val in self.model.body.state_dict().items()]
                initial_parameters += [val.cpu().numpy() for _, val in self.model.head.state_dict().items()]
            else:  # FedAvg, Proposal FedHybridAvgLG, Proposal FedHybridAvgLGDual
                initial_parameters = [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        if isinstance(initial_parameters, list):
            initial_parameters = weights_to_parameters(weights=initial_parameters)
        return initial_parameters
