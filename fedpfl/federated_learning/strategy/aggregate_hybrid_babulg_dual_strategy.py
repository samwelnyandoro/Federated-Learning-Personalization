from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from flwr.common import (
    EvaluateIns,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate

from fedpfl.federated_learning.strategy.initialization_strategy import (
    ServerInitializationStrategy,
)


class AggregateHybridBABULGStrategy(ServerInitializationStrategy):
    """FedHybridBABULGDual Aggregation strategy implementation."""

    def __init__(self, save_path: Path = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(has_fixed_head=True, *args, **kwargs)
        self.save_path = save_path
        if save_path is not None:
            self.save_path = save_path / "models"
            self.save_path.mkdir(parents=True, exist_ok=True)

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configure the next round of training. Adds the fixed head to the global parameters.

        Args:
            rnd: The current round of federated learning.
            parameters: The current (global) model parameters.
            client_manager: The client manager which holds all currently connected clients.

        Returns:
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `FitIns` for this particular `ClientProxy`. If a particular `ClientProxy`
            is not included in this list, it means that this `ClientProxy`
            will not participate in the next round of federated learning.
        """
        # Same as superclass method but adds the fixed head in the FedBABU algorithm

        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(rnd)

        agg_weights = parameters_to_weights(parameters=parameters)

        # Consider only the BABU body parameters
        weights = agg_weights[:len(self.model.body.state_dict().keys())]

        # Add BABU Fixed Head
        weights.extend([val.cpu().numpy() for _, val in self.model.fixed_head.state_dict().items()])

        # Add LG head from parameters
        weights.extend(agg_weights[-len(self.model.head.state_dict().keys()):])

        parameters = weights_to_parameters(weights=weights)

        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """
        Configure the next round of evaluation.

        Args:
            rnd: The current round of federated learning.
            parameters: The current (global) model parameters.
            client_manager: The client manager which holds all currently
                connected clients.

        Returns:
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `EvaluateIns` for this particular `ClientProxy`. If a particular
            `ClientProxy` is not included in this list, it means that this
            `ClientProxy` will not participate in the next round of federated
            evaluation.
        """
        # Same as superclass method but adds the fixed head in the FedBABU algorithm

        # Do not configure federated evaluation if a centralized evaluation
        # function is provided
        if self.eval_fn is not None:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(rnd)

        agg_weights = parameters_to_weights(parameters=parameters)

        # Consider only the BABU body parameters
        weights = agg_weights[:len(self.model.body.state_dict().keys())]

        # Add BABU Fixed Head
        weights.extend([val.cpu().numpy() for _, val in self.model.fixed_head.state_dict().items()])

        # Add LG head from parameters
        weights.extend(agg_weights[-len(self.model.head.state_dict().keys()):])

        parameters = weights_to_parameters(weights=weights)

        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        if rnd >= 0:
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )
        else:
            clients = list(client_manager.all().values())

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate the received local parameters, set the global model parameters and save the global model.

        Args:
            rnd: The current round of federated learning.
            results: Successful updates from the previously selected and configured
                clients. Each pair of `(ClientProxy, FitRes)` constitutes a
                successful update from one of the previously selected clients. Not
                that not all previously selected clients are necessarily included in
                this list: a client might drop out and not submit a result. For each
                client that did not submit an update, there should be an `Exception`
                in `failures`.
            failures: Exceptions that occurred while the server was waiting for client
                updates.
        Returns:
            If parameters are returned, then the server will treat these as the
            new global model parameters (i.e., it will replace the previous
            parameters with the ones returned from this method). If `None` is
            returned (e.g., because there were only failures and no viable
            results) then the server will no update the previous model
            parameters, the updates received in this round are discarded, and
            the global model parameters remain the same.
        """
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        body_weights_results = []
        head_weights_results = []

        for _, fit_res in results:

            weights = parameters_to_weights(fit_res.parameters)
            # Calculate body results
            body_weights_results.append(
                ([weights[k] for k in range(len(self.model.body.state_dict().keys()))], fit_res.num_examples)
            )

            if len(weights) > len(self.model.body.state_dict().keys()):
                # Calculate head results (only bigger clients send the head)
                head_weights_results.append(
                    ([weights[k] for k in range(-len(self.model.head.state_dict().keys()), 0)], fit_res.num_examples)
                )

        if head_weights_results == []:
            # In case no big clients were selected add the previous head
            agg_weights = aggregate(body_weights_results) + [val.cpu().numpy() for _, val in self.model.head.state_dict().items()]
        else:
            agg_weights = aggregate(body_weights_results) + aggregate(head_weights_results)


        # Update Server Model
        parameters = agg_weights
        model_keys = [k for k in self.model.state_dict().keys() if k.startswith("_body") or k.startswith("_head")]
        params_dict = zip(model_keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.set_parameters(state_dict)

        if self.save_path is not None:
            # Save Model
            torch.save(self.model, self.save_path / f"model-ep_{rnd}.pt")

        return weights_to_parameters(agg_weights), {}
