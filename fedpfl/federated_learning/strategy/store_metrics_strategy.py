from typing import Dict, List, Optional, Tuple

from flwr.common import EvaluateRes, FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy

from fedpfl.federated_learning.strategy.store_history_strategy import (
    StoreHistoryStrategy,
)


class StoreMetricsStrategy(StoreHistoryStrategy):
    """Server FL metrics storage per training/evaluation round strategy implementation."""

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate the received local parameters and store the train aggregated metrics.

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
        aggregates = super().aggregate_fit(rnd=rnd, results=results, failures=failures)

        self.hist["trn"][rnd] = {k.cid: {"num_examples": v.num_examples, **v.metrics} for k, v in results}

        return aggregates

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate the received local parameters and store the evaluation aggregated metrics.

        Args:
            rnd: The current round of federated learning.
            results: Successful updates from the
                previously selected and configured clients. Each pair of
                `(ClientProxy, FitRes` constitutes a successful update from one of the
                previously selected clients. Not that not all previously selected
                clients are necessarily included in this list: a client might drop out
                and not submit a result. For each client that did not submit an update,
                there should be an `Exception` in `failures`.
            failures: Exceptions that occurred while the server
                was waiting for client updates.
        Returns:
            Optional `float` representing the aggregated evaluation result. Aggregation
            typically uses some variant of a weighted average.
        """

        aggregates = super().aggregate_evaluate(rnd=rnd, results=results, failures=failures)

        self.hist["tst"][rnd] = {
            k.cid: {"num_examples": v.num_examples, "loss": v.loss, **v.metrics} for k, v in results
        }

        return aggregates
