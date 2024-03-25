from fedpfl.federated_learning import constants
from fedpfl.federated_learning.clients.base_client import BaseClient


class HybridClient(BaseClient):
    """Base Client for hybrid algorithms (algorithms which combine two different algorithms)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.smaller_client = self._check_if_smaller_client(
            threshold=self.config.get("hybrid_threshold", constants.HYBRID_SMALL_CLIENT_THRESHOLD_DATA_POINTS)
        )

    def _check_if_smaller_client(self, threshold=constants.HYBRID_SMALL_CLIENT_THRESHOLD_DATA_POINTS) -> bool:
        """
        Check If the client is a smaller client. A client is considered small \
            if its data set size is equal to or smaller than a threshold (default of 2200 data points).
        Args:
            threshold: the upper bound to consider a client small (inclusive).
        """
        total_datapoints = self.model_manager.total_dataset_size()
        return total_datapoints <= threshold
