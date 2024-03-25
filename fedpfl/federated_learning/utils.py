import json
from pathlib import Path
from typing import Any, Dict, Type, Union

from flwr.server.strategy.strategy import Strategy

from fedpfl.federated_learning.constants import Algorithms
from fedpfl.federated_learning.clients.base_client import BaseClient
from fedpfl.federated_learning.clients.fedbabu_client import FedBABUClient
from fedpfl.federated_learning.clients.fedper_client import FedPerClient
from fedpfl.federated_learning.clients.fedrep_client import FedRepClient
from fedpfl.federated_learning.clients.hybrid_avglg_client import HybridAvgLGClient
from fedpfl.federated_learning.clients.hybrid_avglg_dual_client import (
    HybridAvgLGDualClient,
)
from fedpfl.federated_learning.clients.hybrid_babulg_dual_client import (
    HybridBABULGDualClient,
)
from fedpfl.federated_learning.clients.lgfedavg_client import LGFedAvgClient
from fedpfl.federated_learning.strategy.strategy_pipeline import (
    AggregateBodyStrategyPipeline,
    AggregateHeadStrategyPipeline,
    AggregateHybridBABULGStrategyPipeline,
    DefaultStrategyPipeline,
)


def load_config(config_file: Union[str, Path]) -> Dict[str, Any]:
    """Load the config file into a dictionary."""
    filepath = Path(config_file)

    f = open(filepath, mode='rt', encoding='utf-8')
    config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError("config must be a dict or the name of a file containing a dict.")

    return config

def get_client_cls(algorithm: str) -> Type[BaseClient]:
    """Get client class from algorithm (default is FedAvg)."""
    if algorithm == Algorithms.FEDPER.value:
        return FedPerClient
    elif algorithm == Algorithms.LG_FEDAVG.value:
        return LGFedAvgClient
    elif algorithm == Algorithms.FEDREP.value:
        return FedRepClient
    elif algorithm == Algorithms.FEDBABU.value:
        return FedBABUClient
    elif algorithm == Algorithms.PROPOSAL_HYBRID_AVGLG.value:
        return HybridAvgLGClient
    elif algorithm == Algorithms.PROPOSAL_HYBRID_AVGLG_DUAL.value:
        return HybridAvgLGDualClient
    elif algorithm == Algorithms.PROPOSAL_HYBRID_BABULG_DUAL.value:
        return HybridBABULGDualClient
    elif algorithm == Algorithms.FEDAVG.value:
        return BaseClient
    else:
        raise ValueError(f"No such algorithm: {algorithm}")

def get_server_strategy(algorithm: str) -> Strategy:
    """
    Gets the server strategy pipeline corresponding to the received algorithm.

    Args:
        algorithm: the federated algorithm to be performed.

    Returns:
        The pipeline to be used.
    """
    if algorithm in [Algorithms.FEDPER.value, Algorithms.FEDREP.value, Algorithms.FEDBABU.value]:
        return AggregateBodyStrategyPipeline
    elif algorithm in [Algorithms.LG_FEDAVG.value]:
        return AggregateHeadStrategyPipeline
    elif algorithm in [Algorithms.PROPOSAL_HYBRID_BABULG_DUAL.value]:
        return AggregateHybridBABULGStrategyPipeline
    elif algorithm in [Algorithms.FEDAVG.value, Algorithms.PROPOSAL_HYBRID_AVGLG.value, Algorithms.PROPOSAL_HYBRID_AVGLG_DUAL.value]:  # FedAvg, Proposal FedHybridAvgLG and Proposal FedHybridAvgLGDual
        return DefaultStrategyPipeline
    else:
        raise ValueError(f"No such algorithm: {algorithm}")
