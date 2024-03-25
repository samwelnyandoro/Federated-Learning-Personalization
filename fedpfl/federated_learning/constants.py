from pathlib import Path
from enum import Enum

# FL Algorithms
class Algorithms(Enum):
    FEDAVG = "FedAvg"
    LG_FEDAVG = "LG-FedAvg"
    FEDPER = "FedPer"
    FEDREP = "FedRep"
    FEDBABU = "FedBABU"
    PROPOSAL_HYBRID_AVGLG = "FedHybridAvgLG"
    PROPOSAL_HYBRID_AVGLG_DUAL = "FedHybridAvgLGDual"
    PROPOSAL_HYBRID_BABULG_DUAL = "FedHybridBABULGDual"

# FL Default Train and Fine-Tuning Epochs
DEFAULT_TRAIN_EP = 5
DEFAULT_FT_EP = 5

# Hybrid Proposal smaller client data points count threshold
HYBRID_SMALL_CLIENT_THRESHOLD_DATA_POINTS = 2200

DEFAULT_DATA_PATH = Path('../data')
