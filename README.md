# Personalized Federated Learning

This project contains the implementation of some of the Federated Learning personalization algorithms based on the parameter decoupling technique for the [Flower](https://flower.dev/docs/quickstart-pytorch.html) Framework using [PyTorch](https://github.com/pytorch/pytorch). Currently the code is compatible with the "0.17.0" version of Flower.

**The implemented algorithms are:**
- [Federated Averaging (FedAvg)](http://proceedings.mlr.press/v54/mcmahan17a?ref=https://githubhelp.com) -> the default Federated Learning algorithm
- [Federated Learning with Personalization Layers (FedPer)](https://arxiv.org/abs/1912.00818)
- [Local Global Federated Averaging (LG-FedAvg)](https://arxiv.org/abs/2001.01523)
- [Federated Representation Learning (FedRep)](https://proceedings.mlr.press/v139/collins21a.html)
- [Federated Averaging with Body Aggregation and Body Update (FedBABU)](https://arxiv.org/abs/2106.06042)



I also provide the implementation of some experimental algorithms which attempt to combine some of the previous algorithms, such that, the algorithm used by each client depends on the amount of data of the client, we deem these hybrid algorithms. Clients with data equal or below the defined threshold are considered small clients and clients above the threshold are considered large clients. 

**The implemented hybrid algorithms are:**
- Federated Hybrid FedAvg LG-FedAvg (FedHybridAvgLG) - executes the FedAvg algorithm for smaller clients and the LG-FedAvg algorithm for larger clients.
- Federated Hybrid FedBABU LG-FedAvg Dual Model (FedHybridBABULGDUAL) - executes the FedBABU algorithm for smaller clients and the LG-FedAvg algorithm for larger clients, but requires the larger clients to calculate two models.
- Federated Hybrid FedAvg LG-FedAvg Dual Model (FedHybridAvgLGDUAL) - executes the FedAvg algorithm for smaller clients and the LG-FedAvg algorithm for larger clients, but requires the larger clients to calculate two models.

# Implementation

The implementation is model-agnostic, such that, it is possible to run these algorithms with any model. In order to do that, concrete implementations for two abstract classes are needed: **ModelManager** and; **ModelSplit**.

## Model Manager

The first class needs to extend from the [ModelManager](fedpfl/model/model_manager.py) class and implement its abstract methods. The purpose of this class is to manage a model, that is, create the model and perform the training/testing of the model.

Firstly the method `_create_model` needs to be implemented. This method simply creates the base model.

```Python
@abstractmethod
def _create_model(self) -> nn.Module:
```

The second method to be implemented is `train`. This method trains the model and returns the training metrics.
```Python
@abstractmethod
def train(
    self,
    train_id: int,
    epochs: int = 1,
    tag: Optional[str] = None,
    fine_tuning: bool = False
) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
```

The third method to be implemented is `test`. This method tests the model and returns the test metrics.
```Python
@abstractmethod
def test(self, test_id: int) -> Dict[str, float]:
```

Lastly, three methods to get the size of the client data sets need to be implemented:
- `train_dataset_size` -> returns the size of the train data set.
```Python
@abstractmethod
def train_dataset_size(self) -> int:
```

- `test_dataset_size` -> returns the size of the test data set.
```Python
@abstractmethod
def train_dataset_size(self) -> int:
```

- `total_dataset_size` -> returns the total number of data points of the clients (all data sets combined, validation and any other data sets included).
```Python
@abstractmethod
def total_dataset_size(self) -> int:
```

## Model Split

The second class needs to extend from the [ModelSplit](/fedpfl/model/model_split.py) class and implement its abstract methods. The purpose of this class is to divide the model into two parts the body and the head.

This class needs the method `_get_model_parts` to be implemented. This method receives the model to be split and returns a tuple containing the body of the model in the first position and the head in the second position.

```Python
@abstractmethod
def _get_model_parts(self, model: nn.Module) -> Tuple[nn.Module, nn.Module]:
```

## Client

### Configurations

The client configurations can be defined in the [config file](/fedpfl/data/config.json) in json format, the available configurations are:
- `algorithm` -  the algorithm to use, one of: `"FedAvg"`, `"LG-FedAvg"`, `"FedPer"`, `"FedRep"`, `"FedBABU"`, `"FedHybridAvgLG"`, `"FedHybridBABULGDual"` or `"FedHybridAvgLGDual"`.
- `hybrid_threshold` - the threshold for the hybrid algorithms (default is 2200).
- `epochs` - number of training epochs for each model part:
    - `full`  - number of epochs for the full model (used  in `"FedAvg"`, `"LG-FedAvg"`, `"FedPer"`, `"FedHybridAvgLG"`, `"FedHybridBABULGDual"` or `"FedHybridAvgLGDual"`).
    - `body` number of training epochs for the body of the model (used in `"FedRep"` and `"FedBABU"`).
    - `head` number of training epochs for the head of the model (used in `"FedRep"`).
    - `fine-tuning` number of training epochs for the fine-tuning of the model (used in all models).
- `fine-tuning` - whether fine-tuning should be performed to the full model before evaluation (`True`) or not (`False`) (`"FedBABU"` always performs fine-tuning before evaluation).


Example file:

```JSON
{
    "algorithm": "FedAvg",
    "hybrid_threshold": 2200,
    "epochs": {
        "full": 1,
        "body": 1,
        "head": 1,
        "fine-tuning": 1
    },
    "fine-tuning": false
}
```
### Client File

After the two classes **ModelManager** and **ModelSplit** are implemented, the client class can be created:

```Python
# Load the configuration file using load_config from fedpfl.federated_learning.utils
config = load_config("../../data/config.json")

# Get the client from the algorithm using get_client_cls from fedpfl.federated_learning.utils
client_cls = get_client_cls(algorithm=config.get("algorithm", "FedAvg"))

# Create the client and pass it the implemented model manager class
client = client_cls(
    config=config,
    client_id=0,
    model_manager_class=<ImplementedModelManagerClass>
)
```

## Server Initialization

The client configuration file can be reused for the server. Configurations available:
- `algorithm` -  the algorithm to use, one of: `"FedAvg"`, `"LG-FedAvg"`, `"FedPer"`, `"FedRep"`, `"FedBABU"`, `"FedHybridAvgLG"`, `"FedHybridBABULGDual"` or `"FedHybridAvgLGDual"`.
- `num_rounds` - number of communication rounds to be performed by the server.

Example file:
```JSON
{
    "algorithm": "FedAvg",
    "num_rounds": 3
}
```

### Server File

By default, Flower uses the model parameters from one client to initialize the global model. However, these personalized algorithms require the clients to start from the same model, thus server model initialization is required. Therefore, when instantiating the strategy used by the server a function to create the model needs to be passed.

```Python
# Load the configuration file using load_config from fedpfl.federated_learning.utils
config = load_config("../../data/config.json")

# Get the server strategy from the algorithm using get_server_strategy from fedpfl.federated_learning.utils
strategy = get_server_strategy(algorithm=algorithm)

# Define the function that creates the model for server initialization
def create_model(config: Dict[str, Any]) -> <ModelClass>:
    """Create initial CNN model."""
    return <ModelClass>().to(DEVICE)

# Instantiate the strategy and pass it the implemented model split class
strategy = strategy(
    config=config,
    algorithm=algorithm,
    create_model=create_model,
    model_split_class=<ImplementedModelSplitClass>,
    min_fit_clients=2,
    min_eval_clients=2,
    min_available_clients=2
)

# Start server with 3 communication rounds
fl.server.start_server(
    server_address="127.0.0.1:8080",
    config={"num_rounds": config.get("num_rounds", 3)},
    strategy=strategy
)
```

## Example

I provide an [example](/fedpfl/federated_learning/example) which was adapted from ["Flower Example using PyTorch"](https://github.com/adap/flower/tree/2728140a28514b3174fafee0c64d368e8b568645/examples/quickstart_pytorch).

### Initial Setup

Start by downloading the source files with:

```bash
git clone https://github.com/samwelnyandoro/FederatedLearning-Personalization.git
```

This will create a folder `fl_personalization` with the source code. Then change the directory to the `fl_personalization` with:

```bash
cd fl_personalization
```

To run the example first you need to setup the virtual environment. For this we use [Poetry](https://python-poetry.org/docs/). The requirements for the project are present in `pyproject.toml`.

Then, install the dependencies of the project with:

```bash
poetry install
```

Open a virtual shell with:

```bash
poetry shell
```

### Running the Server

To run the server change the directory to:

```bash
cd fedpfl/federated_learning/example
```

Then launch the server with:

```bash
python flower_server_cnn.py
```

### Running the Clients

After starting the server, the clients can be started. In a new terminal change the directory to `fedpfl/federated_learning/example` and launch the first client with:

```bash
python flower_client_cnn.py
```

In another terminal change the directory to `fedpfl/federated_learning/example` And launch the second client with:

```bash
python flower_client_cnn.py
```
