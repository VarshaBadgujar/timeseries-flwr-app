"""timeseries-flwr-app: A Flower / TensorFlow app."""
import os
from typing import List, Tuple
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from timeseries_flwr_app.task import build_model, get_weights, get_centralized_test_data


# Make TensorFlow log less verbose, show only errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregates metrics from an evaluate round."""
    # Loop trough all metrics received compute maes x examples
    maes = [num_examples * m["mae"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    # Return weighted average accuracy
    return {"federated_evaluate_mae": sum(maes) / total_examples}

# Global/Centralized model evaluation function
def gen_evaluate_fn(x_test, y_test, n_input, num_features, n_output):
    """Return a callback that evaluates the global model.""" 
    def evaluate(server_round, parameters, _): # '_' for unused config
        """Evaluate global model using provided centralised testset."""
        # Instantiate model
        model = build_model(n_input, num_features, n_output)
        # Apply global_model parameters
        model.set_weights(parameters)
        loss, mae = model.evaluate(x_test, y_test, verbose=0)
        print(f"[Centralized Evaluation] Round {server_round}:")
        return loss, {"centralized_evaluate_mae": mae}
    return evaluate

def handle_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """ Aggregate and log metrics from a fit round.
    This function showed a mechanism to communicate almost arbitrary metrics by
    converting them into a JSON on the ClientApp side. Here that JSON string is
    deserialized and reinterpreted as a dictionary we can use.
    """
   # Weighted average
    total_examples = sum(num_examples for num_examples, _ in metrics)
    avg_loss = sum(m["loss"] * num_examples for num_examples, m in metrics) / total_examples
    avg_mae  = sum(m["mae"] * num_examples for num_examples, m in metrics) / total_examples

    return {"avg_loss": avg_loss, "avg_mae": avg_mae}


def on_fit_config(server_round: int) -> Metrics:
    """ Adjusts learning rate based on current round.
        Construct `config` that clients receive when running `fit()`
    """
    lr = 0.01
    # Appply a simple learning rate decay
    if server_round > 3:
        lr = 0.008
    return {"lr": lr}


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Hyperparameters from pyproject.toml
    n_input = context.run_config["n_input"]
    n_output = context.run_config["n_output"]
    num_features = context.run_config["num_features"]
    learning_rate = context.run_config["learning-rate"]
    stride = context.run_config["stride"]
    one_year_hours = context.run_config["one_year_hours"]
    # Get initial parameters to initialize global model
    #parameters = ndarrays_to_parameters(build_model(n_input, num_features, n_output).get_weights())

    # Initialize model parameters
    ndarrays = get_weights(build_model(n_input, num_features, n_output, learning_rate))
    parameters = ndarrays_to_parameters(ndarrays)

    x_test, y_test = get_centralized_test_data(
                        n_input=n_input,
                        n_output=n_output,
                        stride=stride,
                        one_year_hours=one_year_hours,
                        )
    # Define strategy
    strategy = FedAvg(
        fraction_fit = 1.0,
        fraction_evaluate = 1.0,
        min_available_clients = 2,
        #min_evaluate_clients = 2,
        #eta = 0.001,      # Learning rate
        #beta_1 = 0.9,
        #beta_2 = 0.99,
        #tau=1e-9,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=on_fit_config,
        fit_metrics_aggregation_fn=handle_fit_metrics,
        evaluate_fn=gen_evaluate_fn(
                        x_test,
                        y_test,
                        n_input,
                        num_features,
                        n_output),
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds = num_rounds)

    return ServerAppComponents(strategy = strategy, config = config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
