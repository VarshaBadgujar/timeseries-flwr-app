"""timeseries-flwr-app: A Flower / TensorFlow app."""
import os
import csv
from typing import Any
from pathlib import Path
import tensorflow as tf
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context, RecordDict
from timeseries_flwr_app.task import build_model, get_client_data, get_weights

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Only show errors

def log_client_metrics(client_id, round_num, loss, mae):
    """ Log each client’s metrics during training rounds """

    # Create a "log_metrics" directory under the current working directory
    log_dir = Path.cwd() / "log_metrics"
    log_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

    # Full path to log file
    filename = log_dir / f"client_eval_log_{client_id}.csv"

    file_exists = Path(filename).exists()
    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Client", "Round", "Loss", "MAE"])
        writer.writerow([client_id, round_num, loss, mae])


# Define Flower Client and client_fn
class LSTMClient(NumPyClient):
    """ LSTM client model """
    def __init__(self, client_state: RecordDict, data, n_input,
                 num_features, n_output, epochs, batch_size, verbose):
        self.client_state = client_state
        self.data = data
        self.n_input = n_input
        self.num_features = num_features
        self.n_output = n_output
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.x_train, self.y_train = data["train"]
        self.x_val, self.y_val = data["val"]
        self.cid: Any = data['cid']
        self.model = build_model(self.n_input, self.num_features, self.n_output)

    def get_parameters(self, config):
        """if self.model is None:
            # Build model architecture without compiling
            self.model = build_model(self.n_input, self.num_features, self.n_output)
        """
        return self.model.get_weights()

    def fit(self, parameters, config):
        # config expected to have "lr"
        learning_rate = float(config.get("lr", 0.01))  # default lr fallback

        # Build model if not yet built
        if self.model is None:
            self.model = build_model(self.n_input, self.num_features, self.n_output, learning_rate)

        self.model.set_weights(parameters)
        print(config)
        history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        final_loss = history.history["loss"][-1]
        final_mae = history.history["mae"][-1]
        return get_weights(self.model), len(self.x_train), {
                "loss": final_loss,
                "mae": final_mae,
        }

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        # Update local model with global parameters
        self.model.set_weights(parameters)
        # Evaluate global model parameters on the local test data
        loss, mae = self.model.evaluate(self.x_val, self.y_val, verbose=0)

        # Log and Return results, including the custom mae metric
        round_num = config.get("server_round", 0)
        log_client_metrics(self.cid, round_num, loss, mae)
        return loss, len(self.x_val), {"mae": mae}


def client_fn(context: Context):
    """Construct a Flower Client for TensorFlow simulation using predefined data."""
    # Ensure a new session is started
    tf.keras.backend.clear_session()
    # Extract hyperparameters from config
    # Identify client
    partition_id = context.node_config["partition-id"]

    # Hyperparameters from pyproject.toml
    n_input = context.run_config["n_input"]
    n_output = context.run_config["n_output"]
    stride = context.run_config["stride"]
    one_year_hours = context.run_config["one_year_hours"]
    num_features = context.run_config["num_features"]
    #learning_rate = context.run_config["learning-rate"]

    # Load client data from task.py
    data = get_client_data(partition_id, n_input, n_output, stride, one_year_hours)

    # Load Model
    #model = build_model(n_input, num_features, n_output, float(config["lr"]))
    # Other training param
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")
    # Return Client instance
    # We pass the state to persist information across
    # participation rounds. Note that each client always
    # receives the same Context instance (it's a 1:1 mapping)
    client_state = context.state

    # Initialize LSTMClient
    #return LSTMClient(client_state, model, data, epochs, batch_size, verbose).to_client()
    return LSTMClient(client_state, data, n_input, num_features,
                      n_output, epochs, batch_size, verbose).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)
