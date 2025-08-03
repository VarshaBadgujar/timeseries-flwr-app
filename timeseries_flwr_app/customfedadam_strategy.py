""" Custom FedAdam Strategy """
import csv
from pathlib import Path
from typing import Optional, Tuple, Dict
from flwr.server.strategy import FedAdam
from flwr.common import Scalar, Parameters
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class CustomFedAdam(FedAdam):
    """ Custom FedAdam Strategy- has extra functionality
    This strategy: (1) saves results to the filesystem, (2) saves a
    checkpoint of the global  model when a new best is found, (3) logs
    results to W&B if enabled.
    """
    def __init__(self, *args, num_rounds=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_rounds = num_rounds
        self.global_log = []

    def log_global_metrics(self, rnd: int, y_true, y_pred, loss, mae):
        """ Log global metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        self.global_log.append((rnd, loss, mae, rmse, r2))

    def save_global_log(self, path: Path = None):
        """ Save global metrics """
        if path is None:
            log_dir = Path.cwd() / "log_metrics"
            log_dir.mkdir(parents=True, exist_ok=True)
            path = log_dir / "global_eval_log.csv"
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Round", "Loss", "MAE", "RMSE", "R2"])
            writer.writerows(self.global_log)

    def store_results_and_log(self, server_round: int, tag: str, results_dict: dict):
        """federated_evaluate - A helper method that stores results and logs them to CSV."""
        # Store for internal tracking (e.g., Flower History)
        #self._store_results(tag=tag, results_dict={"round": server_round, **results_dict})

        # Save to CSV
        csv_dir = Path.cwd() / "log_metrics"
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_file = csv_dir / f"{tag}_log.csv"

        # If file doesn't exist, write header
        write_header = not csv_file.exists()
        with csv_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["round"] + list(results_dict.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow({"round": server_round, **results_dict})

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model using centralized test set."""
        if self.evaluate_fn is None:
            return None
        loss, metrics = self.evaluate_fn(server_round, parameters, {})
        y_true = metrics.get("y_true")
        y_pred = metrics.get("y_pred")
        mae = metrics.get("centralized_evaluate_mae", 0.0)

        if y_true is not None and y_pred is not None:
            self.log_global_metrics(server_round, y_true, y_pred, loss, mae)

        if server_round == self.num_rounds:
            self.save_global_log()

        return loss, {"mae": mae}

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate results from federated evaluation."""
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        # Store and log
        self.store_results_and_log(
            server_round=server_round,
            tag="federated_evaluate",
            results_dict={"federated_evaluate_loss": loss, **metrics},
        )
        return loss, metrics
