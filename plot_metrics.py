import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
log_dir = Path.cwd() / "log_metrics"
global_path = log_dir / "global_eval_log.csv"
client_paths = {
    "Client 1": log_dir / "client_eval_log_Client_1.csv",
    "Client 2": log_dir / "client_eval_log_Client_2.csv",
    "Client 3": log_dir / "client_eval_log_Client_3.csv",
}

# Load and convert global MAE
df_global = pd.read_csv(global_path)
df_global["MAE"] = pd.to_numeric(df_global["MAE"], errors="coerce")  # <--- Important
df_global = df_global.dropna(subset=["MAE"])  # Remove any bad rows
df_global = df_global.tail(6)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(df_global["Round"], df_global["MAE"], label="Global Model MAE", color="black")

# Load and plot client MAEs
for name, path in client_paths.items():
    if path.exists():
        df_client = pd.read_csv(path)
        df_client["MAE"] = pd.to_numeric(df_client["MAE"], errors="coerce")  # <---
        df_client = df_client.dropna(subset=["MAE"])
        df_client = df_client.tail(5)
        plt.plot(df_client["Round"], df_client["MAE"], label=f"{name} MAE", linestyle="--")

plt.xlabel("Federated Round")
plt.ylabel("MAE")
plt.title("MAE vs Federated Round (Global & Clients)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mae_vs_rounds.png", dpi=300)  # Save the figure
#plt.show()
