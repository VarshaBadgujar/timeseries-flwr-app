"""timeseries-flwr-app: A Flower / TensorFlow app - Data and model"""
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# path and client files
DATA_DIR = "/home/varsha/flwr_tf_project/data"  # Path to .pkl files
client_files = [
    os.path.join(DATA_DIR, "client1_695645.pkl"),
    os.path.join(DATA_DIR, "client2_695947.pkl"),
    os.path.join(DATA_DIR, "client3_696204.pkl")
]
client_ids = ['Client_1', 'Client_2', 'Client_3']

# Model architecture
def build_model(n_input, num_features, n_output, learning_rate: float = 0.01):
    """ Build and compile LSTM model """
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(n_input, num_features)),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(n_output),
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model.compile(optimizer = optimizer, loss='mse', metrics=['mae'])
    return model

# Data loading and split
def load_and_split(client_file, train_ratio, val_ratio, client_name, one_year_hours):
    """ Data loading, preparation and train-test-val split"""
    df = pd.read_pickle(client_file)

    if len(df) < one_year_hours:
        print(f"Warning: {client_name} has less than {one_year_hours} of data")
    else:
        df = df[-one_year_hours:]

    df = df[['kwh', 'm3h']].astype(np.float32)

    total = len(df)
    train_end = int(train_ratio * total)
    val_end = int((train_ratio + val_ratio) * total)

    train, val, test = df[:train_end], df[train_end:val_end], df[val_end:]

    # MinMax scaler
    full_scaler = MinMaxScaler()
    train_scaled = full_scaler.fit_transform(train)
    val_scaled = full_scaler.transform(val)
    test_scaled = full_scaler.transform(test)

    target_scaler = MinMaxScaler()
    target_scaler.fit(train[['kwh']])

    return train_scaled, val_scaled, test_scaled, full_scaler, target_scaler

def create_multistep_sequences(data, n_input, n_output, stride):
    """ Sequence generation for time-series data """
    x, y = [], []
    for i in range(0, len(data) - n_input - n_output + 1, stride):
        x.append(data[i:i + n_input, 0:2])
        y.append(data[i + n_input:i + n_input + n_output, 0])
    return np.array(x), np.array(y)

def get_client_data(partition_id, n_input, n_output, stride, one_year_hours):
    """Load and preprocess data for a specific client."""
    client_file = client_files[partition_id]
    client_name = client_ids[partition_id]

    train, val, test, full_scaler, target_scaler = load_and_split(
        client_file, train_ratio=0.6, val_ratio = 0.2,
        client_name=client_name, one_year_hours=one_year_hours
    )

    x_train, y_train = create_multistep_sequences(train, n_input, n_output, stride)
    x_val, y_val = create_multistep_sequences(val, n_input, n_output, stride)
    x_test, y_test = create_multistep_sequences(test, n_input, n_output, stride)

    return {
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "test": (x_test, y_test),
        "full_scaler": full_scaler,
        "target_scaler": target_scaler,
        "cid": client_name,
    }

def get_centralized_test_data(n_input, n_output, stride, one_year_hours):
    """Use each client's get_client_data() to gather and combine all test sequences."""
    all_x_test, all_y_test = [], []

    for client_idx in range(len(client_files)):
        client_data = get_client_data(client_idx, n_input, n_output, stride, one_year_hours)
        x_test, y_test = client_data["test"]

        all_x_test.append(x_test)
        all_y_test.append(y_test)

    # Combine all test data
    x_test_all = np.concatenate(all_x_test, axis=0)
    y_test_all = np.concatenate(all_y_test, axis=0)

    return x_test_all, y_test_all

def get_weights(model):
    """Extract parameters from a TensorFlow (Keras) model.

    This returns a list of NumPy arrays representing the model's weights.
    """
    return [w.numpy() for w in model.weights]
