from typing import Dict, Optional, Tuple
from pathlib import Path
import argparse
import os
from pathlib import Path
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import flwr as fl
import csv
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix,accuracy_score
import numpy as np

save = []
cnt=0
client_number = 4
save.append(('Epochs', 'Loss', 'Accuracy', 'AUC', 'Sensitivity', 'Specificity'))
def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu', input_shape=(52, 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.BinaryCrossentropy()

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])


    # def save_model_callback(server, num_round):
    #     # Save the global model after each round
    #     model.save(f"global_model_round_{num_round}.h5")
    #     print(f"Global model saved to 'global_model_round_{num_round}.h5'")



    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5, # (0.1 - 0.5) The fraction of clients used for training on each round.
        fraction_evaluate=0.2, # (0.05 - 0.2) The fraction of clients that will be randomly selected to evaluate the global model after each round
        min_fit_clients=4, # (3 - 5) The minimum number of clients that must be available to participate in each round
        min_evaluate_clients=2, # (2 - 3) The minimum number of clients that must be available to evaluate the global model
        min_available_clients=4,# (20% - 30%) The minimum number of clients that must be connected and available to participate in each round
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )

    # Start Flower server (SSL-enabled) for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=2),
        strategy=strategy,
        certificates=(
            Path(".cache/certificates/ca.crt").read_bytes(),
            Path(".cache/certificates/server.pem").read_bytes(),
            Path(".cache/certificates/server.key").read_bytes(),
        ),
         # Add the save_model_callback to the on_round_end method
        # on_round_end=save_model_callback,
    )

    model.save(f"global_model_client_{client_number}.h5")


def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    df = pd.read_csv('G:\\FL&&OB\\Dataset\\Parts\\part_1.csv')
    y_val = df["Is Laundering"]
    x_val = df.drop(columns=["Is Laundering"])
    


    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        global save
        global cnt
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)

        y_pred = model.predict(np.expand_dims(x_val, axis=2))
        y_pred = np.round(y_pred)

        cm = confusion_matrix(y_val, y_pred)
        sensitivity = cm[0][0]/(cm[0][0]+cm[1][0])
        specificity = cm[1][1]/(cm[0][1]+cm[1][1])
        auc = roc_auc_score(y_val, y_pred)
        save.append((cnt, loss, accuracy, auc, sensitivity, specificity))
        cnt+=1

        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        # "local_epochs": 1 if server_round < 2 else 2,
        "local_epochs": 2,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()
    print(save)
    with open(f'metrics_{client_number}.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(save)

