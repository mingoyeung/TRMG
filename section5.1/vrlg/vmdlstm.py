import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import optuna
import time
import json
from sklearn.preprocessing import StandardScaler
import random
import os
from datetime import datetime

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    # Configure GPU settings
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_visible_devices(gpus[0], 'GPU')
        except RuntimeError:
            pass


set_random_seed(42)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


class MultiQuantileLoss(keras.losses.Loss):
    """
    Joint multi-quantile loss function.
    Computes tilted absolute loss for all quantiles simultaneously.
    """

    def __init__(self, tau_list, **kwargs):
        super().__init__(**kwargs)
        # Convert tau_list to tensor of shape (1, n_quantiles) for broadcasting
        self.tau_tensor = tf.constant(tau_list, dtype=tf.float32)
        self.tau_tensor = tf.reshape(self.tau_tensor, (1, -1))
        self._one = tf.constant(1.0, dtype=tf.float32)

    def call(self, y_true, y_pred):
        """
        y_true: shape (batch_size, 1) - ground truth values
        y_pred: shape (batch_size, n_quantiles) - predicted quantiles
        """
        # Expand y_true dims to match y_pred: (batch_size, 1) -> (batch_size, n_quantiles)
        y_true_expanded = tf.tile(y_true, [1, tf.shape(y_pred)[1]])

        # Compute error
        error = y_true_expanded - y_pred  # shape: (batch_size, n_quantiles)

        # Compute tilted absolute loss for each quantile
        loss = tf.where(
            error > 0,
            self.tau_tensor * error,
            (self.tau_tensor - self._one) * error
        )

        # Return mean loss across all quantiles
        return tf.reduce_mean(loss)


def build_lstm_model(input_shape, n_quantiles, n_hidden=32, n_hidden2=32, dropout=0.2):
    """
    Build a multi-output LSTM quantile regression network.

    Args:
        input_shape: Input shape (timesteps, features)
        n_quantiles: Number of quantiles (output dimension)
        n_hidden: Number of units in first LSTM layer
        n_hidden2: Number of units in second LSTM layer
        dropout: Dropout rate
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(n_hidden, return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(n_hidden2, return_sequences=False),  # Return only the last timestep output
        layers.Dropout(dropout),
        layers.Dense(n_quantiles)  # Output layer: predict all quantiles simultaneously
    ])
    return model


class LSTMQRModel:
    """
    Multi-output LSTM quantile regression model wrapper.
    Predicts all quantiles in a single forward pass.

    Args:
        n_hidden: Number of units in first LSTM layer
        n_hidden2: Number of units in second LSTM layer
        tau_list: List of quantile levels
        dropout: Dropout rate
    """

    def __init__(
            self,
            n_hidden=32,
            n_hidden2=32,
            tau_list=[0.1, 0.5, 0.9],
            dropout=0.2
    ):
        self.n_hidden = n_hidden
        self.n_hidden2 = n_hidden2
        self.tau_list = tau_list
        self.dropout = dropout

        self.model = None  # Single multi-output model
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit(
            self,
            X_train,
            y_train,
            epochs=500,
            batch_size=32,
            learning_rate=0.01,
            validation_data=None,
            verbose=1,
            early_stopping_patience=50
    ):
        """Train multi-output model"""
        # Reshape X to 3D: (samples, timesteps, features)
        if len(X_train.shape) == 2:
            X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        else:
            X_train_reshaped = X_train

        # Standardize data - for LSTM, normalize before reshaping
        n_samples, n_timesteps, n_features = X_train_reshaped.shape
        X_2d = X_train_reshaped.reshape(-1, n_timesteps * n_features)
        X_scaled_2d = self.scaler_x.fit_transform(X_2d)
        X_scaled = X_scaled_2d.reshape(n_samples, n_timesteps, n_features)

        y_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1))  # shape: (n_samples, 1)

        # Prepare validation data
        val_data = None
        if validation_data is not None:
            X_val, y_val = validation_data
            if len(X_val.shape) == 2:
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

            X_val_2d = X_val.reshape(-1, n_timesteps * n_features)
            X_val_scaled_2d = self.scaler_x.transform(X_val_2d)
            X_val_scaled = X_val_scaled_2d.reshape(X_val.shape[0], n_timesteps, n_features)
            y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1))
            val_data = (X_val_scaled, y_val_scaled)

        if verbose:
            print(f"\nTraining multi-output LSTM model (predicting {len(self.tau_list)} quantiles simultaneously)...")

        # Build model
        self.model = build_lstm_model(
            (n_timesteps, n_features),
            len(self.tau_list),  # output dim = number of quantiles
            self.n_hidden,
            self.n_hidden2,
            self.dropout
        )

        # Compile model with multi-quantile loss function
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=MultiQuantileLoss(tau_list=self.tau_list)
        )

        # Training callbacks
        callbacks = []
        if early_stopping_patience > 0:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor='val_loss' if val_data else 'loss',
                    patience=early_stopping_patience,
                    restore_best_weights=True,
                    verbose=0
                )
            )

        # Train model
        history = self.model.fit(
            X_scaled,
            y_scaled,  # shape: (n_samples, 1)
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data if val_data else None,
            validation_split=0.1 if val_data is None else 0.0,
            callbacks=callbacks,
            verbose=verbose
        )

        return history

    def predict(self, X_test):
        """Predict multiple quantiles in a single forward pass"""
        # Reshape input
        if len(X_test.shape) == 2:
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        n_samples, n_timesteps, n_features = X_test.shape
        X_test_2d = X_test.reshape(-1, n_timesteps * n_features)
        X_scaled_2d = self.scaler_x.transform(X_test_2d)
        X_scaled = X_scaled_2d.reshape(n_samples, n_timesteps, n_features)

        # Predict and inverse-transform
        y_pred_scaled = self.model.predict(X_scaled, verbose=0)  # shape: (n_samples, n_quantiles)

        # Inverse-transform each quantile separately
        predictions = {}
        for i, tau in enumerate(self.tau_list):
            y_pred_i = y_pred_scaled[:, i:i + 1]  # Keep 2D shape
            y_pred = self.scaler_y.inverse_transform(y_pred_i)
            predictions[tau] = y_pred.flatten()

        return predictions


def evaluate_model_denormalized(model, X_val, y_val):
    """
    Evaluate model performance on denormalized data.
    Returns overall loss and per-quantile losses.
    """
    # Get predictions (already denormalized)
    predictions = model.predict(X_val)

    # Compute per-quantile losses
    quantile_losses = {}
    total_loss = 0
    y_val_flat = y_val.flatten()

    for tau, pred in predictions.items():
        error = y_val_flat - pred
        loss = np.mean(np.where(error > 0, tau * error, (tau - 1) * error))
        quantile_losses[f'tau={tau:.2f}'] = float(loss)
        total_loss += loss

    avg_loss = total_loss / len(predictions)

    return avg_loss, quantile_losses


def objective(trial, train_X, train_y, val_X, val_y, tau_list):
    """
    Optuna objective function: minimize multi-quantile loss on validation set.
    """
    # Define hyperparameter search space
    n_hidden = trial.suggest_int('n_hidden', 1, 10)
    n_hidden2 = trial.suggest_int('n_hidden2', 1, 10)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)

    # Create model
    model = LSTMQRModel(
        n_hidden=n_hidden,
        n_hidden2=n_hidden2,
        tau_list=tau_list,
        dropout=dropout
    )

    # Train model
    history = model.fit(
        train_X,
        train_y,
        epochs=200,
        batch_size=64,
        learning_rate=0.01,
        validation_data=(val_X, val_y),
        verbose=0,
        early_stopping_patience=15
    )

    # Evaluate on validation set
    avg_val_loss, _ = evaluate_model_denormalized(model, val_X, val_y)

    return avg_val_loss


def optuna_search_hyperparameters(train_X, train_y, val_X, val_y, tau_list,
                                  n_trials=50, epochs=1000, learning_rate=0.01):
    """Hyperparameter optimization using Optuna"""

    optuna_start_time = time.time()

    print("Starting Optuna hyperparameter search...")

    study = optuna.create_study(
        direction='minimize',
        study_name='lstm_qrnn_hyperparam_opt',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # Pass data via lambda
    study.optimize(
        lambda trial: objective(trial, train_X, train_y, val_X, val_y, tau_list),
        n_trials=n_trials,
        show_progress_bar=True
    )

    optuna_end_time = time.time()
    optuna_duration = optuna_end_time - optuna_start_time

    print("\n" + "=" * 50)
    print("Best hyperparameters:")
    print(f"  n_hidden: {study.best_params['n_hidden']}")
    print(f"  n_hidden2: {study.best_params['n_hidden2']}")
    print(f"  dropout: {study.best_params['dropout']}")
    print(f"  Validation loss (denormalized): {study.best_value:.4f}")
    print(f"  Optuna runtime: {optuna_duration:.2f} seconds")
    print("=" * 50)

    print("\nRetraining model with best hyperparameters...")
    train_start_time = time.time()

    best_model = LSTMQRModel(
        n_hidden=study.best_params['n_hidden'],
        n_hidden2=study.best_params['n_hidden2'],
        tau_list=tau_list,
        dropout=study.best_params['dropout']
    )

    history = best_model.fit(
        train_X,
        train_y,
        epochs=epochs,
        batch_size=512,
        learning_rate=learning_rate,
        validation_data=(val_X, val_y),
        verbose=1,
        early_stopping_patience=30
    )

    train_end_time = time.time()
    train_duration = train_end_time - train_start_time

    timing_info = {
        'optuna_duration_seconds': optuna_duration,
        'final_train_duration_seconds': train_duration,
        'optuna_duration_formatted': f"{optuna_duration / 60:.2f} minutes",
    }

    return study, best_model, timing_info


def save_results_to_json(study, timing_info, test_metrics, filename='qrlstm_optimization_results.json'):
    """Save optimization results to JSON file"""

    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'best_hyperparameters': {
            'n_hidden': int(study.best_params['n_hidden']),
            'n_hidden2': int(study.best_params['n_hidden2']),
            'dropout': float(study.best_params['dropout'])
        },
        'best_validation_loss_denormalized': float(study.best_value),
        'timing': timing_info,
        'test_metrics': test_metrics,
        'optimization_info': {
            'n_trials': len(study.trials),
            'study_name': study.study_name
        }
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nResults saved to: {filename}")


def split_data(data, horizon, window):
    n, m = data.shape
    X = np.zeros((n - window - horizon, window))
    Y = np.zeros((n - window - horizon, 1))
    EX = np.zeros((n - window - horizon, 2))
    for i in range(n - window - horizon):
        start = i
        end = start + window
        X[i, :] = data[start:end, 0]  #Choose: 0-vsub1; 1-vsub2; 2-vsub3; 3-vsub4; 4-vsub5
        Y[i] = data[end + horizon - 1, 0]
        EX[i, :] = data[end + horizon - 1, 5:6]
    return X, Y, EX

if __name__ == "__main__":

    # ===============================================================
    #  MODE selection
    #    'B' -> Option B
    #    'C' -> Option C
    MODE = 'C'
    # ===============================================================
    set_random_seed(42)

    # Load data
    print("Loading data...")
    df = pd.read_excel('vmd_decomposition_fujian.xlsx')

    a = np.array(df.values[:, :])
    # Build time-series dataset
    X, y, EX = split_data(a, horizon=1, window=96)
    X_arg = np.hstack((X, EX))

    # Split: 80% train, 10% validation, 10% test
    n1 = X_arg.shape[0]
    train_X, train_y = X_arg[:round(0.8 * n1)], y[:round(0.8 * n1)]
    val_X, val_y = X_arg[round(0.8 * n1):round(0.9 * n1)], y[round(0.8 * n1):round(0.9 * n1)]
    test_X, test_y = X_arg[round(0.9 * n1):], y[round(0.9 * n1):]
    print(f"Train set: {train_X.shape}, Validation set: {val_X.shape}, Test set: {test_X.shape}")

    # Define hyperparameters
    EPOCHS = 1000
    LEARNING_RATE = 0.01
    N_TRIALS = 100
    tau_list = [round(i, 2) for i in np.arange(0.05, 1.0, 0.05)]

    # ===============Using hyperparameters directly, without optuna (Option B)==================
    if MODE == 'B':
        RESULTS_JSON = 'vmdlstmf1_results.json'
        # Change:vsub1:vmdlstmf1_results;vsub2:vmdlstmf2_results;vsub3:vmdlstmf3_results;vsub4:vmdlstmf4_results;vsub5:vmdlstmf5_results
        print(f"\nLoading best hyperparameters from {RESULTS_JSON}...")
        with open(RESULTS_JSON, 'r', encoding='utf-8') as f:
            saved = json.load(f)

        best_params = saved['best_hyperparameters']
        n_hidden  = best_params['n_hidden']
        n_hidden2 = best_params['n_hidden2']
        dropout   = best_params['dropout']

        print(f"  n_hidden  = {n_hidden}")
        print(f"  n_hidden2 = {n_hidden2}")
        print(f"  dropout   = {dropout}")
        print(f"  (Source timestamp: {saved['timestamp']})")

        print("\nTraining final model with best parameters...")
        best_model = LSTMQRModel(n_hidden=n_hidden,
                                 n_hidden2=n_hidden2,
                                 tau_list=tau_list,
                                 dropout=dropout)

        best_model.fit(
            train_X, train_y,
            epochs=EPOCHS, learning_rate=LEARNING_RATE, verbose=True,
            validation_data=(val_X, val_y), early_stopping_patience=30)

        #Validation set evaluation
        print("\n" + "=" * 50)
        print("Validation set evaluation (denormalized):")
        print("=" * 50)
        val_total_loss, val_quantile_losses = evaluate_model_denormalized(
            best_model, val_X, val_y)
        print(f"Overall loss: {val_total_loss:.4f}")
        for quantile, loss in val_quantile_losses.items():
            print(f"  {quantile}: {loss:.4f}")

        #Test set evaluation
        print("\n" + "=" * 50)
        print("Test set evaluation (denormalized):")
        print("=" * 50)
        test_total_loss, test_quantile_losses = evaluate_model_denormalized(
            best_model, test_X, test_y)

        print(f"Overall loss: {test_total_loss:.4f}")
        for quantile, loss in test_quantile_losses.items():
            print(f"  {quantile}: {loss:.4f}")
    #  =======================Optuna searching (Option C),uncomment to use================================
    elif MODE == 'C':
        print(f"\nStart hyperparameter optimization (trials={N_TRIALS}, epochs={EPOCHS})...")
        study, best_model, timing_info = optuna_search_hyperparameters(
            train_X, train_y, val_X, val_y,tau_list,
            n_trials=N_TRIALS,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE)

        history = best_model.fit(train_X, train_y,
                                 epochs=EPOCHS,
                                 learning_rate=LEARNING_RATE,
                                 validation_data=(val_X, val_y),
                                 verbose=True,
                                 early_stopping_patience=30)

        #Validation set evaluation
        print("\n" + "=" * 50)
        print("Validation set evaluation (denormalized):")
        print("=" * 50)
        val_total_loss, val_quantile_losses = evaluate_model_denormalized(
            best_model, val_X, val_y)

        print(f"Overall loss: {val_total_loss:.4f}")
        for quantile, loss in val_quantile_losses.items():
            print(f"  {quantile}: {loss:.4f}")

        #Test set evaluation
        print("\n" + "=" * 50)
        print("Test set evaluation (denormalized):")
        print("=" * 50)
        test_start_time = time.time()
        test_total_loss, test_quantile_losses = evaluate_model_denormalized(
            best_model, test_X, test_y)

        test_duration = time.time() - test_start_time

        print(f"Overall loss: {test_total_loss:.4f}")
        for quantile, loss in test_quantile_losses.items():
            print(f"  {quantile}: {loss:.4f}")
        print(f"Test duration: {test_duration:.4f} seconds")

        timing_info['test_duration_seconds'] = test_duration
        timing_info['test_duration_formatted'] = f"{test_duration:.4f} seconds"

        test_metrics = {
        'total_loss_denormalized': float(test_total_loss),
        'quantile_losses_denormalized': test_quantile_losses}

        save_results_to_json(
            study, timing_info, test_metrics,
            filename='vmdlstmf1_results.json')
        # Obtain:vsub1:vmdlstmf1_results;vsub2:vmdlstmf2_results;vsub3:vmdlstmf3_results;vsub4:vmdlstmf4_results;vsub5:vmdlstmf5_results
    #  =======================Generating prediction results================================
    predictions = best_model.predict(test_X)

    # Organize predictions into DataFrame
    pred_df = pd.DataFrame(predictions)
    pred_df.columns = [f'quantile_{tau:.2f}' for tau in pred_df.columns]
    pred_df.to_excel('vmdlstmf1.xlsx', index=False)#Save:vsub1:vmdlstmf1;vsub2:vmdlstmf2;vsub3:vmdlstmf3;vsub4:vmdlstmf4;vsub5:vmdlstmf5