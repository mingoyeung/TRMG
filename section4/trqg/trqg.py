import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import optuna
import time
from sklearn.preprocessing import StandardScaler
import random
import os
import json

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
# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

class MultiQuantileLoss(keras.losses.Loss):

    def __init__(self, tau_list, **kwargs):
        super().__init__(**kwargs)

        self.tau_tensor = tf.constant(tau_list, dtype=tf.float32)
        self.tau_tensor = tf.reshape(self.tau_tensor, (1, -1))
        self._one = tf.constant(1.0, dtype=tf.float32)

    def call(self, y_true, y_pred):

        y_true_expanded = tf.tile(y_true, [1, tf.shape(y_pred)[1]])

        error = y_true_expanded - y_pred  # shape: (batch_size, n_quantiles)

        loss = tf.where(
            error > 0,
            self.tau_tensor * error,
            (self.tau_tensor - self._one) * error
        )

        return tf.reduce_mean(loss)

def build_qrnn_model(input_dim, n_quantiles, n_hidden=15, n_hidden2=15, activation='sigmoid', penalty=0.0):

    regularizer = keras.regularizers.l2(penalty) if penalty > 0 else None

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(n_hidden, activation=activation, kernel_regularizer=regularizer),
        layers.Dense(n_hidden2, activation=activation, kernel_regularizer=regularizer),
        layers.Dense(n_quantiles, kernel_regularizer=regularizer)
    ])
    return model


class QRNNModel:

    def __init__(
            self,
            n_hidden=15,
            n_hidden2=15,
            tau_list=[0.1, 0.5, 0.9],
            penalty=0.0,
            activation='sigmoid'
    ):
        self.n_hidden = n_hidden
        self.n_hidden2 = n_hidden2
        self.tau_list = tau_list
        self.penalty = penalty
        self.activation = activation

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
            verbose=0,
            early_stopping_patience=30
    ):
        # Standardize data
        X_scaled = self.scaler_x.fit_transform(X_train)
        y_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1))  # shape: (n_samples, 1)

        # Prepare validation data
        val_data = None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_scaled = self.scaler_x.transform(X_val)
            y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1))
            val_data = (X_val_scaled, y_val_scaled)

        if verbose:
            print(f"\nTraining multi-output model (predicting {len(self.tau_list)} quantiles simultaneously)...")

        self.model = build_qrnn_model(
            X_train.shape[1],
            len(self.tau_list),  # output dim = number of quantiles
            self.n_hidden,
            self.n_hidden2,
            self.activation,
            self.penalty
        )

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=MultiQuantileLoss(tau_list=self.tau_list)
        )

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

        history = self.model.fit(
            X_scaled,
            y_scaled,  # shape: (n_samples, 1)
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data if val_data else None,
            callbacks=callbacks,
            verbose=verbose
        )

        return history

    def predict(self, X_test):

        X_scaled = self.scaler_x.transform(X_test)

        y_pred_scaled = self.model.predict(X_scaled, verbose=0)  # shape: (n_samples, n_quantiles)

        # Inverse-transform each quantile separately
        predictions = {}
        for i, tau in enumerate(self.tau_list):
            y_pred_i = y_pred_scaled[:, i:i + 1]  # Keep 2D shape
            y_pred = self.scaler_y.inverse_transform(y_pred_i)
            predictions[tau] = y_pred.flatten()

        return predictions


def objective(trial, train_X, train_y, val_X, val_y):
    """
    Optuna objective function: minimize multi-quantile loss on validation set.
    """
    # Define hyperparameter search space
    n_hidden = trial.suggest_int('n_hidden', 1, 10)
    n_hidden2 = trial.suggest_int('n_hidden2', 1, 10)
    penalty = (trial.suggest_float('penalty', 1e-4, 1e-1))

    # Keep tau list unchanged
    tau_list = [round(i, 2) for i in np.arange(0.05, 1.0, 0.05).tolist()]

    # Create model
    model = QRNNModel(
        n_hidden=n_hidden,
        n_hidden2=n_hidden2,
        tau_list=tau_list,
        penalty=penalty
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

    # Predict on validation set
    val_predictions = model.predict(val_X)

    # Compute validation loss
    total_val_loss = 0
    val_y_flat = val_y.flatten()

    for tau, pred in val_predictions.items():
        error = val_y_flat - pred
        loss = np.mean(np.where(error > 0, tau * error, (tau - 1) * error))
        total_val_loss += loss

    avg_val_loss = total_val_loss / len(tau_list)

    return avg_val_loss

def optimize_hyperparameters(train_X, train_y, val_X, val_y, tau_list,
                              n_trials=50, epochs=1000, learning_rate=0.01):
    """Hyperparameter optimization using Optuna"""
    print("Starting hyperparameter optimization...")
    optuna_start_time = time.time()
 
    study = optuna.create_study(
        direction='minimize',
        study_name='qrnn_hyperparam_opt',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
 
    # Pass data via lambda
    study.optimize(
        lambda trial: objective(trial, train_X, train_y, val_X, val_y),
        n_trials=n_trials,
        show_progress_bar=True
    )
    optuna_end_time = time.time()
    optuna_duration = optuna_end_time - optuna_start_time
 
    print("\nOptimization complete!")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_trial.value:.6f}")
    print(f"Best hyperparameters: {study.best_trial.params}")
    print(f"  Optuna search duration: {optuna_duration:.2f} seconds")
    print("=" * 50)
 
    print("\nRetraining model with best parameters...")
    best_params = study.best_trial.params
 
    final_model = QRNNModel(
        n_hidden=best_params['n_hidden'],
        n_hidden2=best_params['n_hidden2'],
        tau_list=tau_list,
        penalty=best_params['penalty']
    )
 
    train_start_time = time.time()
    history = final_model.fit(
        train_X,
        train_y,
        epochs=epochs,
        batch_size=512,
        learning_rate=learning_rate,
        validation_data=(val_X, val_y),
        verbose=0,
        early_stopping_patience=30
    )
    train_end_time = time.time()
    train_duration = train_end_time - train_start_time
 
    timing_info = {
        'optuna_duration_seconds': f"{optuna_duration:.1f}",
        'optuna_duration_formatted': f"{optuna_duration:.3f} seconds",
    }
    return study, timing_info, final_model

def save_results_to_json(study, timing_info, test_metrics, filename='trqg1_results.json'):

    results = {
        'best_hyperparameters': {
            'n_hidden': int(study.best_params['n_hidden']),
            'n_hidden2': int(study.best_params['n_hidden2']),
            'penalty': float(study.best_params['penalty'])
        },
        'best_validation_loss': float(study.best_value),
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

    for i in range(n - window - horizon):
        start = i
        end = start + window
        X[i, :] = data[start:end, 0]
        Y[i] = data[end + horizon - 1, 0]

    return X, Y

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
    df = pd.read_excel('imfreconstruction_TenneT.xlsx')
    a = np.array(df.values[:, 0])#Choose:0-sub1; 1-sub2; 2-sub3; 3-sub4; 4-sub5
    a = np.expand_dims(a, axis=1)

    # Create time series dataset
    print("Create time series dataset...")
    X, Y = split_data(a, horizon=1, window=96)

    # Data split: 80% train, 10% validation, 10% test
    n1 = X.shape[0]
    train_X, train_y = X[:round(0.8 * n1)], Y[:round(0.8 * n1)]
    val_X, val_y = X[round(0.8 * n1):round(0.9 * n1)], Y[round(0.8 * n1):round(0.9 * n1)]
    test_X, test_y = X[round(0.9 * n1):], Y[round(0.9 * n1):]
    print(f"Train: {train_X.shape}, Val: {val_X.shape}, Test: {test_X.shape}")

    EPOCHS = 1000
    LEARNING_RATE = 0.01
    N_TRIALS = 100
    tau_list = [round(i, 2) for i in np.arange(0.05, 1.0, 0.05)]
    # ===============Using hyperparameters directly, without optuna (Option B)==================
    if MODE == 'B':
        RESULTS_JSON = 'trqg1_results.json'#Change:sub1:trqg1_results;sub2:trqg2_results;sub3:trqg3_results;sub4:trqg4_results;sub5:trqg5_results
        print(f"\nLoading best hyperparameters from {RESULTS_JSON}...")
        with open(RESULTS_JSON, 'r', encoding='utf-8') as f:
            saved = json.load(f)

        best_params = saved['best_hyperparameters']
        n_hidden  = best_params['n_hidden']
        n_hidden2 = best_params['n_hidden2']
        penalty   = best_params['penalty']

        print(f"  n_hidden  = {n_hidden}")
        print(f"  n_hidden2 = {n_hidden2}")
        print(f"  penalty   = {penalty:.6f}")

        print("\nTraining final model with best parameters...")

        final_model = QRNNModel(n_hidden=best_params['n_hidden'],
                                n_hidden2=best_params['n_hidden2'],
                                tau_list=tau_list,
                                penalty=best_params['penalty'])
        
        history = final_model.fit(
            train_X,
            train_y,
            epochs=EPOCHS,
            batch_size=512,
            learning_rate=LEARNING_RATE,
            validation_data=(val_X, val_y),
            verbose=0,
            early_stopping_patience=30)

        #Validation set evaluation
        print("\n" + "=" * 50)
        print("Validation set evaluation:")
        print("=" * 50)
        val_predictions = final_model.predict(val_X)
        val_total_loss = 0
        val_quantile_losses = {}

        for tau, pred in val_predictions.items():
            error = val_y.flatten() - pred
            loss = np.mean(np.where(error > 0, tau * error, (tau - 1) * error))
            val_total_loss += loss
            val_quantile_losses[f'quantile_{tau:.2f}'] = float(loss)
            print(f"  Quantile {tau:.2f}: {loss:.4f}")

        print(f"best_validation_loss: {val_total_loss / len(val_predictions):.4f}")

        #Test set evaluation
        print("\n" + "=" * 50)
        print("Test set evaluation:")
        print("=" * 50)
        test_predictions = final_model.predict(test_X)
        test_total_loss = 0
        test_quantile_losses = {}

        for tau, pred in test_predictions.items():
            error = test_y.flatten() - pred
            loss = np.mean(np.where(error > 0, tau * error, (tau - 1) * error))
            test_total_loss += loss
            test_quantile_losses[f'quantile_{tau:.2f}'] = float(loss)
            print(f"  Quantile {tau:.2f}: {loss:.4f}")

        print(f"total_loss: {test_total_loss / len(test_predictions):.4f}")
    #  =======================Optuna searching (Option C),uncomment to use================================
    elif MODE == 'C':
        print(f"\nStart hyperparameter optimization (trials={N_TRIALS}, epochs={EPOCHS})...")

        study, timing_info, best_model = optimize_hyperparameters(train_X, train_y, val_X, val_y, tau_list=tau_list,n_trials=N_TRIALS)

        print("\n" + "=" * 50)
        print("Best hyperparameters:")
        print(f"  n_hidden: {study.best_params['n_hidden']}")
        print(f"  n_hidden2: {study.best_params['n_hidden2']}")
        print(f"  penalty: {study.best_params['penalty']:.6f}")
        print("=" * 50)

        # Validation set evaluation
        print("\n" + "=" * 50)
        print("Validation set evaluation:")
        print("=" * 50)
        val_predictions = best_model.predict(val_X)
        val_total_loss = 0
        val_quantile_losses = {}

        for tau, pred in val_predictions.items():
            error = val_y.flatten() - pred
            loss = np.mean(np.where(error > 0, tau * error, (tau - 1) * error))
            val_total_loss += loss
            val_quantile_losses[f'quantile_{tau:.2f}'] = float(loss)
            print(f"  Quantile {tau:.2f}: {loss:.4f}")

        print(f"Overall loss: {val_total_loss / len(val_predictions):.4f}")

        #Test set evaluation
        print("\n" + "=" * 50)
        print("Test set evaluation:")
        print("=" * 50)
        test_start_time = time.time()
        test_predictions = best_model.predict(test_X)
        test_duration = time.time() - test_start_time

        test_total_loss = 0
        test_quantile_losses = {}

        for tau, pred in test_predictions.items():
            error = test_y.flatten() - pred
            loss = np.mean(np.where(error > 0, tau * error, (tau - 1) * error))
            test_total_loss += loss
            test_quantile_losses[f'quantile_{tau:.2f}'] = float(loss)
            print(f"  Quantile {tau:.2f}: {loss:.4f}")

        print(f"Overall loss: {test_total_loss / len(test_predictions):.4f}")
        print(f"Test duration: {test_duration:.4f} seconds")

        timing_info['test_duration_seconds'] = test_duration
        timing_info['test_duration_formatted'] = f"{test_duration:.4f} seconds"

        test_metrics = {
        'total_loss': float(test_total_loss),
        'quantile_losses': test_quantile_losses}

        save_results_to_json(study, timing_info, test_metrics,
                             filename='trqg1_results.json')#Obtain:sub1:trqg1_results;sub2:trqg2_results;sub3:trqg3_results;sub4:trqg4_results;sub5:trqg5_results
    #  =======================Generating prediction results================================
    print("\nGenerating prediction results...")
    pred_df = pd.DataFrame(test_predictions)
    pred_df.columns = [f'quantile_{tau:.2f}' for tau in pred_df.columns]
    pred_df.to_excel('trqg1.xlsx', index=False)#Save: sub1:trqg1;sub2:trqg2;sub3:trqg3;sub4:trqg4;sub5:trqg5