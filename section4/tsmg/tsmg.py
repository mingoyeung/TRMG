import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import regularizers
import optuna
from sklearn.preprocessing import StandardScaler
import time
import random
import os
import json
from datetime import datetime

tf.config.run_functions_eagerly(True)

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

class PositiveConstraint(tf.keras.constraints.Constraint):

    def __init__(self, min_value=1e-7):
        self.min_value = min_value

    def __call__(self, w):
        w = tf.nn.relu(w)
        w = w + self.min_value
        return w

class McqrnnInputDense(tf.keras.layers.Layer):
    def __init__(self, out_features, activation, penalty=0, **kwargs):
        super().__init__(**kwargs)
        self.out_features = out_features
        self.activation = activation
        self.penalty = penalty

    def build(self, input_shape):
        self.w_inputs = self.add_weight(
            name="w_inputs",
            shape=(input_shape[-1], self.out_features),
            initializer="random_normal",
            regularizer=regularizers.l2(self.penalty),
            trainable=True
        )
        self.w_tau = self.add_weight(
            shape=(1, self.out_features),
            initializer="random_normal",
            constraint=PositiveConstraint(),
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.out_features,),
            initializer="random_normal",
            trainable=True
        )

    def call(self, inputs, tau):
        tau_weighted = tf.matmul(tau, self.w_tau)
        inputs_weighted = tf.matmul(inputs, self.w_inputs)
        outputs = tau_weighted + inputs_weighted + self.b
        return self.activation(outputs)


class McqrnnDense(tf.keras.layers.Layer):
    def __init__(self, dense_features, activation, penalty=0, **kwargs):
        super().__init__(**kwargs)
        self.dense_features = dense_features
        self.activation = activation
        self.penalty = penalty

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.dense_features),
            initializer="random_normal",
            regularizer=regularizers.l2(self.penalty),
            constraint=PositiveConstraint(),
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.dense_features,),
            initializer="random_normal",
            trainable=True
        )

    def call(self, inputs):
        outputs = tf.matmul(inputs, self.w) + self.b
        return self.activation(outputs)

class McqrnnOutputDense(tf.keras.layers.Layer):

    def __init__(self, penalty=0, **kwargs):
        super().__init__(**kwargs)
        self.penalty = penalty

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            regularizer=regularizers.l2(self.penalty),
            constraint=PositiveConstraint(),
            trainable=True
        )
        self.b = self.add_weight(
            shape=(1,),
            initializer="random_normal",
            trainable=True
        )

    def call(self, inputs):
        outputs = tf.matmul(inputs, self.w) + self.b
        return outputs

class Mcqrnn(tf.keras.Model):
    def __init__(self, n_hidden, n_hidden2, penalty=0, activation=tf.nn.sigmoid, **kwargs):
        super().__init__(**kwargs)
        self.input_dense = McqrnnInputDense(n_hidden, activation, penalty)
        self.dense = McqrnnDense(n_hidden2, activation, penalty)
        self.output_dense = McqrnnOutputDense(penalty)

    def call(self, inputs, tau):
        x = self.input_dense(inputs, tau)
        x = self.dense(x)
        return self.output_dense(x)

class TiltedAbsoluteLoss(tf.keras.losses.Loss):
    def __init__(self, tau, **kwargs):
        super().__init__(**kwargs)
        self._tau = tf.cast(tau, dtype=tf.float32)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        loss = tf.where(error >= 0, self._tau * error, (self._tau - 1) * error)
        return tf.reduce_mean(loss)

class DataScaler:

    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        self.is_fitted = False

    def fit_transform(self, X, y):

        X_scaled = self.scaler_X.fit_transform(X).astype('float32')
        y_scaled = self.scaler_Y.fit_transform(y.reshape(-1, 1)).astype('float32')
        self.is_fitted = True
        return X_scaled, y_scaled

    def transform(self, X, y):

        if not self.is_fitted:
            raise ValueError("Scaler has not been fitted yet. Call fit_transform first.")
        X_scaled = self.scaler_X.transform(X).astype('float32')
        y_scaled = self.scaler_Y.transform(y.reshape(-1, 1)).astype('float32')
        return X_scaled, y_scaled

    def inverse_transform_y(self, y_scaled):

        if not self.is_fitted:
            raise ValueError("Scaler has not been fitted yet.")
        return self.scaler_Y.inverse_transform(y_scaled)

class DataTransformer:
    def __init__(self, x, taus, y=None):
        self.x = x.astype('float32')
        self.y = y.astype('float32') if y is not None else None
        self.taus = taus.astype('float32')
        self._transform()

    def _transform(self):
        n_taus = len(self.taus)
        n_x = len(self.x)

        self.x_trans = np.repeat(self.x, n_taus, axis=0)
        self.tau_trans = np.tile(self.taus, n_x).reshape((-1, 1))

        if self.y is not None:
            self.y_trans = np.repeat(self.y, n_taus, axis=0).reshape((-1, 1))

    def __call__(self):
        return self.x_trans, self.y_trans, self.tau_trans

@tf.function
def train_step(model, inputs, output, tau, loss_func, optimizer):
    with tf.GradientTape() as tape:
        predicted = model(inputs, tau)
        loss = loss_func(output, predicted)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train_model(x_train, y_train, taus, n_hidden, n_hidden2, penalty,
                epochs=1000, learning_rate=0.01, verbose=False):
    """Train a single model configuration"""
    data_transformer = DataTransformer(x_train, taus, y_train)
    x_trans, y_trans, tau_trans = data_transformer()

    model = Mcqrnn(n_hidden, n_hidden2, penalty)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_func = TiltedAbsoluteLoss(tau_trans)

    for epoch in range(epochs):
        loss = train_step(model, x_trans, y_trans, tau_trans, loss_func, optimizer)

        if verbose and epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")

    return model, loss.numpy()

def evaluate_model_denormalized(model, x_val, y_val, taus, scaler):

    test_transformer = DataTransformer(x_val, taus, y_val)
    x_val_trans, y_val_trans, tau_val_trans = test_transformer()

    # Model prediction (normalized space)
    y_pred_normalized = model(x_val_trans, tau_val_trans)

    # Inverse normalization
    y_true_denorm = scaler.inverse_transform_y(y_val_trans)
    y_pred_denorm = scaler.inverse_transform_y(y_pred_normalized.numpy())

    # Convert to TensorFlow tensors
    y_true_denorm_tf = tf.constant(y_true_denorm, dtype=tf.float32)
    y_pred_denorm_tf = tf.constant(y_pred_denorm, dtype=tf.float32)

    # Compute overall loss (denormalized space)
    loss_func = TiltedAbsoluteLoss(tau_val_trans)
    total_loss = loss_func(y_true_denorm_tf, y_pred_denorm_tf).numpy()

    # Per-quantile losses
    quantile_losses = {}
    n_samples = len(x_val)

    for i, tau in enumerate(taus):
        start_idx = i * n_samples
        end_idx = (i + 1) * n_samples

        y_true_q = y_true_denorm_tf[start_idx:end_idx]
        y_pred_q = y_pred_denorm_tf[start_idx:end_idx]
        tau_q = tau_val_trans[start_idx:end_idx]

        loss_q = TiltedAbsoluteLoss(tau_q)(y_true_q, y_pred_q).numpy()
        quantile_losses[f'tau={tau:.2f}'] = float(loss_q)

    return total_loss, quantile_losses


def objective(trial, x_train, y_train, x_val, y_val, taus, scaler,
              epochs=1000, learning_rate=0.01):

    n_hidden = trial.suggest_int('n_hidden', 1, 10)
    n_hidden2 = trial.suggest_int('n_hidden2', 1, 10)
    penalty = trial.suggest_float('penalty', 1e-4, 1e-1)

    model, train_loss = train_model(
        x_train, y_train, taus, n_hidden, n_hidden2, penalty,
        epochs, learning_rate, verbose=False
    )

    val_loss, _ = evaluate_model_denormalized(model, x_val, y_val, taus, scaler)

    del model
    tf.keras.backend.clear_session()

    return val_loss


def optuna_search_hyperparameters(x_train, y_train, x_val, y_val, taus, scaler,
                                  n_trials=30, epochs=1000, learning_rate=0.01):

    optuna_start_time = time.time()

    # Create Optuna study
    study = optuna.create_study(
        direction='minimize',
        study_name='hyperparameter_optimization',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    print("Starting Optuna hyperparameter search...")
    study.optimize(
        lambda trial: objective(trial, x_train, y_train, x_val, y_val, taus, scaler,
                                epochs, learning_rate),
        n_trials=n_trials,
        show_progress_bar=True
    )

    optuna_end_time = time.time()
    optuna_duration = optuna_end_time - optuna_start_time

    print("\n" + "=" * 50)
    print("Best hyperparameters:")
    print(f"  n_hidden: {study.best_params['n_hidden']}")
    print(f"  n_hidden2: {study.best_params['n_hidden2']}")
    print(f"  penalty: {study.best_params['penalty']:.6f}")
    print(f"  Validation loss (denormalized): {study.best_value:.4f}")
    print(f"  Optuna runtime: {optuna_duration:.2f} seconds")
    print("=" * 50)

    print("\nRetraining model with best hyperparameters...")
    train_start_time = time.time()

    best_model, _ = train_model(
        x_train, y_train, taus,
        study.best_params['n_hidden'],
        study.best_params['n_hidden2'],
        study.best_params['penalty'],
        epochs, learning_rate, verbose=True
    )

    train_end_time = time.time()
    train_duration = train_end_time - train_start_time

    timing_info = {
        'optuna_duration_seconds': optuna_duration,
        'final_train_duration_seconds': train_duration,
        'optuna_duration_formatted': f"{optuna_duration / 60:.2f} minutes",
    }

    return study, best_model, timing_info

def save_results_to_json(study, timing_info, test_metrics, filename='optimization_results.json'):

    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'best_hyperparameters': {
            'n_hidden': int(study.best_params['n_hidden']),
            'n_hidden2': int(study.best_params['n_hidden2']),
            'penalty': float(study.best_params['penalty'])
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
    df = pd.read_csv('sereconstruction.csv')
    a = np.array(df.values[:, 0])#Choose: 0-sesub1; 1-sesub2;2-sesub3;3-sesub4
    a = np.expand_dims(a, axis=1)

    print("Create time series dataset...")
    X, y = split_data(a, horizon=1, window=96)

    n1 = X.shape[0]
    train_X, train_y = X[:round(0.8 * n1)], y[:round(0.8 * n1)]
    val_X, val_y = X[round(0.8 * n1):round(0.9 * n1)], y[round(0.8 * n1):round(0.9 * n1)]
    test_X, test_y = X[round(0.9 * n1):], y[round(0.9 * n1):]
    print(f"Train: {train_X.shape}, Val: {val_X.shape}, Test: {test_X.shape}")

    # Data normalization
    print("Data normalization...")
    scaler = DataScaler()
    x_train, y_train = scaler.fit_transform(train_X, train_y)
    x_val, y_val = scaler.transform(val_X, val_y)
    x_test, y_test = scaler.transform(test_X, test_y)

    EPOCHS = 1000
    LEARNING_RATE = 0.01
    N_TRIALS = 100
    taus = np.array([round(i, 2) for i in np.arange(0.05, 1.0, 0.05)])

    # ===============Using hyperparameters directly, without optuna (Option B)==================
    if MODE == 'B':
        RESULTS_JSON = 'tsmg1_results.json' #Change:sesub1:tsmg1_results;sesub2:tsmg2_results;sesub3:tsmg3_results;sesub4:tsmg4_results
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
        print(f"  (Source timestamp: {saved['timestamp']})")

        print("\nTraining final model with best parameters...")

        best_model, _ = train_model(
            x_train, y_train, taus,
            n_hidden, n_hidden2, penalty,
            epochs=EPOCHS, learning_rate=LEARNING_RATE, verbose=True)

        print("\n" + "=" * 50)
        print("Validation set evaluation (original scale):")
        print("=" * 50)
        val_total_loss, val_quantile_losses = evaluate_model_denormalized(
            best_model, x_val, y_val, taus, scaler)
        print(f"Overall loss: {val_total_loss:.4f}")
        for quantile, loss in val_quantile_losses.items():
            print(f"  {quantile}: {loss:.4f}")

        print("\n" + "=" * 50)
        print("Test set evaluation (original scale):")
        print("=" * 50)

        test_total_loss, test_quantile_losses = evaluate_model_denormalized(
            best_model, x_test, y_test, taus, scaler)

        print(f"Overall loss: {test_total_loss:.4f}")
        for quantile, loss in test_quantile_losses.items():
            print(f"  {quantile}: {loss:.4f}")
    #  =======================Optuna searching (Option C),uncomment to use================================
    elif MODE == 'C':
        print(f"\nStart hyperparameter optimization (trials={N_TRIALS}, epochs={EPOCHS})...")
        study, best_model, timing_info = optuna_search_hyperparameters(
            x_train, y_train, x_val, y_val, taus, scaler,
            n_trials=N_TRIALS,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE)

        print("\n" + "=" * 50)
        print("Validation set evaluation (denormalized):")
        print("=" * 50)
        val_total_loss, val_quantile_losses = evaluate_model_denormalized(
            best_model, x_val, y_val, taus, scaler)

        print(f"Overall loss: {val_total_loss:.4f}")
        for quantile, loss in val_quantile_losses.items():
            print(f"  {quantile}: {loss:.4f}")

        print("\n" + "=" * 50)
        print("Test set evaluation (denormalized):")
        print("=" * 50)
        test_start_time = time.time()
        test_total_loss, test_quantile_losses = evaluate_model_denormalized(
            best_model, x_test, y_test, taus, scaler)

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
            filename='tsmg1_results.json') #Obtain:sesub1:tsmg1_results;sesub2:tsmg2_results;sesub3:tsmg3_results;sesub4:tsmg4_results

    #  =======================Generating prediction results================================
    print("\nGenerating prediction results...")
    test_transformer = DataTransformer(x_test, taus, y_test)
    x_test_trans, y_test_trans, tau_test_trans = test_transformer()

    y_pred_normalized = best_model(x_test_trans, tau_test_trans)
    y_pred_denormalized = scaler.inverse_transform_y(y_pred_normalized.numpy())

    pred_df = pd.DataFrame(
        y_pred_denormalized.reshape(len(x_test), len(taus)),
        columns=[f'quantile_{tau:.2f}' for tau in taus]
    )
    pred_df.to_excel('tsmg1.xlsx', index=False)#Save: sesub1:tsmg1;sesub2:tsmg2;sesub3:tsmg3;sesub4:tsmg4
