import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.src.ops import dtype
from pyarrow.dataset import dataset
from streamlit import dataframe
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
from file_processing import load_from_json, save_to_json


def extract_features_target_columns(df):
    """Extract features and target columns from dataframe."""
    return df.iloc[:, :-1], df.iloc[:, -1]


def train_val_test_split(df):
    """Split the data to training, validation and test sets."""
    train_size = int(len(df) * 0.8)
    val_size = int(len(df) * 0.1)

    train_df = df[:train_size]
    val_df = df[train_size: train_size + val_size]
    test_df = df[train_size + val_size:]

    features, target = extract_features_target_columns(df)
    train_df = shuffle_data(train_df)
    val_df = shuffle_data(val_df)
    test_df = shuffle_data(test_df)

    X_train, y_train = train_df[features.columns], train_df[target.name]
    X_val, y_val = val_df[features.columns], val_df[target.name]
    X_test, y_test = test_df[features.columns], test_df[target.name]

    return X_train, y_train, X_val, y_val, X_test, y_test


def create_model(X_train, y_train, X_val, y_val, model_name=None):
    """Create Multi-Layer Perceptron neural network."""
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=30,
        restore_best_weights=True
    )

    model = Sequential([
        Input(shape=(132,)),
        Dense(132, activation="relu"),
        Dropout(0.2),
        Dense(66, activation="relu"),
        Dropout(0.2),
        Dense(42, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss=SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=300,
        batch_size=33,
        callbacks=[early_stopping],
        verbose=1
    )

    plot_train_val_loss_functions(history)
    model.save(model_name)

    return model


def print_metrics(test_data, predictions):
    """Print classification metrics."""
    print(classification_report(test_data, predictions))


def plot_train_val_loss_functions(history):
    """Plot training and validation loss functions."""
    lowest_point = min(history.history["val_loss"])
    plt.plot(history.history["loss"][10:], label="Training Error")
    plt.plot(history.history["val_loss"][10:], label="Validation Error")
    plt.plot(history.history["val_loss"].index(lowest_point),
             lowest_point,
             label="Lowest Val Point",
             color="red",
             marker="o", ms=5)
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("Training History")
    plt.legend()
    plt.grid()
    plt.show()
    print(f"Lowest Point Value from Validation Loss: {lowest_point}")


def plot_actual_predicted_values(test_data, predictions):
    """Plot actual vs. predicted values from the model."""
    plt.figure(figsize=(12, 6))
    plt.plot(test_data, label='Actual', linewidth=5)
    plt.plot(predictions, label='Predicted')
    plt.yticks(range(0, 42))
    plt.title("Actual vs Predicted Disease")
    plt.xlabel("Patient")
    plt.ylabel("Disease")
    plt.legend()
    plt.grid()
    plt.show()


def compare_actual_predicted_values(test_data, predictions):
    """Compare actual vs. predicted values from the model."""
    predicted_true = 0
    predicted_false = 0

    for item in range(len(test_data)):
        if test_data[item] != predictions[item]:
            predicted_false += 1
        else:
            predicted_true += 1
    print(f"Total Values in Data: {len(test_data)}")
    print(f"Model Predicted Correctly: {predicted_true}")
    print(f"Model Predicted Incorrectly: {predicted_false}")


def create_comparison_dataframe(test_data, predictions):
    """Create comparison matrix for actual vs. predicted values."""
    diseases = load_from_json("disease.json")
    test_data_list = [diseases[str(item)] for item in test_data]
    predictions_list = [diseases[str(item)] for item in predictions]

    return pd.DataFrame({
        "Actual Values" : test_data_list,
        "Predicted Values" : predictions_list
    })


def shuffle_data(df):
    """Shuffle data"""
    return df.sample(frac=1).reset_index(drop=True)


def make_prediction(model, X_test, y_test):
    """Predict the outcome from the model"""
    prediction_classes = model.predict(X_test)
    predictions = prediction_classes.argmax(axis=1)

    print_metrics(y_test, predictions)
    y_test = np.array(y_test, dtype=np.int64).flatten()

    plot_actual_predicted_values(y_test, predictions)
    compare_actual_predicted_values(y_test, predictions)

    comparison_matrix = create_comparison_dataframe(y_test, predictions)
    print(comparison_matrix.head())
