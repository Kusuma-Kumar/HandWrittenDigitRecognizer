import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def load_data():
    """
    Load and preprocess the MNIST dataset.
    Returns:
        X_train, X_val, y_train, y_val, test: Preprocessed training, validation, and test sets.
    """
    # Load datasets
    train = pd.read_csv('data/mnist_train.csv')
    print(train.shape)
    test = pd.read_csv('data/mnist_test.csv').iloc[:, 1:]
    print(test.shape)

    # Separate features and labels
    X = train.iloc[:, 1:].values  # All pixel values (convert to NumPy array)
    y = train.iloc[:, 0].values   # Labels (convert to NumPy array)

    # Normalize pixel values to range [0, 1]
    X = X / 255.0
    test = test.values / 255.0  # Convert test DataFrame to NumPy array and normalize

    # Reshape data for CNN (28x28x1 for grayscale images)
    X = X.reshape(-1, 28, 28, 1)
    test = test.reshape(-1, 28, 28, 1)

    # One-hot encode labels
    y = to_categorical(y, num_classes=10)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    return X_train, X_val, y_train, y_val, test


def build_model():
    """
    Build and compile a Convolutional Neural Network (CNN) model.
    Returns:
        model: A compiled CNN model.
    """
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """
    Train the CNN model.
    Args:
        model: The CNN model to train.
        X_train, y_train: Training data and labels.
        X_val, y_val: Validation data and labels.
    Returns:
        history: Training history object.
    """
    history = model.fit(
        X_train, y_train,
        epochs=10,  
        batch_size=100,
        validation_data=(X_val, y_val),
        verbose=2
    )
    return history

def plot_history(history):
    """
    Plot the training and validation accuracy and loss over epochs.
    Args:
        history: Training history object.
    """
    # Plot accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.close()

def plot_confusion_matrix(model, X_val, y_val):
    """
    Generate and display a confusion matrix for the model predictions.
    Args:
        model: Trained CNN model.
        X_val: Validation features.
        y_val: Validation labels (one-hot encoded).
    """
    try:
        # Get the true labels (convert one-hot encoding back to integer labels)
        true_labels = np.argmax(y_val, axis=1)

        # Predict the validation set
        predictions = model.predict(X_val)
        predicted_labels = np.argmax(predictions, axis=1)

        # Generate the confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=range(10))

        # Display the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
        disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
        plt.title("Confusion Matrix")

        # Save the confusion matrix as an image
        plt.savefig('confusion_matrix.png')
        print("Confusion matrix saved as 'confusion_matrix.png'")
        plt.show()
    except Exception as e:
        print(f"An error occurred while plotting the confusion matrix: {e}")



if __name__ == "__main__":
    # Load data
    X_train, X_val, y_train, y_val, test = load_data()

    # Build model
    model = build_model()

    # Train model
    history = train_model(model, X_train, y_train, X_val, y_val)

    # Plot training history
    plot_history(history)

    # Plot the confusion matrix for the validation set
    plot_confusion_matrix(model, X_val, y_val)