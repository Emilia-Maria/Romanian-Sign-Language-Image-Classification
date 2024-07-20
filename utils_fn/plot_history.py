import matplotlib.pyplot as plt

def plot_history(history):
    """
    Plots the training and validation accuracy and loss from a Keras history object.

    Parameters:
    history (dict): Dictionary containing the training history
    """
    accuracy = history["accuracy"]
    val_accuracy = history["val_accuracy"]
    loss = history["loss"]
    val_loss = history["val_loss"]
    epochs = range(1, len(accuracy) + 1)

    # plot accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy, "bo", label="Train accuracy")
    plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
    plt.title("Train / validation accuracy")
    plt.legend()

    # plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "bo", label="Train loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Train / validation loss")
    plt.legend()

    plt.tight_layout()
    plt.show()