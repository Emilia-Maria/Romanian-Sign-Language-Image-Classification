import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(model, dataset, class_names):
    """
    Plots a confusion matrix.

    Parameters:
    model (tf.keras.Model): The trained model.
    dataset (tf.data.Dataset): The dataset to predict and evaluate.
    class_names (list): List of class names.
    """
    y_pred = np.argmax(model.predict(dataset), axis=-1)
    y_true = np.concatenate([y for x, y in dataset], axis=0)
    y_true = np.argmax(y_true, axis=-1)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
