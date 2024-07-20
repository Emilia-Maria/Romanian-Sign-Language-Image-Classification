import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(model, dataset, features=None, labels=None):
    """
    Plots a confusion matrix.

    Parameters:
    model (tf.keras.Model): The trained model.
    dataset (tf.data.Dataset): The dataset to predict and evaluate.
    features (np.ndarray): The features to predict. Default is None.
    labels (np.ndarray): The true labels. Default is None.
    """
    if features is not None and labels is not None:
        y_pred = np.argmax(model.predict(features), axis=-1)
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            y_true = np.argmax(labels, axis=-1)
        else:
            y_true = labels
    elif dataset is not None:
        y_pred = np.argmax(model.predict(dataset), axis=-1)
        y_true = np.concatenate([y for x, y in dataset], axis=0)
        y_true = np.argmax(y_true, axis=-1)
    else:
        raise ValueError("Either dataset or both features and labels must be provided.")
    
    target_names = dataset.class_names

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()