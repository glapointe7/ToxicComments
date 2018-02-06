import matplotlib.pyplot as plt
import itertools
import numpy as np

from sklearn.metrics import roc_curve, confusion_matrix, classification_report


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          color_map=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=color_map)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True class')
    plt.xlabel('Predicted class')

    
def plot_auc_roc(output_classes, y_test, predictions, auc_roc):
    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    left  = 0.125  # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.9      # the top of the subplots of the figure
    wspace = 0.6   # the amount of width reserved for blank space between subplots
    hspace = 0.5   # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)

    for i, label in enumerate(output_classes):
        fpr, tpr, threshold = roc_curve(y_test.values[:, i], predictions[:, i])

        row = int(i / 3)
        col = i % 3

        axes[row, col].set_title('ROC AUC of ' + label)
        axes[row, col].plot(fpr, tpr, 'b', label = 'ROC AUC = %0.2f' % auc_roc[i])
        axes[row, col].legend(loc = 'lower right')
        axes[row, col].plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        axes[row, col].set_xlabel('False Positive Rate')
        axes[row, col].set_ylabel('True Positive Rate')
        

def print_confusion_matrix_and_plot(y_test, predictions, output_classes):
    for i, label in enumerate(output_classes): 
        conf_matrix = confusion_matrix(y_test.values[:, i], predictions[:, i])

        plt.figure(figsize=(5,5))

        plot_confusion_matrix(conf_matrix, classes=['Non '+label, label], normalize=False, title='Confusion matrix')
        plt.show()

        print(classification_report(y_test.values[:, i], predictions[:, i], target_names=['Non '+label, label]))
        