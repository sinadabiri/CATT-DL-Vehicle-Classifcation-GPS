import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    cm_norm = np.round_(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]*100, decimals=1)
    print("Normalized confusion matrix")
    print(cm_norm)
    matplotlib.rcParams.update({'font.size': 12})
    plt.imshow(cm_norm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm_norm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{}({})'.format(cm_norm[i, j], cm[i, j]),
                 horizontalalignment="center", fontsize=12,
                 color="white" if cm_norm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.gcf().subplots_adjust(left=0.15)
    plt.ylabel('True Vehicle Class')
    plt.xlabel('Predicted Vehicle Class')

# Compute confusion matrix
#cnf_matrix = np.array([[7045, 1449], [ 237, 955]])  # for 2&3
#cnf_matrix = np.array([[ 6944, 2742], [ 8453, 27757]])  # for light& heavy

# the possible confusion matrix for dataset light&mid&heavy
cnf_matrix = np.array([[ 6400,  1515,   580],   [ 7401, 20237,  5937],   [  496,  1310,  2020]])  # for light&mid& heavy


np.set_printoptions(precision=2)
#class_names = ['class2', 'class3']
#class_names = ['light', 'heavy']
class_names = ['light', 'medium', 'heavy']
# Plot non-normalized confusion matrix

plt.figure(figsize=(7, 7))
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='light&heavy Dataset')

#plt.show()

#plt.savefig('confusion_2&3.png', dpi=600)
#plt.savefig('confusion_2&3.pdf', dpi=600)

#plt.savefig('confusion_light&heavy.png', dpi=600)
#plt.savefig('confusion_light&heavy.pdf', dpi=600)

plt.savefig('confusion_light&medium&heavy.png', dpi=600)
plt.savefig('confusion_light&medium&heavy.pdf', dpi=600)