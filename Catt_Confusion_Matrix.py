import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm_norm = np.round_(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    print("Normalized confusion matrix")
    print(cm_norm)

    plt.imshow(cm_norm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm_norm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{}({})'.format(cm[i, j], cm_norm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm_norm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.gcf().subplots_adjust(left=0.15)
    plt.ylabel('True Vehicle Class')
    plt.xlabel('Predicted Vehicle Class')

# Compute confusion matrix
#cnf_matrix = np.array([[7045, 1449], [ 237, 955]])  # for 2&3
#cnf_matrix = np.array([[ 6944, 2742], [ 8453, 27757]])  # for light& heavy
#cnf_matrix = np.array([[ 6817, 1153, 524],  [ 9183, 17767, 6626],  [655,  1124,  2047]])  # for light&mid& heavy
cnf_matrix = np.array([[ 6400,  1515,   580],   [ 7401, 20237,  5937],   [  496,  1310,  2020]])  # for light&mid& heavy
#cnf_matrix = np.array([[ 6497,  1453,   545],    [ 7678, 18940,  6957],    [  488,  1202,  2136]])  # for light&mid& heavy
#cnf_matrix = np.array([[ 6961,  1298,   235],     [ 9375, 20287,  3914],     [  680,  1491,  1655]])  # for light&mid& heavy
#cnf_matrix = np.array([[ 7018,  1064,   413],     [ 9531, 19058,  4986],     [  658 , 1333 , 1835]])  # for light&mid& heavy


np.set_printoptions(precision=2)
#class_names = ['class2', 'class3']
class_names = ['light', 'heavy']
class_names = ['light', 'medium', 'heavy']
# Plot non-normalized confusion matrix


plt.figure(figsize=(7, 7))
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='light&medium&heavy Dataset')

#plt.show()

#plt.savefig('confusion_2&3.png')
#plt.savefig('confusion_2&3.pdf')

#plt.savefig('confusion_light&heavy.png')
#plt.savefig('confusion_light&heavy.pdf')

plt.savefig('confusion_light&medium&heavy.png')
plt.savefig('confusion_light&medium&heavy.pdf')