##### Standard Libraries #####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("poster")

#%matplotlib inline
##### Other Libraries #####
from sklearn.cluster import KMeans
## Classification Algorithms ##
from sklearn.svm import SVC

## For building models ##
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

## For measuring performance ##
from sklearn import metrics
from sklearn.model_selection import cross_val_score

## To visualize decision tree ##
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

## Ignore warnings ##
import warnings
warnings.filterwarnings('ignore')


from itertools import cycle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import precision_recall_curve, roc_auc_score
from imblearn.over_sampling import RandomOverSampler, SMOTE

def confmatrix(y,y_kmeans, title):
    cm = metrics.confusion_matrix(y, y_kmeans,labels=[1, 2, 3, 4])
    df_cm = pd.DataFrame(cm, index=[i for i in "1234"], columns=[i for i in "1234"])
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    
    plt.figure(figsize = (10,7))
    plt.title(title)
    
    sns.set(font_scale=1.4) # For label size
    sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16}) # Font size
    plt.show()

def recall(y,y_kmeans):
    precision = dict()
    recall = dict()
    y = pd.get_dummies(y)
    y_kmeans = pd.get_dummies(y_kmeans)
    y_kmeans = y_kmeans.drop([0], axis=1)
    #y = label_binarize(y, classes=[1,2, 3, 4])
    #n_classes = y.shape[1]
    for i in range(1,5):
            precision[i], recall[i], _ = precision_recall_curve(y[i], y_kmeans[i])
            plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))

    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.show()

""""""
col_names = ['class', 'lymphatics', 'block of affere', 'bl. of lymph. c', 'bl. of lymph. s', 'by pass', 
 'extravasates', 'regeneration of', 'early uptake in', 'lym.nodes dimin', 'lym.nodes enlar', 
'changes in lym.', 'defect in node', 'changes in node', 'changes in stru', 'special forms', 
'dislocation of', 'exclusion of no', 'no. of nodes in']

df = pd.read_csv("lymphography.csv", names=col_names)



### Split the features and the target column.
X = df.drop('class', axis=1)
y = df['class']


  #y = label_binarize(y, classes=[1,2, 3, 4])
    #n_classes = y.shape[1]
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=2, n_init=9, random_state=1)
y_kmeans = kmeans.fit_predict(X)

centroids = kmeans.cluster_centers_
print("\nEtichette:")
print(kmeans.labels_)
### Split the dataset into training and test sets


### Print classification report for regular
print("----- Regular Training Set Used -----")
print("Classification report for :\n{}".format(metrics.classification_report(y, y_kmeans,labels=[1, 2, 3, 4])))
#print("Accuracy score:", clf_score)

### Plot confusion matrix
confmatrix(y,y_kmeans, "Confusion Matrix\n Kmeans - Regular Training Set")
#confmatrix(clf_pred_up, "Confusion Matrix\nRandom Forest - Upsampled Training Set")
#classifier = OneVsRestClassifier(kmeans)
#occurve(X,y,classifier)

print(y)
print(y_kmeans)

recall(y,y_kmeans)
