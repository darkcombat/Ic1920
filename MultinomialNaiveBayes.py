import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import pandas as pd
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
##### Standard Libraries #####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("poster")

#%matplotlib inline
##### Other Libraries #####

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import precision_recall_curve, roc_auc_score
from imblearn.over_sampling import RandomOverSampler, SMOTE

def confmatrix(y_pred, title):
    cm = metrics.confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, columns=np.unique(y_test), index = np.unique(y_test))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    
    plt.figure(figsize = (10,7))
    plt.title(title)
    
    sns.set(font_scale=1.4) # For label size
    sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16}) # Font size
    plt.show()
def roccurve(X,y,classifier):
    y = label_binarize(y, classes=[1,2, 3, 4])
    n_classes = y.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    y_score = classifier.fit(X_train, y_train).predict(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    lw = 2
    # Plot all ROC curves
 #   plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    recall(y_test,y_score,n_classes)

def recall(y_test,y_score,n_classes):
    precision = dict()
    recall = dict()
    for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
            plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))

    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.show()

def cvscore(clf,X,y):
    cv_scores = cross_val_score(clf, X, y, cv=10) * 100
    print('\ncv_scores mean:{}'.format(np.mean(cv_scores)))
    print('\ncv_score variance:{}'.format(np.var(cv_scores)))
    print('\ncv_score dev standard:{}'.format(np.std(cv_scores)))
    print('\n')
    data = {'variance': np.var(cv_scores), 'standard dev': np.std(cv_scores)}
    names = list(data.keys())
    values = list(data.values())
    fig,axs = plt.subplots(1, 1, figsize=(6, 3), sharey = True)
    axs.bar(names, values)
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



### Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
### Instantiate algorithm

clf = MultinomialNB()

classifier = OneVsRestClassifier(clf)

### Fit the model to the data
clf.fit(X_train, y_train)

### Predict on the test set
clf_pred = clf.predict(X_test)


### Get performance metrics
clf_score = metrics.accuracy_score(y_test, clf_pred) * 100

### Print classification report for regular
print("----- Regular Training Set Used -----")
print("Classification report for :\n{}".format(metrics.classification_report(y_test, clf_pred)))
print("Accuracy score:", clf_score)

### Plot confusion matrix
confmatrix(clf_pred, "Confusion Matrix\nMultinomialNB - Regular Training Set")
#confmatrix(clf_pred_up, "Confusion Matrix\nRandom Forest - Upsampled Training Set")

### Perform cross-validation then get the mean
clf_cv = np.mean(cross_val_score(clf, X, y, cv=10) * 100)
print("Cross-Validation score for MultinomialNB :", clf_cv)
#classifier = OneVsRestClassifier(clf)
#y_score = classifier.fit(X_train, y_train).decision_function(X_test)
#y_score = classifier.fit(X_train, y_train).decision_function(X_test)
cvscore(clf,X,y)
roccurve(X,y,classifier)

### Bilanciamento dataset

sm =  RandomOverSampler(random_state=1)

X, y = sm.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, random_state = 1)

#clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

### Print classification report for upsampled
print("\n----- Balanced Training Set Used -----")
print("Classification report for :\n{}".format(metrics.classification_report(y_test, prediction)))
#print("Accuracy score:", clf_score_up)
confmatrix(prediction, "Confusion Matrix\nMultinomialNB - Balanced Training Set")

cvscore(clf,X,y)
roccurve(X,y,classifier)


