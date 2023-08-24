#standard ds libraries
import pandas as pd
import numpy as np

# stats testing 
from scipy import stats

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split # was in my prepare.py 
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


######################################## 

def get_metrics(model,xtrain,ytrain,xtest,ytest) -> str:
    
    labels = sorted(y_train.unique()) # orders the array of the target variable 

    # OUTPUTS AN ARRAY OF PREDICTIONS
    preds = model.predict(xtest)
    print("Accuracy Score:", model.score(xtest,ytest))
    print()
    print('Confusion Matrix:')
    conf = confusion_matrix(ytest,preds)
    TP = conf[0, 0]
    TN = conf[1, 1]
    FP = conf[0, 1]
    FN = conf[1, 0]
    conf = pd.DataFrame(conf,
            index=[str(label) + '_actual'for label in labels],
            columns=[str(label) + '_predict'for label in labels])
    print(conf)
    print('')
    print(f'True Positive -> {TP}')
    print(f'True Negative -> {TN}')
    print(f'False Positive -> {FP}')
    print(f'False Negative -> {FN}')
    print()
    print("Classification Report:")
    print(classification_report(ytest, preds))

################################################################

def compute_metrics(TN,FP,FN,TP):
    all_ = (TP + TN + FP + FN)

    accuracy = (TP + TN) / all_

    TPR = recall = TP / (TP + FN)
    FPR = FP / (FP + TN)

    TNR = TN / (FP + TN)
    FNR = FN / (FN + TP)

    precision =  TP / (TP + FP)
    f1 =  2 * ((precision * recall) / ( precision + recall))

    support_pos = TP + FN
    support_neg = FP + TN

    print(f"Accuracy: {accuracy}\n")
    print(f"True Positive Rate/Sensitivity/Recall/Power: {TPR}")
    print(f"False Positive Rate/False Alarm Ratio/Fall-out: {FPR}")
    print(f"True Negative Rate/Specificity/Selectivity: {TNR}")
    print(f"False Negative Rate/Miss Rate: {FNR}\n")
    print(f"Precision/PPV: {precision}")
    print(f"F1 Score: {f1}\n")
    print(f"Support (0): {support_pos}")
    print(f"Support (1): {support_neg}")

################################################################

def dis_tree(train_X, validate_X, train_y, validate_y):

    #get decision tree accuracy on train and validate data

    # create classifier object
    clf = DecisionTreeClassifier(max_depth=3, random_state=123)

    #fit model on training data
    clf = clf.fit(train_X, train_y)

    # print result
    print(f"Accuracy of Decision Tree on train data is {clf.score(train_X, train_y)}")
    print(f"Accuracy of Decision Tree on validate data is {clf.score(validate_X, validate_y)}")

################################################################

def rand_forest(train_X, validate_X, train_y, validate_y):
    '''get random forest accuracy on train and validate data'''

    # create model object and fit it to training data
    rf = RandomForestClassifier(max_depth=3, random_state=123)
    rf.fit(train_X,train_y)

    # print result
    print(f"Accuracy of Random Forest on train is {rf.score(train_X, train_y)}")
    print(f"Accuracy of Random Forest on validate is {rf.score(validate_X, validate_y)}")

################################################################

def log_reg(train_X, validate_X, train_y, validate_y):
    '''get logistic regression accuracy on train and validate data'''

    # create model object and fit it to the training data
    logit = LogisticRegression(C=1, random_state=123)
    logit.fit(train_X, train_y)

    # print result
    print(f"Accuracy of Logistic Regression on train is {logit.score(train_X, train_y)}")
    print(f"Accuracy of Logistic Regression on validate is {logit.score(validate_X, validate_y)}")

################################################################

def get_knn(train_X, validate_X, train_y, validate_y):
    '''get KNN accuracy on train and validate data'''

    # create model object and fit it to the training data
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_X, train_y)

    # print results
    print(f"Accuracy of Logistic Regression on train is {knn.score(train_X, train_y)}")
    print(f"Accuracy of Logistic Regression on validate is {knn.score(validate_X, validate_y)}")