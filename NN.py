import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
if __name__ == "__main__":
    col_list=['ratings','polarity1','actual','expected']
    df = pd.read_csv('polar1.csv',usecols=col_list)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(999, inplace=True)
    array=df.values
    x=array[:,0:2]
    y1=array[:,2]
    X_train1, X_validation1, Y_train1, Y_validation1 = train_test_split(x, y1, test_size=0.20, random_state=1)
    print("------------------accuracy in our method-------------------")
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))
    # evaluate each model in turn
    our_results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train1, Y_train1, cv=kfold, scoring='accuracy')
        our_results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    print("-----------------accuracy in existing method-----------------")
    y2=array[:,3]
    X_train2, X_validation2, Y_train2, Y_validation2 = train_test_split(x, y2, test_size=0.20, random_state=1)
    models = []
    #models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    #models.append(('SVM', SVC(gamma='auto')))
    # evaluate each model in turn
    prev_results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train2, Y_train2, cv=kfold, scoring='accuracy')
        prev_results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    
print("-----------------other metrics in our method-----------------")
# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train1, Y_train1)
predictions = model.predict(X_validation1)

# Evaluate predictions
print(accuracy_score(Y_validation1, predictions))
print(confusion_matrix(Y_validation1, predictions))
print(classification_report(Y_validation1, predictions))


print("-----------------other metrics in existing method-----------------")
# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train2, Y_train2)
predictions = model.predict(X_validation2)

# Evaluate predictions
print(accuracy_score(Y_validation2, predictions))
print(confusion_matrix(Y_validation2, predictions))
print(classification_report(Y_validation2, predictions))





