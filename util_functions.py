
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

def draw_boxplots(data, x=None, y=None, vertical=True, figsize=(15, 25), hspace=0.4, wspace=0.2):
    plt.figure(figsize=figsize)

    if vertical:
        for i, column in enumerate(y):
            plt.subplot((len(y)+1)//2, 2, i+1)
            plt.subplots_adjust(hspace=hspace, wspace=wspace)
            plt.title('Boxplot for ' + column)
            sns.boxplot(data=data, x=x, y=column)

    else:
        for i, column in enumerate(x):
            plt.subplot((len(x)+1)//2, 2, i+1)
            plt.subplots_adjust(hspace=hspace, wspace=wspace)
            plt.title('Boxplot for ' + column)
            sns.boxplot(data=data, x=column, y=y)

    plt.show()


class build_classifier(object):
    def __init__(self, clf):
        self.clf = clf
    
    def TuneHParam(self, X_train, y_train, params, cv=3, scoring='accuracy'):
        grid = GridSearchCV(self.clf, params, cv=cv, scoring=scoring, n_jobs=-1)
        grid.fit(X_train, y_train)

        self.best_model = grid.best_estimator_
        self.best_params = grid.best_params_
        self.best_score = grid.best_score_

        print("Best Parameters for %s: %s" % (self.clf, self.best_params))
        print("Best Score %s: %.3f" % (self.clf, self.best_score))

