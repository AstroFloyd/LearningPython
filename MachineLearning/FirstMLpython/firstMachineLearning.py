#!/bin/env python3

""" firstMachineLearning.py:  my first machine-learning script.
    2021-04-02, MvdS: initial version.
"""


import colored_traceback
colored_traceback.add_hook()

import pandas as pd
from matplotlib import pyplot as plt
    


def main():
    """Main function of my first machine-learning script."""
    
    # check_libraries()
    
    # Load libraries:
    # from pandas import read_csv
    # from pandas.plotting import scatter_matrix
    # from matplotlib import pyplot
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import StratifiedKFold
    # from sklearn.metrics import classification_report
    # from sklearn.metrics import confusion_matrix
    # from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    
    # Load dataset:
    # url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    url = "iris.csv"  # Downloaded file
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pd.read_csv(url, names=names)
    
    print_data(dataset)
    
    plt.style.use('dark_background')        # Invert colours
    
    # plot_data(dataset)
    
    # Split-out validation dataset
    array = dataset.values
    X = array[:,0:4]
    y = array[:,4]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
    
    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))
    
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    
    
    # Compare algorithms:
    plt.boxplot(results, labels=names)
    plt.title('Algorithm Comparison')
    plt.show()
    
    exit(0)


def check_libraries():
    """Check the versions of my Python libraries.
    """
    
    # Python version:
    import sys
    print('Python: {}'.format(sys.version))
    
    # scipy:
    import scipy
    print('scipy: {}'.format(scipy.__version__))
    
    # numpy:
    import numpy
    print('numpy: {}'.format(numpy.__version__))
    
    # matplotlib:
    import matplotlib
    print('matplotlib: {}'.format(matplotlib.__version__))
    
    # pandas:
    import pandas
    print('pandas: {}'.format(pandas.__version__))
    
    # scikit-learn:
    import sklearn
    print('sklearn: {}'.format(sklearn.__version__))
    
    
    return


def print_data(dataset):
    """Print the data.

    Parameters:  
      dataset (pandas.DataFrame):  Data to print.
    
    """
    
    print(dataset)
    print(dataset.shape)
    print(dataset.head(20))
    print(dataset.describe())
    
    # class distribution:
    print(dataset.groupby('class').size())
    
    return


def plot_data(dataset):
    """Plot the raw data.

    Parameters:  
      dataset (pandas.DataFrame):  the dataset to plot.
    
    """
    
    # box and whisker plots:
    dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    plt.show()
    
    # histograms
    dataset.hist()
    plt.show()
    
    # scatter plot matrix
    pd.plotting.scatter_matrix(dataset)
    plt.show()
    
    return


if(__name__ == "__main__"): main()

