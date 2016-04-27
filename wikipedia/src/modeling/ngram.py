import numpy as np
import pandas as pd
from scipy.stats import pearsonr


from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import joblib

import matplotlib.pyplot as plt
import seaborn as sns


def get_labeled_comments(d, labels):
    """
    Get comments corresponding to rev_id labels
    """
    c = d[['rev_id', 'clean_diff']].drop_duplicates(subset = 'rev_id')
    c.index = c.rev_id
    c = c['clean_diff']
    data = pd.DataFrame()
    data['x'] = c
    data['y'] = labels
    return data.dropna()

def split(data, test_size = 0.2,):
    """
    Split data into train and test
    """
    m = data.shape[0]
    np.random.seed(seed=0)
    shuffled_indices = np.random.permutation(np.arange(m))
    s = int(m*test_size)
    return (data.iloc[shuffled_indices[s:]], data.iloc[shuffled_indices[:s]])

def tune (X, y, alg, param_grid, scoring, n_jobs = 1, dev_size = 0.2, verbose = False):
    """
    Determine the best model via cross validation. This should be run on training data.
    """ 
    # generate train + dev set
    m = len(X)
    np.random.seed(seed=0)
    shuffled_indices = np.random.permutation(np.arange(m))
    s = int(m*dev_size)
    split = [(shuffled_indices[:s], shuffled_indices[s:])]
    
    # perform gridsearch
    model = GridSearchCV(cv  = split, estimator = alg, param_grid = param_grid, scoring = scoring, n_jobs=n_jobs, refit=True)
    model.fit(X,y)
    
    if verbose:
        print("\nBest parameters set found:")
        best_parameters, score, _ = max(model.grid_scores_, key=lambda x: x[1])
        print(best_parameters, score)
        print ("\n")
        print("Grid scores:")
        for params, mean_score, scores in model.grid_scores_:
            print("%0.5f (+/-%0.05f) for %r"
                  % (mean_score, scores.std() / 2, params))
    return model


def evaluate(model, data, metric, plot = False):
    """
    Compute Spearman correlation of model on data
    """
    pred = model.predict(data['x'])
    result = metric(data['y'],pred)
    if plot:
        sns.jointplot(data['y'].values, pred, kind="reg")
        plt.xlabel('true score')
        plt.ylabel('predicted score')
    return result

def tune_and_eval(data, plot = False):
    train, test = split(data)
    model = tune (train['x'], train['y'], reg_pipeline, param_grid, 'mean_squared_error', n_jobs=8, verbose=True)
    metrics =  {'train': evaluate(model, train, plot = plot), 'test': evaluate(model, test)}
    return model, metrics

def eval_blended_training(pipeline, blocked, random, metric, test_size = 0.2):
    
    blocked_train, blocked_test = split(blocked, test_size = test_size)
    random_train, random_test = split(random, test_size = test_size)
    
    m = min(blocked_train.shape[0], random_train.shape[0])
    
    blocked_metric = []
    random_metric = []
    alphas = np.arange(0,1.01, 0.2)
    
    for alpha in alphas:
        train = pd.concat([blocked_train[:int(alpha*m)], random_train[:int((1-alpha)*m)]])        
        model = pipeline.fit(train['x'].values, train['y'].values)
        blocked_metric.append(evaluate(model, blocked_test, metric))
        random_metric.append(evaluate(model, random_test, metric))
        
    return  alphas,  blocked_metric, random_metric

def plot_blended_training(alphas, b, r):
    plt.plot(alphas, b, label = 'Eval on Blocked')
    plt.plot(alphas, r, label = 'Eval on Random')
    plt.legend()
    plt.xlabel('Fraction of Training Data coming from from Blocked')
    plt.ylabel('metric')


def eval_adding_other_data(pipeline, a, b, metric, test_size = 0.2):

    train, test = split(a, test_size = test_size)
    k = 10
    step = int((b.shape[0]) / float(k))
    ms = range(0, b.shape[0], step)
    metrics = []    
    for m in ms:
        train = pd.concat([train, b[:m]])
        model = pipeline.fit(train['x'].values, train['y'].values)
        metrics.append(evaluate(model, test, metric))
    
    return ms, metrics

def plot_adding_other_data(ms, metrics):
    plt.plot(ms, metrics)
    plt.legend()
    plt.xlabel('number of example from B added for training')
    plt.ylabel('metric on held out A data')



