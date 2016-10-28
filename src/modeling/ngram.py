import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy import interp
from baselines import *
from sklearn.metrics import brier_score_loss
from sklearn.grid_search import RandomizedSearchCV
from sklearn.grid_search import ParameterSampler



class RandomizedSearchCVWithDependencies(RandomizedSearchCV):
    
    def __init__(self, estimator, param_distributions, n_iter=10, scoring=None,
                 fit_params=None, n_jobs=1, iid=True, refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs', random_state=0,
                 error_score='raise', dependencies=[]):

        self.dependencies = dependencies
        
        super(RandomizedSearchCVWithDependencies, self).__init__(
            estimator=estimator, param_distributions=param_distributions, 
            n_iter=n_iter, scoring=scoring, fit_params=fit_params,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, random_state=random_state, 
            error_score=error_score,
            )
        
        
    def fit(self, X, y=None):
        
        sampled_params = ParameterSampler(self.param_distributions,
                                          self.n_iter,
                                          random_state=self.random_state)
        
        sampled_params = list(sampled_params)
        
        for p in sampled_params:
            for a,b in self.dependencies:
                p[b] = p[a]
        
        
        return self._fit(X, y, sampled_params)



def tune (X_train,
           y_train,
           X_dev,
           y_dev,
           alg,
           param_grid,
           n_iter,
           scoring,
           n_jobs = 6,
           verbose = True,
           dependencies=[]):
    """
    Determine the best model via cross validation.
    """
    X = np.concatenate([X_train, X_dev])
    y = np.concatenate([y_train, y_dev])

    # train on train, eval on dev
    m_train = X_train.shape[0]
    m_dev = X_dev.shape[0]
    split = [[np.arange(m_train), np.arange(m_train, m_train + m_dev)]]

    model = RandomizedSearchCVWithDependencies( cv = split,
                                estimator = alg,
                                param_distributions = param_grid, 
                                n_iter = n_iter,
                                scoring = scoring, 
                                n_jobs=n_jobs,
                                refit=False,
                                dependencies=dependencies)
    model.fit(X, y)
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


def roc_scorer(clf, X, true):
    pred = clf.predict_proba(X)
    return multi_class_roc_auc(true, pred)

def spearman_scorer(clf, X, true):
    pred = clf.predict_proba(X)
    return multi_class_spearman(true, pred) 


def eval_multiclass_classifier(model, X, true, plot = False, verbose = True):
    true_oh = one_hot(true)
    pred = model.predict_proba(X)

    if plot:
        multi_class_roc_plotter(true_oh, pred, plot = plot)
    else:
        roc = multi_class_roc_auc(true_oh, pred, average = 'macro')
        spearman = multi_class_spearman(true, pred)

        if verbose:
            print('\tROC: %.3f' % roc)
            print('\tSpearman: %.3f' % spearman)
        return roc, spearman


def test_cross(model, data, xtype):

    nss = [ ('user', ['user']),
           ('article', ['article']),
           ('both', ['user', 'article']),
           ]

    samples = [
            ('random', ['random']),
            ('blocked', ['blocked']),
            ('both', ['random', 'blocked']),
           ]

    df = pd.DataFrame({'random':[0, 0,0], 'blocked': [0, 0,0], 'both': [0, 0,0]})
    df.index = ['user', 'article', 'both']
    df = df[['random', 'blocked', 'both']]

    roc_df = df.copy()
    spearman_df = df.copy()

    for nsn, ns in nss:
        for sn, s in samples:
            X, y = assemble_data(data, xtype, 'empirical_dist', nss = ns, samples = s, splits = ['test'])
            roc, spearman =  eval_multiclass_classifier(model, X, y, verbose = False)  
            roc_df[sn].loc[nsn] = roc
            spearman_df[sn].loc[nsn] = spearman

    return roc_df, spearman_df


def test_custom_cross(optimal_pipeline, data, xtype, ytype, train_params, test_params):
    for train_param in train_params:

        X, y = assemble_data(data, xtype, ytype, nss = train_param['nss'], samples = train_param['samples'], splits = ['train'])
        model = optimal_pipeline.fit(X, y)

        for test_param in test_params:

            X, y = assemble_data(data, xtype, 'empirical_dist', nss = test_param['nss'], samples = test_param['samples'], splits = ['test'])

            print('\nTrain: ', train_param)
            print('Test: ', test_param, '\n')
            eval_multiclass_classifier(model, X, y)



def two_class_roc_plotter(y_test, y_score):
    plt.figure()
    y_test = one_hot(y_test)
    fpr, tpr, _ = roc_curve(y_test[:, 1], y_score[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label='area = %.3f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")


def two_class_fpr_tpr_plotter(y_test, y_score):
    plt.figure()
    y_test = one_hot(y_test)
    fpr, tpr, thresholds = roc_curve(y_test[:, 1], y_score[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.plot(thresholds, fpr, label='FPR')
    plt.plot(thresholds, tpr, label='TPR')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Threshold')
    plt.ylabel('FPR/TPR')
    plt.legend(loc="lower right")

def two_class_precision_recall_plotter(y_test, y_score):
    plt.figure()
    y_test = one_hot(y_test)
    precision, recall, thresholds = precision_recall_curve(y_test[:, 1], y_score[:, 1])
    plt.plot(thresholds, precision[1:], label='precision')
    plt.plot(thresholds, recall[1:], label='recall')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Decision threshold')
    plt.xlabel('Precision/Recall')
    plt.legend(loc="lower right")

def f1(p, r):
    return 2 * ((p*r)/(p+r))

def two_class_combo_plotter(y_test, y_score):
    plt.figure()
    y_test = one_hot(y_test)
    fpr, tpr, thresholds = roc_curve(y_test[:, 1], y_score[:, 1])

    for i,t in enumerate(thresholds):
        if t < 0.425:
            print(t, fpr[i], tpr[i])
            break
    roc_auc = auc(fpr, tpr)

    plt.plot(thresholds, fpr, label='FPR')
    plt.plot(thresholds, tpr, label='TPR / Recall')


    precision, recall, thresholds = precision_recall_curve(y_test[:, 1], y_score[:, 1])

    for i,t in enumerate(thresholds):
        if t > 0.425:
            print(t, precision[i], recall[i])
            break

    plt.plot(thresholds, precision[1:], label='Precision')

    plt.plot(thresholds, [f1(precision[i], recall[i]) for i in range(len(thresholds))], label='F1')



    plt.plot([0.408] * 100, np.arange(0, 1, 0.01), 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Threshold')
    plt.legend(loc="lower right")

def multi_class_roc_plotter(y_test, y_score, plot = True):
    """
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    """
    
    y_test = one_hot(y_test)

    n_classes = y_test.shape[1]
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    # Compute macro-average ROC curve and ROC area
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

    if plot:
        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.3f})'
                       ''.format(roc_auc["micro"]),
                 linewidth=2)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.3f})'
                       ''.format(roc_auc["macro"]),
                 linewidth=2)

        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.3f})'
                                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.show()
    return roc_auc


def calibration_curve_plotter(y_test, prob_pos, n_bins = 10):

    brier = brier_score_loss(y_test, prob_pos, pos_label=1)

    fig = plt.figure(0, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    
    df = pd.DataFrame({'true': y_test})
    bins = np.linspace(0., 1. , n_bins + 1) 
    binids = np.digitize(prob_pos, bins) - 1
    df['Bin center'] = bins[binids] + .5/n_bins
    df[''] = 'Model calibration: (%1.5f)' % brier
    o = bins + .5/n_bins

    df2 = pd.DataFrame({'true': o, 'Bin center': o})
    df2[''] = 'Perfect calibration'

    df = pd.concat([df, df2])
    
    sns.pointplot(x='Bin center', y = 'true', data = df, order = o, hue = '', ax = ax1)


    ax2.hist(prob_pos, range=(0, 1), bins=10, label='Model',
             histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    #ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots')

    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Count")

    plt.tight_layout()



def eval_adding_other_data(pipeline, ax_train, ay_train, ax_test, ay_test, bx_train, by_train, metric_function):

     
    k = 4
    step = int((bx_train.shape[0]) / float(k))

    if ax_train is None or ax_train.empty:
        ms = range(step, bx_train.shape[0]+1, step)
    else:
        ms = range(0,bx_train.shape[0]+1, step)
    
    metrics = []    
    for m in ms:
        x_train = pd.concat([ax_train, bx_train[:m]])
        y_train = pd.concat([ay_train, by_train[:m]])

        model = pipeline.fit(x_train, y_train)
        metrics.append(metric_function(model, ax_test, ay_test))
    
    return ms, metrics

def plot_adding_other_data(ms, metrics):
    plt.plot(ms, metrics)
    plt.legend()
    plt.xlabel('number of examples from data set B added for training')
    plt.ylabel('metric on held out examples from data set A')

