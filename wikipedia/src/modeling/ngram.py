import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


from sklearn.cross_validation import train_test_split, ShuffleSplit
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

import tensorflow as tf
import numpy as np
from sklearn import metrics, cross_validation

from sklearn.metrics import roc_curve, auc
from scipy import interp

from sklearn.calibration import calibration_curve
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score, roc_auc_score)
from collections import OrderedDict


from sklearn.metrics import (explained_variance_score, mean_absolute_error , mean_squared_error, median_absolute_error, r2_score )

from sklearn.preprocessing import label_binarize

from pprint import pprint

def get_labeled_comments(d, labels):
    """
    Get comments corresponding to rev_id labels
    """
    c = d[['rev_id', 'clean_diff']].drop_duplicates(subset = 'rev_id')
    c.index = c.rev_id
    c = c['clean_diff']
    c.name = 'x'
    data = pd.concat([c, labels], axis = 1)

    # shuffle
    m = data.shape[0]
    np.random.seed(seed=0)
    shuffled_indices = np.random.permutation(np.arange(m))
    return data.iloc[shuffled_indices].dropna()



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
    model = GridSearchCV(cv  = 3, estimator = alg, param_grid = param_grid, scoring = scoring, n_jobs=n_jobs, refit=True)
    model = model.fit(X,y)
    if verbose:
        print("\nBest parameters set found:")
        best_parameters, score, _ = max(model.grid_scores_, key=lambda x: x[1])
        print(best_parameters, score)
        print ("\n")
        print("Grid scores:")
        for params, mean_score, scores in model.grid_scores_:
            print("%0.5f (+/-%0.05f) for %r"
                  % (mean_score, scores.std() / 2, params))


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


def eval_adding_other_data(pipeline, a_train, a_test, b_train, metric_function, test_size = 0.2):

    k = 10
    step = int((b_train.shape[0]) / float(k))
    ms = range(step, b_train.shape[0]+1, step)
    metrics = []    
    for m in ms:
        train = pd.concat([a_train, b_train[:m]])
        model = pipeline.fit(train['x'].values, train['y'].values)
        metrics.append(metric_function(model, a_test))
    
    return ms, metrics

def plot_adding_other_data(ms, metrics):
    plt.plot(ms, metrics)
    plt.legend()
    plt.xlabel('number of examples from data set B added for training')
    plt.ylabel('metric on held out examples from data set A')





def epoch_and_batch_iter(X,y, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    
    for epoch in range(num_epochs):
       yield batch_iter(X, y, batch_size)


def batch_iter(X, y, batch_size):

    if not isinstance(y, np.ndarray):
        y = y.values

    if not isinstance(X, np.ndarray):
        X = X.values

    
    m = len(y)
    num_batches_per_epoch = int(float(m)/batch_size) + 1

    shuffle_indices = np.random.permutation(np.arange(m))
    shuffled_X = X[shuffle_indices]
    shuffled_y = y[shuffle_indices]
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, m)
        yield shuffled_X[start_index:end_index], shuffled_y[start_index:end_index]





def ED_CLF(X_train,
          y_train,
          X_test,
          y_test,
          learning_rate = 0.001,
          training_epochs = 60,
          batch_size = 200,
          display_step = 5,
         ):

    y_test_binary = np.zeros(y_test.shape)
    y_test_binary[list(range(y_test.shape[0])), y_test.argmax(axis = 1)] = 1

    y_train_binary = np.zeros(y_train.shape)
    y_train_binary[list(range(y_train.shape[0])), y_train.argmax(axis = 1)] = 1




    n_input = X_train.shape[1]
    n_classes = y_train.shape[1]

    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # Create model
    def LG(_X, _weights, _biases):
        return tf.matmul(_X, _weights['out']) + _biases['out']

    # Store layers weight & bias
    weights = {
        'out': tf.Variable(tf.random_normal([n_input, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    pred = LG(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
    cost += 5e-4 * tf.nn.l2_loss(weights['out']) # Regularization
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph
    sess = tf.Session()
    sess.run(init)

    batch = 0

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        m = 0
        batches = batch_iter(X_train.toarray(), y_train, batch_size)
        # Loop over all batches
        for batch_xs, batch_ys in batches:
            batch_m = len(batch_ys)
            m += batch_m
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}) * batch_m
            # Display logs per epoch step
            if batch % display_step == 0:
                print ("Batch:", '%04d' % (batch+1), "cost=", "{:.9f}".format(avg_cost/m))

                #correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                #print ("Train Accuracy:", accuracy.eval({x: X_train.toarray(), y: y_train}, session=sess))
                #print ("Test Accuracy:", accuracy.eval({x: X_test.toarray(), y: y_test}, session=sess)) 
                y_test_score = np.array(pred.eval({x: X_test.toarray()}, session=sess))
                y_train_score = np.array(pred.eval({x: X_train.toarray()}, session=sess))

                print ("Train ROC:", roc_auc_score(y_train_binary.ravel(), y_train_score.ravel()))
                print ("Test ROC:", roc_auc_score(y_test_binary.ravel(), y_test_score.ravel()))



            batch+=1

            
            

    print ("Optimization Finished!")
    y_score = np.array(pred.eval({x: X_test.toarray()}, session=sess))
    multi_class_roc_plotter(y_test, y_score)


def roc_curve_plotter(y_test, prob_pos):
    fpr, tpr, _ = roc_curve(y_test, prob_pos)

    plt.plot(fpr, tpr, label='ROC curve of class {0} (area = {1:0.2f})'
                                   ''.format(1, auc(fpr, tpr)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

def calibration_curve_plotter(y_test, prob_pos):

    brier = brier_score_loss(y_test, prob_pos, pos_label=1)

    fig = plt.figure(0, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    fraction_of_positives, mean_predicted_value = \
        calibration_curve(y_test, prob_pos, n_bins=10)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="Model: (%1.3f)" % brier)

    ax2.hist(prob_pos, range=(0, 1), bins=10, label='Model',
             histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()


def get_binary_classifier_metrics(prob_pos, y_test):
    """
    http://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html

    """

    ts = [np.percentile(prob_pos, p) for p in np.arange(0, 101, 1)]
    f1s = []
    ps = []
    rs = []

    for t in ts:
        y_pred_t = prob_pos >=t
        f1s.append(f1_score(y_test, y_pred_t))
        ps.append(precision_score(y_test, y_pred_t))
        rs.append(recall_score(y_test, y_pred_t))

    """
    plt.plot(ts, f1s, label = 'F1')
    plt.plot(ts, ps, label = 'precision')
    plt.plot(ts, rs, label = 'recal')
    plt.legend()
    """

    ix = np.argmax(f1s)


    scores = {
                'optimal F1': f1s[ix],
                'precision @ optimal F1': rs[ix],
                'recall @ optimal F1': rs[ix],
                'roc': roc_auc_score(y_test, prob_pos)
    }

    
    print('threshold @ optimal F1:', ts[ix])
    pprint({k: '%0.3f' % v for k,v in scores.items()})

        
    return scores


def eval_binary_classifier(model, data, calibration = True, roc = True):
    prob_pos = model.predict_proba(data['x'])[:,1]
    y_test = data['y']

    if roc:
        roc_curve_plotter(y_test, prob_pos)
    if calibration:
        calibration_curve_plotter(y_test, prob_pos)

    return get_binary_classifier_metrics(prob_pos, y_test)


def get_regression_metrics(y_test, y_pred):

    scores = [

                ('Explained Variance', explained_variance_score(y_test, y_pred)),
                ('R^2', r2_score(y_test, y_pred)),
                ('Mean squared error', mean_squared_error(y_test, y_pred)),
                ('Mean absolute error', mean_absolute_error(y_test, y_pred)),
                ('Median absolute error', median_absolute_error(y_test, y_pred)),
                ('Pearson', pearsonr(y_test, y_pred)[0]),
                ('Spearman', spearmanr(y_test, y_pred)[0]),
    ]
    scores = OrderedDict(scores)


    for k, v in scores.items():
        print ('%s: %0.3f' % (k,v))
  
    return scores

def residual_plotter(y_test, y_pred):
    sns.jointplot(y_test, y_pred, kind="reg")
    plt.xlabel('true score')
    plt.ylabel('predicted score')


def eval_regression(model, data, plot = False):
    """
    Compute Spearman correlation of model on data
    """
    y_pred = model.predict(data['x'])
    y_test = data['y'].values

    if plot:
        residual_plotter(y_test, y_pred)

    return get_regression_metrics(y_test, y_pred)


def multi_class_roc_plotter(y_test, y_score, plot = True):
    """
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    """

    if len(y_test.shape) == 1:
        y_test = label_binarize(y_test, sorted(list(set(y_test))))


    y = np.zeros(y_test.shape)
    y[list(range(y_test.shape[0])), y_test.argmax(axis = 1)] = 1
    y_test = y

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
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 linewidth=2)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 linewidth=2)

        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()
    return roc_auc


def eval_multiclass_classifier(model, data):
    y_score = model.predict_proba(data['x'])
    y_test = data['y'].values

    return multi_class_roc_plotter(y_test, y_score, plot = True)