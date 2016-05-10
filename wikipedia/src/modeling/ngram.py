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

import tensorflow as tf
import numpy as np
from sklearn import metrics, cross_validation

from sklearn.metrics import roc_curve, auc
from scipy import interp


def get_labeled_comments(d, labels):
    """
    Get comments corresponding to rev_id labels
    """
    c = d[['rev_id', 'clean_diff']].drop_duplicates(subset = 'rev_id')
    c.index = c.rev_id
    c = c['clean_diff']
    c.name = 'x'
    data = pd.concat([c, labels], axis = 1)
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
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost/m))

            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print ("Train Accuracy:", accuracy.eval({x: X_train.toarray(), y: y_train}, session=sess))
            print ("Test Accuracy:", accuracy.eval({x: X_test.toarray(), y: y_test}, session=sess)) 

            
            

    print ("Optimization Finished!")
    y_score = np.array(pred.eval({x: X_test.toarray()}, session=sess))
    multi_class_roc(y_test, y_score)


def multi_class_roc(y_test, y_score):
    """
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    """

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



    ##############################################################################
    # Plot ROC curves for the multiclass problem

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
