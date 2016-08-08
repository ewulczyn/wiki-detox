import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr
from scipy.stats import entropy as kl
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error
from math import sqrt
from sklearn.cross_validation import ShuffleSplit
from sklearn.preprocessing import label_binarize
import os

import multiprocessing as mp

def get_baseline(labels, k, agg_function, eval_function, pairs, n_jobs = 8):

    """      
    Say we have 2k human scores for each comment. For each comment, for
    all (i,j) <= (k+1),     spit human scores into 2 non-overlapping sets of I and J
    of size i and j respectively. Compute     correlation between the mean of scores
    in set I and set J for all comments.     Intuitively,     correlation i,j tells
    us how good i humans are are predicting the labels of another group of j
    humans.

    As i increases, we expect to get better predictions and as j increases, we
    expect to get more predictable labels.

    To figure out how many humans we need to label each question, we should
    examine the diagonal of the matrix (where i=j) and pick a value of i=j where
    there are diminishing returns to going further down the diagonal.

    To figure out how hard we should try at building a machine learning model
    for labels that we got from aggregating j_0 human labels we can check the
    correlations for different values of i. We can interpret correlation (i,
    j_0) as how good an "ensemble" of i humans is at predicting the labels.

    So a model that can achieve correlation (1, j_0) is as good as a single
    human. Also, we would expect that a model should not beat correlation (j_0,
    j_0). If it does, then it overfit to the group and you should increase j0.

    """
    

    labels = labels.dropna()
    groups = labels.groupby(labels.index)
    groups = [e[1] for e in groups if e[1].shape[0]>=k]
        
    args = [(groups, i, j, agg_function, eval_function) for i, j in pairs]

    p = mp.Pool(min(n_jobs, len(args)))
    res = p.map(baseline_helper, args)
    p.close()
    p.join()
    #res = [baseline_helper(arg) for arg in args]
         
    return dict(zip(pairs, res))

def baseline_helper(args):
    np.random.seed()

    groups, i, j, agg_function, eval_function = args

    dis = []
    djs = []
    for g in groups:
        if g.shape[0] >= i+j:
            g = g.iloc[np.random.permutation(len(g))]
            dis.append(g[0:i])
            djs.append(g[i:(i+j)])
        else:
            print(i,j, g, "WARNING: Comment had less than k labels")

    di =  pd.concat(dis)
    dj = pd.concat(djs)

    scores_i = agg_function(di).values
    scores_j = agg_function(dj).values

    return eval_function(scores_i,scores_j)



# Aggregation Functions

def average(l):
    """
    Average all labels with the same rev_id
    """
    s = l.groupby(l.index).mean()
    s.name = 'y'
    return s

def remove_na(l):
    l['na'] = l['na'].fillna(value = False)
    s = l.groupby(l.index).filter(lambda x: np.mean(x['na']) < 0.5)
    return s

def plurality(l):
    """
    Take the most common label from all labels with the same rev_id
    """
    s = l.groupby(l.index).apply(lambda x:x.value_counts().index[0])
    s.name = 'y'
    return s

def empirical_dist(l, w = 0.0, index = None):

    """
    Compute empirical distribution over all classes
    using all labels with the same rev_id
    """
    if not index:
        index = sorted(list(set(l.dropna().values)))

    data = {}
    for k, g in l.groupby(l.index):
        data[k] = g.value_counts().reindex(index).fillna(0) + w

    labels = pd.DataFrame(data).T
    labels = labels.fillna(0)
    labels = labels.div(labels.sum(axis=1), axis=0)
    return labels


# Regression Evaluation Metrics

def pearson(x,y):
    return pearsonr(x,y)[0]

def spearman(x,y):
    return spearmanr(x,y)[0]

def rmse(x,y):
    return sqrt(mean_squared_error(x, y))

# Binary Classification Evaluation Metrics


def binary_roc_auc(true, pred):
    true = (true > 0.5).astype(float)
    return roc_auc_score(true, pred)

def binary_optimal_f1(true, pred, step = 1):
    binary_true = (true > 0.5).astype(float)
    ts = [np.percentile(pred, p) for p in np.arange(0, 101, step)]
    f1s = []
    for t in ts:
        y_pred_t = pred >= t
        f1 = f1_score(binary_true, y_pred_t)
        # Note F1 should have a parabolic shape, so no need to continue when the score starts falling
        if len(f1s) > 0 and f1 < f1s[-1] :
            return f1s[-1]
        else:
            f1s.append(f1)

    return f1s[-1]

# Multi-Class Classification Evaluation Metrics

def one_hot(y):
    m = y.shape[0]
    
    if len(y.shape) == 1:
        n = len(set(y.ravel()))
        idxs = y.astype(int)
    else:
        idxs = y.argmax(axis = 1)
        n = y.shape[1]

    y_oh = np.zeros((m, n))
    y_oh[list(range(m)), idxs] = 1
    return y_oh

def expectation(y):
    classes = np.arange(y.shape[1])
    return y.dot(classes)

def multi_class_roc_auc(true, pred, average = 'macro'):
    true = one_hot(true)
    #print(true)
    return roc_auc_score(true, pred, average = average)

def multi_class_spearman(true, pred):
    return spearman(expectation(true), expectation(pred))

def multi_class_pearson(true, pred):
    return pearson(expectation(true), expectation(pred))

def cross_entropy(x, y):
    logy =  np.log(y)
    logy[np.isinf(logy)] = 0
    return - np.multiply(x,logy).sum(axis=1).mean()    

def kl_divergence(x, y):
    return kl(x.T, y.T).mean()

def tidy_labels(d):
    classes = ['not_attack', 'other', 'quoting', 'recipient', 'third_party']
    for e in classes:
        d[e] = d.is_harassment_or_attack.str.contains(e).astype(float)
    d['attack'] = d.is_harassment_or_attack.str.contains('|'.join(classes[1:])).astype(float)
    return d

def map_aggression_score_to_3class(l):
    if l<0.0:
        return 0
    if l > 0.0:
        return 2
    else:
        return 1


def load_comments_and_labels(task):
    base_path = '../../data/annotations/split'
    splits = ['train', 'dev', 'test']
    nss = ['user', 'article']
    samples = ['blocked', 'random']
    dfs = {}
    for split in splits:
        path = os.path.join(base_path, split, 'annotations.tsv')
        df = pd.read_csv(path, sep = '\t')
        #print(df.shape)
        df.index = df.rev_id
        dfs[split] = df


    data = {}
    for ns in nss:
        data[ns] = {}
        for sample in samples:
            data[ns][sample] = {}
            for split in splits:
                data[ns][sample][split] = {'x':{}, 'y':{}}
                df = dfs[split].query("ns=='%s' and sample=='%s'" % (ns, sample))
                comments = df.drop_duplicates(subset='rev_id')['clean_diff']
                #print(comments.shape)
                labels = df[task]
                data[ns][sample][split]['x']['comments'] = comments
                ed = empirical_dist(labels)
                data[ns][sample][split]['y']['empirical_dist'] = ed
                weights = pd.Series(ed.columns, index=ed.columns)
                data[ns][sample][split]['y']['average'] = (ed * weights).sum(1)
                data[ns][sample][split]['y']['plurality'] = ed.idxmax(axis = 1)
    return data



def assemble_data(data, xtype, ytype, nss = ['user', 'article'], samples = ['random', 'blocked'], splits = ['train', 'dev', 'test']):
    xs = []
    ys = []

    for ns in nss:
        for sample in samples:
            for split in splits:
                x = data[ns][sample][split]['x'][xtype]
                #print(x.shape)
                y = data[ns][sample][split]['y'][ytype]
                #print(y.shape)
                x = x.loc[y.index]
                #print(x.shape)
                xs.append(x)
                ys.append(y)

    return pd.concat(xs).values, pd.concat(ys).values







