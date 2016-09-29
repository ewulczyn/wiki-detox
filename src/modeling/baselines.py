import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr
from scipy.stats import entropy as kl
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error
from math import sqrt
import os
import multiprocessing as mp
import dill







def get_annotator_ensemble_baseline(annotations, k, agg_function, eval_function, n_t, n_p):

    assert(n_t + n_p <=k)

    np.random.seed()
    annotations = annotations.dropna()
    groups = annotations.groupby(annotations.index)
    groups = [e[1] for e in groups if e[1].shape[0]>=k]
        
    d_ts = []
    d_ps = []
    for g in groups:
        g = g.iloc[np.random.permutation(len(g))]
        d_ts.append(g[0:n_t])
        d_ps.append(g[n_t:(n_t+n_p)])
       

    d_t =  pd.concat(d_ts)
    d_p = pd.concat(d_ps)

    scores_t = agg_function(d_t).values
    scores_p = agg_function(d_p).values

    return {'score' : eval_function(scores_t, scores_p), 'n_t' : n_t, 'n_p': n_p }



def get_annotator_ensemble_baseline_helper(args):
    return get_annotator_ensemble_baseline(*args)

def get_annotator_ensemble_baselines_parallel(args_list, n_jobs = 8):

    """
    Run function in parallel with args in args_list, function must return dict of results.
    """
    p = mp.Pool(min(n_jobs, len(args_list)))
    res = p.map(get_annotator_ensemble_baseline_helper, args_list)
    p.close()
    p.join()
    #res = [f(args) for args in args_list]
    return pd.DataFrame(res)

def get_model_baseline(model_predictions, annotations, k, agg_function, eval_function, n_t):

    """      
    """

    assert(n_t <= k)

    np.random.seed()
    annotations = annotations.dropna()
    groups = annotations.groupby(annotations.index)
    groups = [e[1] for e in groups if e[1].shape[0]>=k]

    d_ts = []
    for g in groups:
        g = g.iloc[np.random.permutation(len(g))]
        d_ts.append(g[0:n_t])
        

    d_t =  pd.concat(d_ts)

    scores_t = agg_function(d_t)
    model_predictions = model_predictions.loc[scores_t.index]

    return {'score' : eval_function(scores_t.values, model_predictions.values), 'n_t' : n_t }
 

def get_model_baseline_helper(args):
    return get_model_baseline(*args)

def get_model_baselines_parallel(args_list, n_jobs = 8):

    """
    Run function in parallel with args in args_list, function must return dict of results.
    """
    p = mp.Pool(min(n_jobs, len(args_list)))
    res = p.map(get_model_baseline_helper, args_list)
    p.close()
    p.join()
    #res = [f(args) for args in args_list]
    return pd.DataFrame(res)


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

def map_aggression_score_to_2class(l):
    if l<0.0:
        return 1
    if l >= 0.0:
        return 0


def load_comments_and_labels(task):
    base_path = '../../data/annotations/split'
    splits =  ['train', 'dev', 'test', 'baseline']
    nss = ['user', 'article']
    samples = ['blocked', 'random']
    dfs = {}
    for split in splits:
        path = os.path.join(base_path, split, 'annotations.tsv')
        df = pd.read_csv(path, sep = '\t')
        #print(df.shape)
        #print(len(df['rev_id'].unique()))
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
                data[ns][sample][split]['y']['one_hot'] = ed.apply(lambda x: (x > (1.0 / ed.shape[1])).astype(int))
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

    x = pd.concat(xs).values
    #print(x.shape)
    y = pd.concat(ys).values
    #print(y.shape)
    return x, y







