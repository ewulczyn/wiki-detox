import argparse
import os, sys
import requests
import json
import pandas as pd
from baselines import one_hot, empirical_dist
from deep_learning import make_mlp, DenseTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from keras.wrappers.scikit_learn import KerasClassifier
from serialization import save_pipeline, load_pipeline

"""
python get_prod_models.py --task attack --model_dir ../../app/models 
python get_prod_models.py --task aggression --model_dir ../../app/models 
"""


COMMENTS_URL = 'https://ndownloader.figshare.com/files/6703926'
LABELS_URL = 'https://ndownloader.figshare.com/files/6703923'
COMMENTS_FILE = 'crowd_annotated_comments.tsv'
LABELS_FILE = 'crowd_annotations.tsv'
CV_RESULTS = 'cv_results.csv'





def download_file(url, fname):
    r = requests.get(url, stream=True)
    with open(fname, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def download_training_data(data_dir):
    download_file(COMMENTS_URL, os.path.join(data_dir, COMMENTS_FILE))
    download_file(LABELS_URL, os.path.join(data_dir, LABELS_FILE))


def parse_training_data(data_dir, task):
    comments = pd.read_csv(os.path.join(data_dir, COMMENTS_FILE), sep = '\t', index_col = 0)
    annotations = pd.read_csv(os.path.join(data_dir, LABELS_FILE),  sep = '\t', index_col = 0)
    labels = empirical_dist(annotations[task])
    X = comments.sort_index()['comment'].values
    y = labels.sort_index().values
    assert(X.shape[0] == y.shape[0])
    return X, y

def train_model(X, y, model_type, ngram_type, label_type):


    assert(label_type in ['oh', 'ed'])
    assert(model_type in ['linear', 'mlp'])
    assert(ngram_type in ['word', 'char'])

    if label_type == 'oh':
        y = one_hot(y)

    clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('to_dense', DenseTransformer()),
        ('clf', KerasClassifier(build_fn=make_mlp, output_dim = y.shape[1], verbose=False)),
    ])
    cv_results = pd.read_csv('cv_results.csv')
    query = "model_type == '%s' and ngram_type == '%s' and label_type == '%s'" % (model_type, ngram_type, label_type)
    params = cv_results.query(query)['best_params'].iloc[0]
    params = json.loads(params)

    return clf.set_params(**params).fit(X,y)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   default = '/tmp',   help ='directory for saving training data')
    parser.add_argument('--model_dir',  default = '/tmp',   help ='directory for saving model' )
    parser.add_argument('--task',       default = 'attack', help = 'either attack or aggression')
    parser.add_argument('--model_type', default = 'linear', help = 'either linear or mlp')
    parser.add_argument('--ngram_type', default = 'char',   help = 'either word or char')
    parser.add_argument('--label_type', default = 'oh' ,    help = 'either oh or ed')
    args = vars(parser.parse_args())

    print("Downloading Data")
    download_training_data(args['data_dir'])

    print("Parsing Data")
    X, y = parse_training_data(args['data_dir'], args['task'])

    print("Training Model")
    clf = train_model(X, y, args['model_type'], args['ngram_type'], args['label_type'])
    print(clf.predict_proba(['fuck']))

    print("Saving Model")
    clf_name = "%s_%s_%s_%s" % (args['task'], args['model_type'], args['ngram_type'], args['label_type'])
    save_pipeline(clf, args['model_dir'], clf_name)

    print("Reloading Model")
    clf = load_pipeline(args['model_dir'], clf_name)
    print(clf.predict_proba(['fuck']))



    

    