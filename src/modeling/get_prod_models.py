import argparse
import os, sys
import urllib
import json
import pandas as pd
from baselines import one_hot, empirical_dist
from deep_learning import make_mlp, DenseTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from keras.wrappers.scikit_learn import KerasClassifier
from serialization import save_pipeline, load_pipeline
from sklearn.linear_model import LogisticRegression
import numpy as np

"""
USAGE EXAMPLE:
python get_prod_models.py --task attack --model_dir ../../app/models
python wiki-detox/src/modeling/get_prod_models.py --task recipient_attack --model_dir ../../app/models
python get_prod_models.py --task aggression --model_dir ../../app/models
python get_prod_models.py --task toxicity --model_dir ../../app/models

"""

# Figshare URLs for downloading training data
ATTACK_ANNOTATED_COMMENTS_URL = 'https://ndownloader.figshare.com/files/7554634'
ATTACK_ANNOTATIONS_URL = 'https://ndownloader.figshare.com/files/7554637'
AGGRESSION_ANNOTATED_COMMENTS_URL = 'https://ndownloader.figshare.com/files/7038038'
AGGRESSION_ANNOTATIONS_URL = 'https://ndownloader.figshare.com/files/7383748'
TOXICITY_ANNOTATED_COMMENTS_URL = 'https://ndownloader.figshare.com/files/7394542'
TOXICITY_ANNOTATIONS_URL = 'https://ndownloader.figshare.com/files/7394539'
# CSV of optimal  hyper-parameters for each model architecture
CV_RESULTS = 'cv_results.csv'



def download_file(url, fname):
    """
    Helper function for downloading a file to disk
    """
    urllib.request.urlretrieve(url, fname)

def download_training_data(data_dir, task):

    """
    Downloads comments and labels for task
    """

    COMMENTS_FILE = "%s_annotated_comments.tsv" % task
    LABELS_FILE = "%s_annotations.tsv" % task

    if task == "attack":
        download_file(ATTACK_ANNOTATED_COMMENTS_URL,
                      os.path.join(data_dir, COMMENTS_FILE))
        download_file(ATTACK_ANNOTATIONS_URL, os.path.join(data_dir,
                      LABELS_FILE))
    elif task == "recipient_attack":
        download_file(ATTACK_ANNOTATED_COMMENTS_URL,
                      os.path.join(data_dir, COMMENTS_FILE))
        download_file(ATTACK_ANNOTATIONS_URL, os.path.join(data_dir,
                      LABELS_FILE))
    elif task == "aggression":
        download_file(AGGRESSION_ANNOTATED_COMMENTS_URL,
                      os.path.join(data_dir, COMMENTS_FILE))
        download_file(AGGRESSION_ANNOTATIONS_URL,
                      os.path.join(data_dir, LABELS_FILE))
    elif task == "toxicity":
        download_file(TOXICITY_ANNOTATED_COMMENTS_URL,
                      os.path.join(data_dir, COMMENTS_FILE))
        download_file(TOXICITY_ANNOTATIONS_URL,
                      os.path.join(data_dir, LABELS_FILE))
    else:
        print("No training data for task: ", task)



def parse_training_data(data_dir, task):

    """
    Computes labels from annotations and aligns comments and labels for training
    """

    COMMENTS_FILE = "%s_annotated_comments.tsv" % task
    LABELS_FILE = "%s_annotations.tsv" % task

    comments = pd.read_csv(os.path.join(data_dir, COMMENTS_FILE), sep = '\t', index_col = 0)
    # remove special newline and tab tokens

    comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
    comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))


    annotations = pd.read_csv(os.path.join(data_dir, LABELS_FILE),  sep = '\t', index_col = 0)
    labels = empirical_dist(annotations[task])

    X = comments.sort_index()['comment'].values
    y = labels.sort_index().values

    assert(X.shape[0] == y.shape[0])
    return X, y


def train_model(X, y, model_type, ngram_type, label_type):
    """
    Trains a model with the specified architecture. Note that the
    classifier is a Sklearn model when setting label_type == 'oh'
    and model_type == 'linear'. Otherwise the classifier is a
    Keras model. The distinction is important for serialization.
    """


    assert(label_type in ['oh', 'ed'])
    assert(model_type in ['linear', 'mlp'])
    assert(ngram_type in ['word', 'char'])


    # tensorflow models aren't fork safe, which means they can't be served via uwsgi
    # as work around, we can serve a pure sklearn model
    # we should be able to find another fix

    if label_type == 'oh' and model_type == 'linear':

        y = np.argmax(y, axis = 1)

        clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', LogisticRegression()),
        ])

        params = {
            'vect__max_features': 10000,
            'vect__ngram_range': (1,5),
            'vect__analyzer' : ngram_type,
            'tfidf__sublinear_tf' : True,
            'tfidf__norm' :'l2',
            'clf__C' : 10,
        }

    else:

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
    parser.add_argument('--skip_download',   default = 'false',   help ='do not download data if "true"')
    parser.add_argument('--data_dir',   default = '/tmp',   help ='directory for saving training data')
    parser.add_argument('--model_dir',  default = '/tmp',   help ='directory for saving model' )
    parser.add_argument('--task',       default = 'attack', help = 'either attack, recipient_attack, aggression or toxicity')
    parser.add_argument('--model_type', default = 'linear', help = 'either linear or mlp')
    parser.add_argument('--ngram_type', default = 'char',   help = 'either word or char')
    parser.add_argument('--label_type', default = 'oh' ,    help = 'either oh or ed')
    args = vars(parser.parse_args())

    if args['skip_download'] != 'true':
        print("Downloading Data")
        download_training_data(args['data_dir'], args['task'])

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






