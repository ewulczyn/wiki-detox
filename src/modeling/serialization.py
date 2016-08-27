import keras
import keras.wrappers
import keras.wrappers.scikit_learn

import joblib
import os

def save_sklearn_pipepine(pipeline, directory, name):
    joblib.dump(pipeline, os.path.join(directory, '%s_pipeline.pkl' % name))
     
def load_sklearn_pipeline(directory, name):
    return joblib.load(os.path.join(directory, '%s_pipeline.pkl' % name))

def save_keras_pipeline(pipeline, directory, name):
    # save classifier
    clf = pipeline.steps[-1][1]
    # save classifier
    clf.model.save(os.path.join(directory, '%s_clf.h5' % name))
    # save feature extractor
    pipeline.steps = pipeline.steps[:-1]
    joblib.dump(pipeline, os.path.join(directory, '%s_extractor.pkl' % name))
         
def load_keras_pipeline(directory, name):
    clf_raw = keras.models.load_model(os.path.join(directory, '%s_clf.h5' % name))
    clf = keras.wrappers.scikit_learn.KerasClassifier(build_fn=clf_raw)
    clf.model = clf_raw
    pipeline = joblib.load(os.path.join(directory, '%s_extractor.pkl' % name))
    pipeline.steps.append(('clf', clf))
    return pipeline

def save_pipeline(pipeline, directory, name):
    
    clf = pipeline.steps[-1][1]
    is_keras = type(clf) == keras.wrappers.scikit_learn.KerasClassifier
    if is_keras and hasattr(clf, 'model'):
        save_keras_pipeline(pipeline, directory, name)
    else:
        save_sklearn_pipepine(pipeline, directory, name)
                 
def load_pipeline(directory, name):
    sklearn_file = os.path.join(directory, '%s_pipeline.pkl' % name)
    keras_clf_file = os.path.join(directory, '%s_clf.h5' % name)
    keras_extractor_file = os.path.join(directory, '%s_extractor.pkl' % name)
    
    if os.path.isfile(sklearn_file):
        return load_sklearn_pipeline(directory, name)
    elif os.path.isfile(keras_clf_file) and os.path.isfile(keras_extractor_file):
        return load_keras_pipeline(directory, name)
    else:
        print('Pipeline not saved')
        return None
