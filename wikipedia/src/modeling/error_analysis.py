def get_scores(model, X):
    try:
        scores = model.decision_function(X)
    except:
        scores = model.predict_proba(X)[:, 1]
    return scores


def get_clf_errors(model, data):
    """
    Returns false negatives and false positives in order of model certainty
    """
    df = data.copy()
    df['prediction'] = model.predict( data['x'])
    df['score'] = get_scores(model, data['x'])
    df['correct'] = (df['y'] == df['prediction'])
    fn = df.query('not correct and y==1').sort_values(by = 'score')
    fp = df.query('not correct and y==0').sort_values(by = 'score', ascending = False)
    return fn, fp


def print_clf_errors(d, n=10, boundary = False):

    if boundary:
        df = d.tail(n)
    else:
        df = d.head(n)

    for i, row in df.iterrows():
        print('COMMENT:')
        print(row['x'][:500].replace('\n', ''))
        print('SCORES: Actual: %d, Predicted: %d, Score: %0.2f' % (row['y'], row['prediction'], row['score']))
        print('\n')


def get_reg_residuals(model, data):
    """
    Returns false negatives and false positives in order of model certainty
    """
    df = data.copy()
    df['prediction'] = model.predict( data['x'])
    df['residual'] = (df['y']- df['prediction'])
    df['magnitude'] = df['residual'].abs()
    
    return df.sort_values(by = 'magnitude', ascending = False)


def print_reg_errors(d, n=10, over = True):

    
    df = d.head(n)
    

    for i, row in df.iterrows():
        print('COMMENT:')
        print(row['x'][:500].replace('\n', ''))
        print('SCORES: Actual: %0.2f, Predicted: %0.2f, Residual: %0.2f' % (row['y'], row['prediction'], row['residual']))
        print('\n')


