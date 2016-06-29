import pandas as pd

def get_errors(comments, true, pred):
    """
    Returns false negatives and false positives in order of model certainty
    """
    df = pd.DataFrame({'x': comments, 'true': true, 'pred': pred})
    df['residual'] = (df['true']- df['pred'])
    df['magnitude'] = df['residual'].abs()

    over = df.query('residual < 0').sort_values(by = 'magnitude', ascending = False)
    under = df.query('residual > 0').sort_values(by = 'magnitude', ascending = False)
    return over, under



def print_errors(d, n=10):

    df = d.head(n)
    
    for i, row in df.iterrows():
        print('COMMENT:')
        print(row['x'][:500].replace('\n', ''))
        print('SCORES: Actual: %0.2f, Predicted: %0.2f, Residual: %0.2f' % (row['true'], row['pred'], row['residual']))
        print('\n')