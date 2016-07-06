import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def mpg(df, score, cols):
    """
    return row with max score in each group of cols values
    """
    return df.sort(score, ascending=False).groupby(cols, as_index=False).first()

def compare_groups(df, x, mpu = False, order = None, hue = None, plot = True, table = True):
    agg = 'pred_aggression_score'
    rec = 'pred_recipient_score'
    
    if table:
        if hue:
            print(df.groupby([x, hue])[agg, rec].mean())
        else:
            print(df.groupby([x])[agg, rec].mean())
    
    if plot:
        
        fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, figsize = (12,6))

        if mpu:
            cols = ['user_text', x]
            plt.figure()
            sns.pointplot(x=x, y= agg, data=mpg(df, agg, cols) , order = order, hue = hue, ax = ax1)
            plt.figure()
            sns.pointplot(x=x, y= rec, data=mpg(df, rec, cols) , order = order, hue = hue, ax = ax2)
        else:
            
            ax = sns.pointplot(x=x, y= agg, data=df, order = order, hue = hue, ax = ax1)
            plt.figure()
            ax = sns.pointplot(x=x, y= rec, data=df, order = order, hue = hue, ax = ax2)
            


def get_genders(d):
    d_gender = pd.read_csv('../../data/genders.tsv', sep = '\t')


    def remap_author_gender(r):
        if r['gender'] == 'male':
            return 'male'
        elif r['gender'] == 'female':
            return 'female'
        elif r['author_anon']:
            return 'unknown:anon'
        else:
            return 'unknown: registered'


    def remap_recipient_gender(r):
        if r['gender'] == 'male':
            return 'male'
        elif r['gender'] == 'female':
            return 'female'
        elif r['recipient_anon']:
            return 'unknown:anon'
        else:
            return 'unknown: registered'


    d = d.\
        merge(d_gender, how = 'left', on = 'user_id', suffixes = ('', '_x')).\
        assign(author_gender = lambda x: x.apply(remap_author_gender, axis = 1)).\
        drop(['gender', 'user_text_x'], axis=1)

    
    d = d.\
        merge(d_gender, how = 'left', left_on = 'page_title', right_on = 'user_text', suffixes = ('', '_x')).\
        assign(recipient_gender = lambda x: x.apply(remap_recipient_gender, axis = 1)).\
        drop(['gender', 'user_text_x', 'user_id_x'], axis=1)

    return d
    
    
