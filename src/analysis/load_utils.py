import os
import pandas as pd
import re


def load_diffs(keep_diff = False):
    
    nick_map = {
        'talk_diff_no_admin_sample.tsv': 'sample',
        'talk_diff_no_admin_2015.tsv': '2015',
        'all_blocked_user.tsv': 'blocked',
        'd_annotated.tsv': 'annotated',
    }

    base = '../../data/samples/'
    nss = ['user', 'article']

    samples = [  
        'talk_diff_no_admin_sample.tsv',
        'talk_diff_no_admin_2015.tsv',
        'all_blocked_user.tsv',
        'd_annotated.tsv'
    ]

    d ={}
    for s in samples:
        dfs = []
        for ns in nss:
            inf = os.path.join(base, ns, 'scored', s)
            df = pd.read_csv(inf, sep = '\t')
            if not keep_diff:
                del df['clean_diff']
            df['ns'] = ns
            dfs.append(df)
        d[nick_map[s]] = augment(pd.concat(dfs))

 
    d['blocked']['blocked'] = 1

    return d

def is_ip(x):
    pattern = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    return re.match(pattern,str(x)) is not None


def augment(df):
    df['author_anon'] = df['user_id'].isnull()
    df['recipient_anon'] = df['page_title'].apply(is_ip)
    df['rev_timestamp'] = pd.to_datetime(df['rev_timestamp'])
    df['year'] = df['rev_timestamp'].apply(lambda x: x.year)
    df['month'] = df['rev_timestamp'].apply(lambda x: x.month)
    df['hour'] = df['rev_timestamp'].apply(lambda x: x.hour)
    df['pred_recipient'] = (df['pred_recipient_score'] > 0.5).astype(int)
    df['own_page'] = df['user_text'] == df['page_title']
    return df



def load_block_events_and_users():
    
    df_events = pd.read_csv('../../data/block_events.tsv', sep = '\t')\
                    .rename(columns= lambda x: x.split('.')[1])\
                    .assign(timestamp= lambda x: pd.to_datetime(x.timestamp),
                            anon = lambda x: x.user_text.apply(is_ip))
                            
                            
    df_events['year'] = df_events['timestamp'].apply(lambda x: x.year)
    df_events['month'] = df_events['timestamp'].apply(lambda x: x.month)
    df_events['hour'] = df_events['timestamp'].apply(lambda x: x.hour)

    df_blocked_user_text = df_events[['user_text']]\
                            .drop_duplicates()\
                            .assign(blocked = 1) 

    return df_events, df_blocked_user_text


