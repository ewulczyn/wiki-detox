from bs4 import BeautifulSoup
import  mwparserfromhell
import re
import pandas as pd
import copy
import time


### Diff Cleaning ###

months = ['January',
          'February',
          'March',
          'April',
          'May',
          'June',
          'July',
          'August',
          'September',
          'October',
          'November',
          'December',
          'Jan',
          'Feb',
          'Mar',
          'Apr',
          'May',
          'Jun',
          'Jul',
          'Aug',
          'Sep',
          'Oct',
          'Nov',
          'Dec',
        ]


month_or = '|'.join(months)
date_p = re.compile('\d\d:\d\d,( \d?\d)? (%s)( \d?\d)?,? \d\d\d\d (\(UTC\))?' % month_or)
    
def remove_date(comment):
    return re.sub(date_p , '', comment )



pre_sub_patterns = [
                    ('\[\[Image:.*?\]\]', ''),
                    ('<!-- {{blocked}} -->', ''),
                    ('\[\[File:.*?\]\]', ''),
                    ('\[\[User:.*?\]\]', ''),
                    ('\[\[user:.*?\]\]', ''),
                    ('\(?\[\[User talk:.*?\]\]\)?', ''),
                    ('\(?\[\[user talk:.*?\]\]\)?', ''),
                    ('\(?\[\[User Talk:.*?\]\]\)?', ''),
                    ('\(?\[\[User_talk:.*?\]\]\)?', ''),
                    ('\(?\[\[user_talk:.*?\]\]\)?', ''),
                    ('\(?\[\[User_Talk:.*?\]\]\)?', ''),
                    ('\(?\[\[Special:Contributions.*?\]\]\)?', ''),
                   ]

post_sub_patterns = [
                    ('--', ''),
                    (' :', ' '),
                    ]

def substitute_patterns(s, sub_patterns):
    for p, r in sub_patterns:
        s = re.sub(p, r, s)
    return s

def strip_html(s):
    try:
        s = BeautifulSoup(s, 'html.parser').get_text()
    except:
        pass
        #print('BS4 HTML PARSER FAILED ON:', s)
    return s

def strip_mw(s):
    try:
        s = mwparserfromhell.parse(s).strip_code()
    except:
        pass
    return s


def clean_comment(s):
    s = remove_date(s)
    s = substitute_patterns(s, pre_sub_patterns)
    s = strip_mw(s)
    s = strip_html(s)
    s = substitute_patterns(s, post_sub_patterns)
    return s


def clean(df):
    df = copy.deepcopy(df)
    df.rename(columns = {'insertion': 'diff'}, inplace = True)
    df.dropna(subset = ['diff'], inplace = True)
    df['clean_diff'] = df['diff']
    df['clean_diff'] = df['clean_diff'].apply(remove_date)
    df['clean_diff'] = df['clean_diff'].apply(lambda x: substitute_patterns(x, pre_sub_patterns))
    df['clean_diff'] = df['clean_diff'].apply(strip_mw)
    df['clean_diff'] = df['clean_diff'].apply(strip_html)
    df['clean_diff'] = df['clean_diff'].apply(lambda x: substitute_patterns(x, post_sub_patterns))

    try:
        del df['rank']
    except:
        pass
    df.dropna(subset = ['clean_diff'], inplace = True)
    if not df.empty:
        df = df[df['clean_diff'] != '']
    return df



def show_comments(d, n = 10):
    for i, r in d[:n].iterrows():
        print(r['diff'])
        print('_' * 80)
        print(r['clean_diff'])
        print('\n\n', '#' * 80, '\n\n')


### Admin Filtering ###
# Currently done in HIVE

def find_pattern(d, pattern, column):
    p = re.compile(pattern)
    return d[d[column].apply(lambda x: p.search(x) is not None)]

def exclude_pattern(d, pattern, column):
    p = re.compile(pattern)
    return d[ d[column].apply(lambda x: p.search(x) is None)]

def exclude_few_tokens(d, n):
    return d[d['clean_diff'].apply(lambda x:  len(x.split(' ')) > n)]

def exclude_short_strings(d, n):
    return d[d['clean_diff'].apply(lambda x:  len(x) > n)]  

def remove_admin(d, patterns):
    d_reduced = copy.deepcopy(d)
    for pattern in patterns:
        d_reduced = exclude_pattern(d_reduced, pattern, 'diff')
    return d_reduced

patterns =[
    '\[\[Image:Octagon-warning',
    '\[\[Image:Stop',
    '\[\[Image:Information.',
    '\[\[Image:Copyright-problem',
    '\[\[Image:Ambox',
    '\[\[Image:Broom',
    '\[\[File:Information',
    '\[\[File:AFC-Logo_Decline',
    '\[\[File:Ambox',
    '\[\[File:Nuvola',
    '\[\[File:Stop',
    '\[\[File:Copyright-problem',
    '\|alt=Warning icon\]\]',
    'The article .* has been \[\[Wikipedia:Proposed deletion\|proposed for deletion\]\]',
    'Your submission at \[\[Wikipedia:Articles for creation\|Articles for creation\]\]',
    'A file that you uploaded or altered, .*, has been listed at \[\[Wikipedia:Possibly unfree files\]\]',
    'User:SuggestBot',
    '\[\[Wikipedia:Criteria for speedy deletion\|Speedy deletion\]\] nomination of',
    "Please stop your \[\[Wikipedia:Disruptive editing\|disruptive editing\]\]. If you continue to \[\[Wikipedia:Vandalism\|vandalize\]\] Wikipedia, as you did to .*, you may be \[\[Wikipedia:Blocking policy\|blocked from editing\]\]",
    "Hello.*and.*\[\[Project:Introduction\|welcome\]\].* to Wikipedia!",
    'Nomination of .* for deletion',
    '==.*Welcome.*==',
    '== 5 Million: We celebrate your contribution ==',
    '==.*listed for discussion ==',
    ]



def clean_and_filter(df, min_words=3, min_chars=20):
    t1 = time.time()
    #print('Raw:', df.shape[0])
    df = clean(df).dropna(subset = ['clean_diff'])
    if df.empty:
        return df
    #print('Cleaned: ', df.shape[0])
    df = exclude_few_tokens(df, min_words)
    #print('No Few Words: ', df.shape[0])
    df = exclude_short_strings(df, min_chars)
    #print('No Few Chars: ', df.shape[0])
    t2 = time.time()
    #print('Cleaning and Filtering Time:',(t2-t1) / 60.0)
    return df

### Data Viz ###
def print_block_data(r):
    block_reasons = r['block_reasons'].split('PIPE')
    block_timestamps = r['block_timestamps'].split('PIPE')
    block_actions = r['block_actions'].split('PIPE')
    block_params = r['block_params'].split('PIPE')
    
    for i in range(len(block_reasons)):
        print('Log Event #: ', i+1)
        print('Action:',block_actions[i] )
        print('Time:', block_timestamps[i] )
        print('Reason:',block_reasons[i] )
        print('Parameters:',block_params[i] )

def print_user_history(d, user):
    
    """
    Print out users comments in order
    """
    d = d.fillna('')
    
    d_u = d[d['user_text'] == user].sort_values(by ='rev_timestamp')
    
    print ('#' * 80)
    print('History for user: ', user)
    print('\nBlock History')
    print_block_data(d_u.iloc[0])
    print ('#' * 80)
    
    
    for i, r in d_u.iterrows():
        print ('\n')
        print ('User: ', r['user_text'])
        print ('User Talk Page: ', r['page_title'])
        print('Timestamp: ', r['rev_timestamp'])
        print ('\n')
        print (r['clean_diff'])
        print ('_' * 80)