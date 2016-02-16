from db_utils import query_analytics_store
import requests
from bs4 import BeautifulSoup
from pprint import pprint
import mwparserfromhell
import re
import sys
import concurrent
import itertools
import pandas as pd
import time
import datetime


def get_blocked_users(keywords):
    
    params = {
        'keywords': '|'.join(keywords)
    }
    
    query = """
    SELECT 
        log_comment as reason,
        log_timestamp as timestamp,
        log_title as user_text
    FROM enwiki.logging 
        WHERE log_type = 'block'
        AND log_comment RLIKE %(keywords)s
    """

    return query_analytics_store(query  , params)



def get_blocked_users_talk_page_comment_meta_data(d_blocked_users, max_comments, namespace):
    """
    Given a Dataframe of blocked users, gets the last max_comments
    namespace comments
    """
    
    def shift_date(mw_timestamp, days):
        ts = datetime.datetime.strptime(str(mw_timestamp), "%Y%m%d%H%M%S")
        ts +=  datetime.timedelta(days=days)
        return ts.strftime("%Y%m%d%H%M%S")


    def get_user_talk_page_comment_meta_data(x):
        i, row = x

        blocked_timestamp = row['timestamp']
        user_text = row['user_text']

        params = {
            'blocked_timestamp':blocked_timestamp,
            'user_text':user_text,
            'max_comments': max_comments,
            'min_date': shift_date(blocked_timestamp, -14), 
            'namespace': namespace
        }

        query = """
            SELECT
                page_id,
                page_namespace,
                page_title,
                rev_comment,
                rev_id,
                rev_minor_edit as minor_edit,
                rev_timestamp as timestamp,
                rev_user as user_id,
                rev_user_text as user_text
            FROM
                enwiki.revision r,
                enwiki.page p 
            WHERE
                r.rev_page = p.page_id
                AND rev_timestamp > %(min_date)s
                AND rev_timestamp < %(blocked_timestamp)s
                AND page_namespace = %(namespace)d
                AND rev_user_text = '%(user_text)s'
            ORDER BY rev_timestamp DESC
            LIMIT %(max_comments)d
        """
        try:
            return query_analytics_store(query % params , {})
        except:
            return pd.DataFrame()
    
    t1 = time.time()
    with concurrent.futures.ThreadPoolExecutor(10) as executor:
        results = executor.map(get_user_talk_page_comment_meta_data, d_blocked_users.iterrows())
        d_meta = pd.concat(results, axis=0)
    t2 = time.time()
    print((t2-t1))
    print('Rate:', float(d_blocked_users.shape[0]) / ((t2-t1)/60.0), 'rpm')
    d_meta.drop_duplicates(subset='rev_id', inplace = True)
    d_meta.index = d_meta['rev_id']
    return d_meta

def get_talk_page_comment_meta_data(n, min_date, namespace):
    """
    Fetches metadata for n talk page comments from namespace
    from after min_date.
    """

    assert(namespace in [1, 3])
    query = """
    SELECT
        page_id,
        page_namespace,
        page_title,
        rev_comment,
        rev_id,
        rev_minor_edit as minor_edit,
        rev_timestamp as timestamp,
        rev_user as user_id,
        rev_user_text as user_text
    FROM
        enwiki.revision r,
        enwiki.page p 
    WHERE
        r.rev_page = p.page_id
        AND rev_timestamp > %(min_date)d
        AND page_namespace = %(namespace)d
    LIMIT %(num_comments)d
    """

    params = {
        'num_comments':n,
        'min_date':min_date,
        'namespace': namespace,
    }

    d = query_analytics_store(query % params , {})
    d.index = d['rev_id']
    return d


def get_raw_diffs_concurrent(d, n_threads = 5):
    """
    Given a Dataframe of revision meta-data, adds
    a columns with the raw diffs via concurrent 
    API calls
    """

    def get_diff(x):
        i, row = x
        rev_id = row['rev_id'] 
        page_id = row['page_id'] 
        url = 'https://en.wikipedia.org/w/api.php?action=query&prop=revisions&revids=%d&rvdiffto=prev&format=json'
        url = url % rev_id
        r = requests.get(url)
        try:
            diff = r.json()['query']['pages'][str(page_id)]['revisions'][0]['diff']['*']
        except:
            diff = 'ERROR'
            print(sys.exc_info()[0])
            
        return (row['rev_id'], diff )  

    t1 = time.time()
    with concurrent.futures.ThreadPoolExecutor(n_threads) as executor:
        results = executor.map(get_diff, d.iterrows())
        index = []
        diffs = []
        for rev_id, diff in results:
            index.append(rev_id)
            diffs.append(diff)
        diff_series = pd.Series(diffs, index = index)
    t2 = time.time()

    d['raw_diff'] = diff_series
    d['diff'] = diff_series

    print('Rate:', float(d.shape[0]) / ((t2-t1)/60.0), 'rpm')
    return d



def clean_raw_diff(diff):
    diff = get_content_added(diff)
    diff = remove_talk_page_link(diff)
    diff = remove_signature(diff)
    diff = remove_date(diff)
    diff = strip_mw(diff)
    diff = strip_html(diff)
    return diff


def clean_raw_diffs(d):
    """
    Takes a dataframe with raw diffs and cleans them up. Empty revisions are deleted
    """
    d['diff'] = d['raw_diff'].apply(clean_raw_diff)
    return d[d['diff'] != '']


# Diff Parsing and Cleaning Functions
def get_content_added(diff):
    soup = BeautifulSoup(diff, 'html.parser')
    cols = soup.find_all('td', attrs={'class':'diff-addedline'})
    cols = [e.text.strip().strip(':') for e in cols]
    cols = [e for e in cols if e]
    comment =  ' '.join(cols)
    return comment

def strip_html(comment):
    return BeautifulSoup(comment, 'html.parser').get_text()

def strip_mw(comment):
    return mwparserfromhell.parse(comment).strip_code()

def remove_signature(comment):
    return re.sub('\[\[User.*?\]\]', '', comment)

def remove_talk_page_link(comment):
    return re.sub('\(\[\[User.*?\]\]\)', '', comment)


months = ['January',
          'February',
          'March',
          'April',
          'June',
          'July',
          'August',
          'September',
          'October',
          'November',
          'December',
        ]


month_or = '|'.join(months)
date_p = re.compile('\d\d:\d\d, \d?\d (%s) \d\d\d\d \(UTC\)' % month_or)
    
def remove_date(comment):
    return re.sub(date_p , '', comment )