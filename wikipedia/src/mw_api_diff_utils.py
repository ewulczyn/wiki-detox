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