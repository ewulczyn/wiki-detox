import sklearn
import requests
import os
import sys
import inspect
from pprint import pprint
from bs4 import BeautifulSoup


currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from wikipedia.src.data_generation.diff_utils import remove_date, substitute_patterns, strip_mw, strip_html,pre_sub_patterns,post_sub_patterns



def clean_raw_diff(diff):
    diff = get_content_added(diff)
    diff = remove_date(diff)
    diff = substitute_patterns(diff, pre_sub_patterns)
    diff = strip_mw(diff)
    diff = strip_html(diff)
    diff = substitute_patterns(diff, post_sub_patterns)
    return diff


def get_content_added(diff):
    soup = BeautifulSoup(diff, 'html.parser')
    cols = soup.find_all('td', attrs={'class':'diff-addedline'})
    cols = [e.text.strip().strip(':') for e in cols]
    cols = [e for e in cols if e]
    comment =  ' '.join(cols)
    return comment


def get_comment(rev_id):
    url = 'https://en.wikipedia.org/w/api.php?action=query&prop=revisions&revids=%s&rvdiffto=prev&format=json'
    url = url % rev_id
    r = requests.get(url)
    try:
        diff = list(r.json()['query']['pages'].values())[0]['revisions'][0]['diff']['*']
    except:
        diff = ''
        print(sys.exc_info()[0])

    return clean_raw_diff(diff)


