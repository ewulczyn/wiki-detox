import sklearn
import requests
import os
import sys
import inspect
from pprint import pprint

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from wikipedia.src.mw_api_diff_utils import clean_raw_diff


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


