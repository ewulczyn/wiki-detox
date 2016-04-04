try:
  from bs4 import BeautifulSoup
  import mwparserfromhell
except:
  print('Could not import bs4 or mwparserfromhell')
  pass
import re
import pandas as pd


def show_comments(d, n = 10):
    for i, r in d[:n].iterrows():
        print(r['diff'])
        print('_' * 80)
        print(r['clean_diff'])
        print('\n\n', '#' * 80, '\n\n')

### Strip Admin Messages

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
    '== Welcome to Wikipedia! ==',
    '== Welcome! ==',
    '== 5 Million: We celebrate your contribution ==',
    ]

compiled_patterns = {p:re.compile(p) for p in patterns}


def find_pattern_df(df,col, pattern):
    p = re.compile(pattern)
    return df[df[col].apply(lambda x: p.search(x) is not None)]

def has_pattern(col, pattern):
    return lambda x: compiled_patterns[pattern].search(x[col]) is not None



def exclude_pattern_df(df, col, pattern):
    p = re.compile(pattern)
    return df[df[col].apply(lambda x: p.search(x) is None)]

def does_not_have_pattern(col, pattern):
    return lambda x: compiled_patterns[pattern].search(x[col]) is None



def exclude_few_tokens_df(df, col, min_tokens):
    return df[df[col].apply(lambda x: len(x.split(' ')) >= min_tokens)]

def has_enough_tokens(col, min_tokens):
    return lambda x: len(x[col].split(' ')) >= min_tokens


def exclude_short_strings_df(df, col, min_chars):
    return df[df[col].apply(lambda x: len(x) >= min_chars)]  

def has_enough_chars(col, min_chars):
    return lambda x: len(x[col]) >= min_chars


def remove_admin_df(df, patterns):
    d_reduced = copy.deepcopy(df)
    for pattern in patterns:
        d_reduced = exclude_pattern(d_reduced, pattern, 'diff')
    return d_reduced

def no_patterns(col, patterns):
    def f(x):
        fs  = [has_pattern(col, p) for p in patterns]
        return sum([f(x) for f in fs]) == 0
    return f 



### Clean up Markup ###
    
def remove_date(comment):

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
    return re.sub(date_p , '', comment )



pre_sub_patterns = [
                      ('\[\[Image:.*?\]\]', ''),
                      ('<!-- {{blocked}} -->', ''),
                      ('NEWLINE', '\n'),
                      ('\[\[File:.*?\]\]', ''),
                      ('\[\[User:.*?\|.*?\]\]', ''),
                      ('\(\[\[User talk:.*?\|talk\]\]\)', ''),
                   ]

post_sub_patterns = [
                      ('--', ''),
                      (' :', ' '),
                    ]

def substitute_patterns(s, sub_patterns):
    for p, r in sub_patterns:
        s = re.sub(p, r, s)
    return s

def strip_html(comment):
    return BeautifulSoup(comment, 'html.parser').get_text()

def strip_mw(comment):
    return mwparserfromhell.parse(comment).strip_code()


def clean(df):
    df = copy.deepcopy(df)
    df.dropna(subset = ['insertion'], inplace = True)
    col = 'clean_insertion'
    df[col] = df['insertion']
    df[col] = df[col].apply(remove_date)
    df[col] = df[col].apply(lambda x: substitute_patterns(x, pre_sub_patterns))
    df[col] = df[col].apply(strip_mw)
    df[col] = df[col].apply(strip_html)
    df[col] = df[col].apply(lambda x: substitute_patterns(x, post_sub_patterns))
    df.dropna(subset = [col], inplace = True)
    df = df[df[col] != '']
    return df


def apply_field_modifier(f, col):
    def mod(x):
      x[col] = f(x[col])
      return x
    return mod

def apply_complex_field_modifier(f, col, aux):
    def mod(x):
      x[col] = f(x[col], aux)
      return x
    return mod


def clean_rdd(rdd):
  col = 'clean_insertion'

  def copy_insertion(x):
    x[col] = x['insertion']

  return rdd.filter(lambda x: x['insertion'] is not None and len(x['insertion']) > 0) \
           .map(copy_insertion) \
           .map(apply_field_modifier(remove_date, col)) \
           .map(apply_complex_field_modifier(substitute_patterns, col, pre_sub_patterns)) \
           .map(apply_field_modifier(strip_mw, col)) \
           .map(apply_field_modifier(strip_html, col)) \
           .map(apply_complex_field_modifier(substitute_patterns, col, post_sub_patterns)) \
           .filter(lambda x: x[col] is not None and len(x[col]) > 0)


