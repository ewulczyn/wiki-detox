import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
from scipy import stats


def create_column_of_counts(df, col):
    return df.apply(lambda x: col in str(x))

def create_column_of_counts_from_nums(df, col):
    return df.apply(lambda x: int(col) == x)

def preprocess(filename):
	dat = pd.read_csv('../data/' + filename)
	# Remove test questions
	dat = dat[dat['_golden'] == False]
	# Replace missing data with 'False'
	dat = dat.replace(np.nan, False, regex=True)
	
	attack_columns = ['not_attack', 'other', 'quoting', 'recipient', 'third_party']
	for col in attack_columns:
		dat[col] = create_column_of_counts(dat['is_harassment_or_attack'], col)

	aggressive_columns = ['-3', '-2', '-1', '0', '1', '2', '3']
	for col in aggressive_columns:
		dat[col] = create_column_of_counts_from_nums(dat['aggression_score'], col)

	dat['not_attack_0'] = 1 - dat['not_attack']
	dat['not_attack_1'] = dat['not_attack']

	# Group the data
	agg_dict = dict.fromkeys(attack_columns, 'mean')
	agg_dict.update(dict.fromkeys(aggressive_columns, 'sum'))
	agg_dict.update({'clean_diff': 'first', 'na': 'mean', 'aggression_score': 'mean', 
                 '_id':'count', 'not_attack_0':'sum', 'not_attack_1': 'sum', 
                 'block_timestamps': 'first', 'rev_timestamp': 'first'})
	grouped_dat = dat.groupby(['rev_id'], as_index=False).agg(agg_dict)

	# Get rid of data which the majority thinks is not in English or not readable
	grouped_dat = grouped_dat[grouped_dat['na'] < 0.5]

	return grouped_dat

def hist_comments(df, bins, plot_by, title):
    plt.figure()
    sliced_array = df[[plot_by]]
    weights = np.ones_like(sliced_array)/len(sliced_array)
    sliced_array.plot.hist(bins = bins, legend = False, title = title, weights=weights)
    plt.ylabel('Proportion')
    plt.xlabel('Average Score')


def sorted_comments(df, sort_by, quartile, num, is_ascending = True):
    n = df.shape[0]
    start_index = int(quartile*n)
    return df[['clean_diff', 'aggression_score',
              'not_attack', 'other', 'quoting', 'recipient', 'third_party']].sort_values(
        by=sort_by, ascending = is_ascending)[start_index:start_index + num]

def plot_and_test_aggressiveness(grouped_dat):
	# Get the timestamps of blocked events
	block_timestamps = grouped_dat['block_timestamps'].apply(lambda x: x.split('PIPE'))
	num_timestamps = block_timestamps.apply(len)
	# Focus on those users who have only been blocked once
	block_timestamps = block_timestamps[num_timestamps == 1]
	# Convert to datetime
	block_timestamps = [datetime.datetime.strptime(t[0], "%Y-%m-%dT%H:%M:%SZ") for t in block_timestamps]
	# Get the timestamps and scores of the corresponding revisions
	rev_timestamps = [datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ") for t in grouped_dat[num_timestamps == 1]['rev_timestamp']]
	rev_score = grouped_dat[num_timestamps == 1]['aggression_score']

	# Take the difference between the block timestamp and revision timestamp in seconds
	diff_timestamps = [np.diff(x)[0].total_seconds() for x in zip(block_timestamps, rev_timestamps)]

	x = pd.DataFrame(diff_timestamps)
	y = pd.DataFrame(rev_score)

	# Plot the aggressiveness score by relative time
	plt.figure()
	plt.plot(x,y,'bo', title = "Aggressiveness Patterns Relative to a Block Event")
	plt.xlim(-1e6, 1e6)
	plt.ylabel('Average Aggressiveness Score')
    plt.xlabel('Time Relative to Block Event (s)')

	# Seperate the revisions before and after a block event
	after_revs = y[x.values > 0]
	before_revs = y[x.values < 0]

	print "The mean aggressiveness before a block event:"
	print np.mean(before_revs)

	print "The mean aggressiveness after a block event: "
	print np.mean(after_revs)

	print "The results of a t-test"
	print stats.ttest_ind(before_revs, after_revs, equal_var=False)

