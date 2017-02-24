#### Establish correlation

Short term effects:

1. Take pairs of consecutive months. Regress 1{participation in month t2 is lower than in t1} on 1{ user experienced at least one attack in month t1}. Consider only taking one sample per user. Interpret coefficient.

2. Regress count(edits in t2) on count(edits t1), count(attacks in t1). Consider only taking one sample per user. Interprest coefficient. Consider poisson regression.

3. Add interaction terms: gender, topics, newness. Interpret interaction coefficients.

4. Limit data to new users, who made no personal attacks before they were attacked

5. Compare short term effects between new and experienced users

6. compare effect across namespaces. Could be that you see increased particiapation in talk and decrease in main

7. See if just participating in toxic discussion pages /articles makes you more likely to leave,  even if you don't recieve attacks on your talk page. As measures, consider edits to controversial topics/articles
https://en.wikipedia.org/wiki/Wikipedia:List_of_controversial_issues
https://en.wikipedia.org/wiki/Category:Wikipedia_controversial_topics
or ranking pages by proportion and volume of toxic comments




Long term effects (WIP)

1. LSTM on count(edits in month t), count(attacks month t), count(attack in t1-t0). Start at month the user registered. Train on balance of users who have and who have not experienced harassment. Compare accuraccy on hold out of users broken down by whether they set have been attacked at least once, with and without attack features.  


2. LSTM on inter-edit time interval. To measure impact: add random attacks to non-attacked users and see if inter edit time predictions increase?