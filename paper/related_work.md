# Talk Page Abuse Modeling Notes

## Community Discussions

### Village Pump Proposal
- Create a special tag / edit filter designed to catch talk page abuse
- Example:` You're so dogged. It really . . . turns me on. But I'm confused. I removed some material from my talk page today at the behest of Slim Virgin. I'm confused as to what more you're asking me to do. Because, you know . . . really . . . I would do almost anything for you. . . .`
- Example: `This blocking thing is a joke - I already found your address and it is being discussed how to destroy you... will watch u scream and enjoy, MJ!!! Can't wait to thrash you with my belt !! —The preceding unsigned comment was added by AlamSrini1 (talk • contribs) 21:20, 14 February 2007 (UTC).`
- Example: `i am going to rape you- bice perason`

#### Envisaged benefits
- An edit filter could warn users before posting that their comment may need to be refactored to be considered appropriate.
- Editors could check recent changes for tagged edits, bringing much-needed third eyes to talk pages where an editor may be facing sexual harassment and other types of abuse.
- Prevention of talk page escalation.
- Improvement of talk page culture.

- there are lots of suggestions for how to incorporate the algorithm into the wiki in the discussion
- call for a clear definition of abuse: e.g. sexual harassment
- opposition mainly stems from perceived infeasibility to build a good model
- skeptics don't think it is a big problem (we should prove it)

- there should be a feedback mechanism that asks people to confirm if the edit is in fact abusive
-  `given that the hostile editing climate is often cited as a reason for both the gender gap and poor editor retention, why not explore whether Wikipedia could be more proactive about talk page interactions, instead of putting the onus on victims to come forward and point this stuff out to an admin? `

### Community Wish List Proposal
- made by same user as above (Jayen466)
- site success in League of Legends Community
- totality of revision-deleted and oversighted talk page posts in the English Wikipedia could provide an initial dataset
- `Keep its purpose to a manageable task (fending off gross abuse) and this bot could make WP a much more attractive place. Currently, Talkpages are only lightly guarded, and a surprising amount of unhelpful remarks (i.e. obvious vandalism in the form of profane non sequiturs and other gratuitous scribbling) is kept forever, much to the detriment of WP's public image`

### Wikimedia -l Thread
- Denny forwards League of Legends project
- someone points out that a competitive battle game elicit stress and intense responses. This is very different from wikipedia.
- `It occurs to me that the English Wikipedia has ready access to such a
dataset: it's the totality of revision-deleted and oversighted talk page
posts. The League of Legends team collaborated with outside scientists to
analyze their dataset. I would love to see the Wikimedia Foundation engage
in a similar research project.`
- halfak proposes getting hand labeled set of "aggressiveness"
- Fluffernutter volunteers to help
- Ziko mentions possible spillover effects: see aggressive tone on a talk page and decide you don't want to contribute there
- Pharos wants to know what fraction of talk page comments are bigoted. 1 in 50 discussions would still be significant.

### Revscoring Talk Page
- talk page scoring could be incorporated into ORES

### Harassment Conslutation
- intended to provide a place to discuss ideas, concerns, proposals and possible solutions regarding Wikimedia communities’ harassment-related challenges.


# Wikimedia Data Sources
### Admin Board
- place for reporting incidents on enwiki for administrators
- oversight issues not reflected here. There is an email list
- data is unstructured and would need to manually combed through to find incidents of abuse

### Users Blocked for Harassment
- 11k blocked users have a log comment including "harassment"
- we don't have a diff with comment that resulted in the block
- need to collect span of comments and label them
- some of these may have been deleted or supressed, which may make labeling difficult by third parties

### Edit Filter
- has a "Personal attacks" filter (rule 294). Implementation is private and probably quite simple. Also, it has not been triggered
- has a "Talk page abuse" filter (rule 478). Also private and no hits.
- these fliters could use an eventual API

### RevisionDelete
- RevisionDelete is an administrative feature that allows individual entries in a page history or log to be removed from public view
- RevisionDelete can hide the text of a revision, the username that made the edit or action, or the edit summary or log summary
- The community's decision was that RevisionDelete should not be used without prior clear consensus for "ordinary" incivility, attacks, or for claims of editorial misconduct. The wider community may need to fully review these at the time and in future, even if offensive.
- there are many reasons for RevisionDelete, not just talk page abuse. Should check if deleted talk page revisions tend to be vandalism


### Suppression/Oversight
- Suppression on Wikipedia (also known as oversight for historical reasons) is a form of enhanced deletion which, unlike normal deletion, expunges information from any form of usual access even by administrators
- for removal of non-public personal information
- for removal of potentially libelous information 
- Hiding of blatant attack names on automated lists and logs, where this does not disrupt edit histories. A blatant attack is one obviously intended to denigrate, threaten, libel, insult, or harass someone.

#External Data Sources

### Stanford Politeness Corpus
- described in paper 3 below
- human annotated 2 sentence requests from Wikipedia and Stack Overflow

 

# Related Work

### A Sentiment Analysis Approachfor Online Dispute Detection

- focus on article talk pages
- focus on classifying the entire talk page, not individual comments
- collect dispute corpus using tags: DISPUTED, TOTALLYDISPUTED, DISPUTEDSECTION,
TOTALLYDISPUTED-SECTION, POV set by editors on articles.
- also consider RFC tag on talk pages
- use the entire talk page as a "discussion"?
- use articles without the above tags as discussions without disputes

### Antisocial Behavior in Online Discussion Communities
- user based modeling
- compare features of users who get banned to those who do not
- use disqus discussion comments from CNN, Breitbart, IGN
- undesired behavior mainly measured via post deletion, not content
- they do measure similarity of post to its parent thread (cosine dist), readability index, positive emotion
- if we do modeling on a per post basis, we should consider adding user features as well as content features such as those discussed in the papers classifier
- having community and moderator response features is less helpful since we want to do the prediction before they have to respond
- for low FBUs want a per comment classifier since they could be having a bad day, etc and may still contribute productively at other times

### A computational approach to politeness with application to social factors
- We choose to annotate requests containing exactly
two sentences, where the second sentence
is the actual request (and ends with a question
mark)
- For each request, the annotator
had to indicate how polite she perceived the
request to be by using a slider with values ranging
from “very impolite” to “very polite”
- split requests into sets of 13 and had each set scored 5 times. Z-score normalized each raters scores and averaged normalized scores across raters to get aggregate "politeness" score
- measure inter-annotator agreement as mean pairwise correlation of z-scored scores
- build 20 linguistic features (lexica and regexes)
- BOW model does pretty well. Adding linguistic features add quite a bit. 
- had data relabeled by humans, to get human performance. Performance is almost as good.
- show that wikipedia amins are more polite before winning election than after winning and more polite before winning that those who run and lose

### Automatic identification of personal insults on social news sites
- generate sentiment labeled data set of news comments
- TLDR

### How Community Feedback Shapes User Behavior