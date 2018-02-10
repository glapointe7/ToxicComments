import numpy as np
import pandas
import re
import string


## Faster with Series than Counter.
def GetWordFrequency(train_comments):
    return pandas.Series(' '.join(train_comments).lower().split()).value_counts()

def ascii_chars_from_text(text, ascii_chars):
    for char in text:
        if char in ascii_chars:
            yield char


def clean_text(comments):
    train_comments_cleaned = []
    ascii_chars = set(string.printable)
    
    for comment in comments:
        ## Convert to lower case , so that Hi and hi are the same.
        comment = comment.lower().strip(' ')
        ## Remove carriage returns.
        comment = re.sub("\\n", " ", comment)
        ## Remove leaky elements like IP addresses.
        #comment = re.sub("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}", "", comment)
        ## Remove usernames.
        comment = re.sub("\[\[.*\]", "", comment)
        comment = re.sub("[\$\*&%#@\"]", " ", comment)
        ## Remove punctuation.
        comment = re.sub('\W', ' ', comment)
        ## Remove non ascii characters.
        comment = ''.join([char for char in ascii_chars_from_text(comment, ascii_chars)])
        comment = re.sub("fck", "fuck", comment)
        comment = re.sub("f ck", "fuck", comment)
        comment = re.sub("fagget", "faggot", comment)
        comment = re.sub("you re", "you are", comment)
        comment = re.sub("\d", " ", comment)

        train_comments_cleaned.append(comment)
        
    return pandas.Series(train_comments_cleaned).astype(str)

def clean(train_perspective):
    classes = ['comment', 'is_toxic']
    train = train_perspective.loc[:, classes]

    train.comment = clean_text(train.comment)
    train.is_toxic = train.is_toxic.astype(np.int64)
    
    return train