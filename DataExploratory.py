import pandas
import re
import string


## Faster with Series than Counter.
def GetWordFrequency(train_comments):
    return pandas.Series(' '.join(train_comments).lower().split()).value_counts()

def remove_non_ascii_chars(text, ascii_chars):
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
        comment = re.sub("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}", "", comment)
        ## Remove usernames.
        comment = re.sub("\[\[.*\]", "", comment)
        ## Remove special characters.
        comment = re.sub("[“”¨«»®´·º½¾¿¡§£₤‘’]", "", comment)
        ## Remove punctuation.
        comment = re.sub('\W', ' ', comment)
        comment = re.sub('\s+', ' ', comment)
        ## Remove non ascii characters.
        comment = ''.join([char for char in remove_non_ascii_chars(comment, ascii_chars)])
        
        #translation = str.maketrans("", "", punctuation)
        #comment = comment.translate(translation)

        train_comments_cleaned.append(comment)
        
    return pandas.Series(train_comments_cleaned).astype(str)