def comment_lengths(comments):
    for comment in comments:
        yield len(comment)
        

def comments_capitals_counter(comments):
    for comment in comments:
        yield sum(1 for char in comment if char.isupper())
        
        
def ratio_feature_vs_length(feature_rows, comment_lengths):
    for feature, comment_length in zip(feature_rows, comment_lengths):
        yield float(feature) / float(comment_length)
        
        
def comments_char_counter(comments, char):
    for comment in comments:
        yield comment.count(char)
        
        
def comments_chars_counter(comments, chars):
    for comment in comments:
        yield sum(comment.count(char) for char in chars)
        
        
def all_words_counter(comments):
    for comment in comments:
        yield len(comment.split())
        
        
def unique_words_counter(comments):
    for comment in comments:
        yield len(set(word for word in comment.split()))