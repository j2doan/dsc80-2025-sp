# lab.py


import pandas as pd
import numpy as np
import os
import re


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def match_1(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_1("abcde]")
    False
    >>> match_1("ab[cde")
    False
    >>> match_1("a[cd]")
    False
    >>> match_1("ab[cd]")
    True
    >>> match_1("1ab[cd]")
    False
    >>> match_1("ab[cd]ef")
    True
    >>> match_1("1b[#d] _")
    True
    """
    pattern = r"^.{2}\[.{2}\].*$"

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_2(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_2("(123) 456-7890")
    False
    >>> match_2("858-456-7890")
    False
    >>> match_2("(858)45-7890")
    False
    >>> match_2("(858) 456-7890")
    True
    >>> match_2("(858)456-789")
    False
    >>> match_2("(858)456-7890")
    False
    >>> match_2("a(858) 456-7890")
    False
    >>> match_2("(858) 456-7890b")
    False
    """
    pattern = r"^\(858\) \d{3}-\d{4}$"

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_3(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_3("qwertsd?")
    True
    >>> match_3("qw?ertsd?")
    True
    >>> match_3("ab c?")
    False
    >>> match_3("ab   c ?")
    True
    >>> match_3(" asdfqwes ?")
    False
    >>> match_3(" adfqwes ?")
    True
    >>> match_3(" adf!qes ?")
    False
    >>> match_3(" adf!qe? ")
    False
    """
    pattern = r"^[A-Za-z0-9\s?]{5,9}\?$"

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_4(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_4("$$AaaaaBbbbc")
    True
    >>> match_4("$!@#$aABc")
    True
    >>> match_4("$a$aABc")
    False
    >>> match_4("$iiuABc")
    False
    >>> match_4("123$$$Abc")
    False
    >>> match_4("$$Abc")
    True
    >>> match_4("$qw345t$AAAc")
    False
    >>> match_4("$s$Bca")
    False
    >>> match_4("$!@$")
    False
    """
    pattern = r"^\$[^abc$]*\$[aA]+[bB]+[cC]+$"

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_5(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_5("dsc80.py")
    True
    >>> match_5("dsc80py")
    False
    >>> match_5("dsc80..py")
    False
    >>> match_5("dsc80+.py")
    False
    """
    pattern = r"^[a-z0-9_]+\.py$"

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_6(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_6("aab_cbb_bc")
    False
    >>> match_6("aab_cbbbc")
    True
    >>> match_6("aab_Abbbc")
    False
    >>> match_6("abcdef")
    False
    >>> match_6("ABCDEF_ABCD")
    False
    """
    pattern = r"^[a-z]+\_[a-z]+$"

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_7(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_7("_abc_")
    True
    >>> match_7("abd")
    False
    >>> match_7("bcd")
    False
    >>> match_7("_ncde")
    False
    """
    pattern = r"^\_.*\_$"

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None



def match_8(string):
    """
    DO NOT EDIT THE DOCSTRING!
    >>> match_8("ASJDKLFK10ASDO")
    False
    >>> match_8("ASJDKLFK0ASDo!!!!!!! !!!!!!!!!")
    True
    >>> match_8("JKLSDNM01IDKSL")
    False
    >>> match_8("ASDKJLdsi0SKLl")
    False
    >>> match_8("ASDJKL9380JKAL")
    True
    """
    pattern = r"^[^Oi1]+$"

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None



def match_9(string):
    '''
    DO NOT EDIT THE DOCSTRING!
    >>> match_9('NY-32-NYC-1232')
    True
    >>> match_9('ca-23-SAN-1231')
    False
    >>> match_9('MA-36-BOS-5465')
    False
    >>> match_9('CA-56-LAX-7895')
    True
    >>> match_9('NY-32-LAX-0000') # If the state is NY, the city can be any 3 letter code, including LAX or SAN!
    True
    >>> match_9('TX-32-SAN-4491')
    False
    '''
    pattern = r"^(CA-\d{2}-(SAN|LAX)-\d{4}|NY-\d{2}-[A-Za-z]{3}-\d{4})$"

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_10(string):
    '''
    DO NOT EDIT THE DOCSTRING!
    >>> match_10('ABCdef')
    ['bcd']
    >>> match_10(' DEFaabc !g ')
    ['def', 'bcg']
    >>> match_10('Come ti chiami?')
    ['com', 'eti', 'chi']
    >>> match_10('and')
    []
    >>> match_10('Ab..DEF')
    ['bde']
    
    '''
    s = string.lower()
    
    s = re.sub(r'[^\w]|a', '', s)
        
    return re.findall(r'.{3}', s)


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def extract_personal(s):
    email = r'\b[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]{2,}\b'
    ssn = r'\b\d{3}-\d{2}-\d{4}\b'
    bitcoin = r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b'
    address = r'\b\d+\s+(?:[A-Z][a-z]*\s)*?(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard)\b'

    emails = [e for e in re.findall(email, s) if e]
    ssns = [s for s in re.findall(ssn, s) if s]
    btc = [b for b in re.findall(bitcoin, s) if b]
    addrs = [a for a in re.findall(address, s) if a]

    return (emails, ssns, btc, addrs)


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def tfidf_data(reviews_ser, review):
    x = pd.Series(review.split()).value_counts()

    df = x.rename("cnt").to_frame()

    df['tf'] = df['cnt'] / len(review)

    idf_vals = []

    for word in df.index:
        doc_count = reviews_ser.str.contains(rf'\b{word}\b', regex=True).sum()
        idf = np.log(len(reviews_ser) / (1 + doc_count))  # add 1 to avoid div by 0
        idf_vals.append(idf)

    df['idf'] = idf_vals

    df['tfidf'] = df['tf'] * df['idf']

    return df


def relevant_word(out):
    return out['tfidf'].idxmax()


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def hashtag_list(tweet_text):
    return tweet_text.apply(
        lambda tweet: [tag[1:] for tag in re.findall(r'#\S+', tweet)]
    )


def most_common_hashtag(tweet_lists):
    hashtag_counts = tweet_lists.explode().value_counts()

    def most_common(tags):
        if not tags:
            return np.nan
        elif len(tags) == 1:
            return tags[0]
        else:
            return max(tags, key=lambda tag: hashtag_counts.get(tag, 0))

    return tweet_lists.apply(most_common)


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def create_features(ira):

    ira = ira['text']

    x = hashtag_list(ira)
    num_hashtags = x.apply(len)

    mc_hashtags = most_common_hashtag(x)

    num_tags = ira.apply(lambda t: len(re.findall(r'@\w+', t)))

    num_links = ira.apply(lambda t: len(re.findall(r'https?://\S+', t)))

    is_retweet = ira.str.startswith("RT")

    def clean_text(t):
        t = re.sub(r'(https?://\S+|@\w+|#\S+|\bRT\b)', ' ', t)
        t = re.sub(r'[^A-Za-z0-9 ]+', ' ', t)
        t = t.lower()
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    cleaned_text = ira.apply(clean_text)

    result = pd.DataFrame({
        'text': cleaned_text,
        'num_hashtags': num_hashtags,
        'mc_hashtags': mc_hashtags,
        'num_tags': num_tags,
        'num_links': num_links,
        'is_retweet': is_retweet
    })

    result.index = ira.index
    return result
