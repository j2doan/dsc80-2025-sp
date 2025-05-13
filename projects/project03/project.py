# project.py


import pandas as pd
import numpy as np
from pathlib import Path
import re
import requests
import time


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_book(url):
    raw = requests.get(url)
    text = raw.text
    time.sleep(0.5)

    title_start = text.find('Title: ')
    title_finding = text[title_start + 7:]
    title_end = title_finding.find('\r')
    title = title_finding[:title_end]

    start_token = f'*** START OF THE PROJECT GUTENBERG EBOOK {title.upper()} ***'
    end_token = f'*** END OF THE PROJECT GUTENBERG EBOOK {title.upper()} ***'

    start_index = text.find(start_token)
    end_index = text.find(end_token)

    book_content = text[start_index + len(start_token): end_index]

    book_content = book_content.replace('\r\n', '\n')

    return book_content


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(book_string):
    paragraphs = re.split(r'(?:\n){2,}', book_string.strip())

    tokens = ['\x02']
        
    token_pattern = re.compile(r"[\w'-]+|[^\w\s]")
        
    for paragraph in paragraphs:
        
        paragraph_tokens = token_pattern.findall(paragraph)
            
        tokens.extend(paragraph_tokens)
            
        tokens.append('\x03')
            
        tokens.append('\x02')

    tokens.pop()

    return tokens


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):


    def __init__(self, tokens):

        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        unique = pd.Series(tokens).drop_duplicates(keep='first')

        probability = 1 / len(unique)

        s = pd.Series(probability, index=unique)

        return s
    
    def probability(self, words):
        product = 1
        for token in words:
            product *= self.mdl.get(token, 0)
        return product
        
    def sample(self, M):
        return ' '.join(np.random.choice(self.mdl.index, size=M, replace=True, p=self.mdl.values))


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


class UnigramLM(object):
    
    def __init__(self, tokens):
        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        s = pd.Series(tokens)

        count_series = s.value_counts()

        frequency_series = count_series / len(tokens)

        unique_tokens = pd.Series(tokens).drop_duplicates(keep='first')  # Keeps the first occurrence of each token
        order_preserved = frequency_series.loc[unique_tokens]

        return order_preserved
    
    def probability(self, words):
        product = 1
        for token in words:
            product *= self.mdl.get(token, 0)
        return product
        
    def sample(self, M):
        return ' '.join(np.random.choice(self.mdl.index, size=M, replace=True, p=self.mdl.values))


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


class NGramLM(object):
    
    def __init__(self, N, tokens):
        # You don't need to edit the constructor,
        # but you should understand how it works!
        
        self.N = N

        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)

    def create_ngrams(self, tokens):
        ...
        
    def train(self, ngrams):
        ...
    
    def probability(self, words):
        ...
    

    def sample(self, M):
        ...
