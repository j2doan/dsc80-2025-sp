# project.py


import pandas as pd
import numpy as np
from pathlib import Path
import re
import requests
import time
from itertools import chain


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
    """
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
    """

    
    x = re.split(r'(\n\n)+', book_string)

    non_empty = filter(lambda x: x.strip(), x)

    x = [['\x02'] + re.findall(r'\w+|[^\w\s]', x) + ['\x03'] for x in non_empty if x.strip()]

    return (list(chain.from_iterable(x)))

    


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
        lst = []

        for i in range(len(tokens) - self.N + 1):
            chunk = tuple(tokens[i:i + self.N])
            lst.append(chunk)
        
        return lst
        
    def train(self, ngrams):
        
        df = pd.DataFrame({'ngram':ngrams})
        df['count_x'] = df.groupby('ngram')['ngram'].transform('count')
        df['n1gram'] = df['ngram'].apply(lambda x: x[:-1])
        df['count_y'] = df.groupby('n1gram')['n1gram'].transform('count')

        df['prob'] = df['count_x'] / df['count_y']

        df = df[['ngram', 'n1gram', 'prob']].sort_values(by='prob').set_index('ngram').reset_index()

        df = df.drop_duplicates()

        return df
    
    def probability(self, words):
        words = tokenize(' '.join(words))

        words = words[1:-1]

        result = []

        for k in range(1, self.N):
            full_ngram = tuple(words[:k])
            context = full_ngram[:-1] if k > 1 else None
            result.append((full_ngram, context))

        for i in range(self.N - 1, len(words)):
            full_ngram = tuple(words[i - self.N + 1 : i + 1])
            context = full_ngram[:-1]
            result.append((full_ngram, context))
            
        product = 1
        
        for i in result:

            length = len(i[0])

            if length == self.N:
                df = self.mdl
            else:
                df = self.prev_mdl
                for j in range(self.N - length - 1):
                    df = df.prev_mdl
                df = df.mdl

            # checks if 2nd element is None
            if i[1] == None:
                p = df.loc[i[0]]

            else:
                p = df[(df["ngram"] == i[0]) & (df["n1gram"] == i[1])]['prob']
                if len(p) == 0:
                    p = 0
                else:
                    p = p.iloc[0]
            
            product *= p

        return product


    def sample(self, M):
        result = ['\x02']  # Start of sentence token
        tokens_generated = 0

        while tokens_generated < M:
            found = False
            n = self.N
            context = tuple(result[-(n - 1):]) if n > 1 else None
            model = self 

            while n > 1:
                df = model.mdl
                options = df[df['n1gram'] == context] if context else df.copy()

                if not options.empty:
                    next_token = np.random.choice(
                        options['ngram'].apply(lambda x: x[-1]),
                        p=options['prob'] / options['prob'].sum()
                    )
                    result.append(next_token)
                    tokens_generated += 1
                    found = True
                    break
                else:
                    model = model.prev_mdl
                    n -= 1
                    context = tuple(result[-(n - 1):]) if n > 1 else None

            if not found:
                # unigram model as last resort
                if isinstance(model, UnigramLM) or isinstance(model, UniformLM):
                    next_token = np.random.choice(
                        model.mdl.index,
                        p=model.mdl.values
                    )
                    result.append(next_token)
                    tokens_generated += 1
                else:
                    # nothing found even in unigram, then STOP token
                    result.append('\x03')
                    tokens_generated += 1

            if result[-1] == '\x03':
                break

        return ' '.join(result)

# my code ends at the first \x03, it should continue, like add next \x02