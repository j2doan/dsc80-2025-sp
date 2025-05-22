# lab.py


import os
import pandas as pd
import numpy as np
import requests
import bs4
import lxml


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def question1():
    """
    NOTE: You do NOT need to do anything with this function.
    The function for this question makes sure you
    have a correctly named HTML file in the right
    place. Note: This does NOT check if the supplementary files
    needed for your page are there!
    """
    # Don't change this function body!
    # No Python required; create the HTML file.
    return

    


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------



def extract_book_links(text):
    soup = bs4.BeautifulSoup(text, features='lxml')
    book_links = []
        
    for book in soup.find_all('li', attrs={'class': 'col-xs-6 col-sm-4 col-md-3 col-lg-3'}):

        star = book.find('p', attrs={'class': ['star-rating Four', 'star-rating Five']})
        price = float(str(book.find('p', attrs={'class': 'price_color'}).contents[0])[2:])

        if star != None and price < 50:
                book_links.append(book.find('a').get('href'))
        
    return book_links

def get_product_info(text, categories):
    soup = bs4.BeautifulSoup(text, features='lxml')

    rate_get = soup.find('div', attrs={'class': 'col-sm-6 product_main'})

    container = soup.find('ul', attrs={'class':'breadcrumb'})

    category = container.find_all('li')[2].find('a').contents[0]

    description = soup.find('meta', attrs={'name':'description'}).get('content')

    rating = [
        rate_get.find('p', attrs={'class':'star-rating One'}),
        rate_get.find('p', attrs={'class':'star-rating Two'}),
        rate_get.find('p', attrs={'class':'star-rating Three'}),
        rate_get.find('p', attrs={'class':'star-rating Four'}),
        rate_get.find('p', attrs={'class':'star-rating Five'})
    ]

    # set so rating is at least 1
    for i in range(len(rating)):
        if rating[i] != None:
            rating = i + 1
            break
    
    rate_converter = {1:'One', 2:'Two', 3:'Three', 4:'Four', 5:'Five'}

    if category in categories:
        product_info = soup.find('table', attrs={'class': 'table table-striped'}).find_all('td')
        title = soup.find('div', attrs={'class': 'col-sm-6 product_main'}).find('h1').contents[0]

        book_dict = {
            'upc': product_info[0].contents[0],
            'Product Type': product_info[1].contents[0],
            'Price (excl. tax)': product_info[2].contents[0],
            'Price: (incl. tax)': product_info[3].contents[0],
            'Tax': product_info[4].contents[0],
            'Avaliability': product_info[5].contents[0],
            'Number of Reviews': product_info[6].contents[0],
            'Category': category,
            'Rating': rate_converter[rating],
            'Description': description,
            'Title': title,

        }

        return book_dict
    
    return

def scrape_books(k, categories):
    df = {
            'upc': [],
            'Product Type': [],
            'Price (excl. tax)': [],
            'Price: (incl. tax)': [],
            'Tax': [],
            'Avaliability': [],
            'Number of Reviews': [],
            'Category': [],
            'Rating': [],
            'Description': [],
            'Title': [],
        }
    
    for i in range(k):
        webpage = requests.get(f'https://books.toscrape.com/catalogue/page-{i+1}.html').text
        book_list = extract_book_links(webpage)
        
        for book in book_list:
            book_link = 'https://books.toscrape.com/catalogue/' + book
            book_page = requests.get(book_link).text

            x = get_product_info(book_page, categories)

            if x != None:
                for key in df.keys():
                    df[key].append(x[key])
    
    return pd.DataFrame(df)


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def stock_history(ticker, year, month):
    
    def end_of_month(y, m):
        if y % 4 == 0 and m == 2:
            return 29
        elif m % 2 == 1:
            return 31
        elif m == 2:
            return 28
        else:
            return 30
        
    start_date = pd.Timestamp(str(year) + '-' + str(month).zfill(2) + '-01')
    end_date = pd.Timestamp(str(year) + '-' + str(month).zfill(2) + '-' + str(end_of_month(year, month)))

    span = list(pd.date_range(start=start_date, end=end_date))

    data = requests.get(f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey=PcMGEuYsDd9FOtYFTCVKMQgxp2OTAvTv&from={str(start_date).split(' ')[0]}&to={(str(end_date)).split(' ')[0]}').json()

    return pd.DataFrame(data['historical'])

def stock_stats(history):
    change = str(round(float((history['close'].iloc[0] - history['close'].iloc[-1]) / history['close'].iloc[-1]) * 100, 3))

    if change[0] != '-':
        change = '+' + change

    total = 0

    for i in range(history.shape[0]):
        total += ((history['high'].iloc[i] + history['low'].iloc[i]) / 2) * history['volume'].iloc[i]

    total /= 1000000000

    total = round(total, 2)

    return (change, str(total) + 'B')


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def get_comments(storyid):
    

    def helper(id, df):
        c = requests.get(f'https://hacker-news.firebaseio.com/v0/item/{id}.json').json()

        if 'kids' not in c.keys():
            if c['type'] != 'story' and 'dead' not in c.keys():
                for key in df.keys():
                    df[key].append(c[key])
            return
        else:
            if c['type'] != 'story' and 'dead' not in c.keys():
                for key in df.keys():
                    df[key].append(c[key])
            for k in c['kids']:
                helper(k, df)

    df = {'id': [], 
          'by': [], 
          'text': [], 
          'parent': [], 
          'time': []}
    
    results = requests.get(f'https://hacker-news.firebaseio.com/v0/item/{storyid}.json').json()

    recurse = helper(storyid, df)

    df = pd.DataFrame(df)

    df['time'] = df.time.apply(lambda x: pd.Timestamp(x, unit='s'))

    return df



            
