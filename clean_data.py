import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd


# loading data
data = pd.read_csv("imdb_top_1000.csv")

# cosidering following columns
columns_to_consider = ['Series_Title', 'Genre', 'Overview', 'Director',
                        'Star1', 'Star2', 'Star3', 'Star4']
data = data[columns_to_consider]

# combining the columns
data['combined_cols'] = data.apply(lambda row: ' '.join(row[col] for col in columns_to_consider), axis=1)

# cleaning data using nltk
stop_words = set(stopwords.words('english'))

def preprocess_text(input_text):
    # defining stopwords once using NLTK's predefined list
    stops = stopwords.words('english')
    
    # converting input text to lowercase to standardize it
    standardized_text = input_text.lower()
    
    # replacing non-alphanumeric characters with a space
    alphanumeric_text = re.sub('[^a-zA-Z0-9]', ' ', standardized_text)
    
    # tokenizing the text to separate into words
    tokens = word_tokenize(alphanumeric_text)
    
    # excluding stopwords from our list of tokens
    tokens_without_stops = filter(lambda token: token not in stops, tokens)
    
    # rejoining the tokens into a string and return it
    clean_text = " ".join(tokens_without_stops)
    return clean_text

data['combined_cols'] = data['combined_cols'].apply(preprocess_text)

# saving the data
data.to_csv("data_final.csv")
