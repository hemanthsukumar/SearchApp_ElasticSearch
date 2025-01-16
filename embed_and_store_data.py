# loading the libraries
import pandas as pd
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer


# loading the clean data
data = pd.read_csv("data_final.csv")

# loding the model
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

# creating the required embeddings using the model
model_embeddings = model.encode(data['combined_cols'].tolist())

# creating the elasicsearch instance to index the data
es = Elasticsearch(
    ['https://localhost:9200'],
    http_auth=('elastic', 'G3Er96kd-iC8XA_=zQHa'),
    verify_certs=True,
    ca_certs='/Users/hemanthsukumar/Downloads/elasticsearch-8.13.0/config/certs/http_ca.crt'
)
es.indices.delete(index='movies', ignore=[400, 404])

# ['Series_Title', 'Genre', 'Overview', 'Director',
#                         'Star1', 'Star2', 'Star3', 'Star4']

mapping_properties = {
            "Series_Title": {"type": "text"},
            "Genre": {"type": "text"},
            "Overview": {"type": "text"},
            "Director": {"type": "text"},
            "Star1": {"type": "text"},
            "Star2": {"type": "text"},
            "Star3": {"type": "text"},
            "Star4": {"type": "dense_vector", "dims": model_embeddings.shape[1]}
        }

mapping = {
    "mappings": {
        "properties": mapping_properties
    }
}

# adding the index
es.indices.create(index='movies', body=mapping)
for i, row in data.iterrows():
    es.index(index='imdb', id=i, body={
        'Series_Title': row['Series_Title'],
        'Genre': row['Genre'],
        'Overview': row['Overview'],
        'Director': row['Director'],
        'Star1': row['Star1'],
        'Star2': row['Star2'],
        'Star3': row['Star3'],
        'Star4': row['Star4'],
        'embedding': model_embeddings[i].tolist()
    })
