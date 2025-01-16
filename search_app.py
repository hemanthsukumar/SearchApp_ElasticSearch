import os
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import streamlit as st
from requests.auth import HTTPBasicAuth

# model loading
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

# Adjust the Elasticsearch connection for HTTPS and security
es = Elasticsearch(
    ['https://localhost:9200'],  # Use HTTPS for secure connections
    http_auth=('elastic', 'G3Er96kd-iC8XA_=zQHa'),  # Authentication credentials
    #use_ssl=True,  # Enable SSL
    verify_certs=True,  # Verify SSL certificates
    ca_certs='/Users/hemanthsukumar/Downloads/elasticsearch-8.13.0/config/certs/http_ca.crt'  # Path to CA cert, adjust as necessary
    # Note: For development, you might set verify_certs=False to bypass verification, but this is not recommended for production.
)

# streamlit app layout
st.title('Movies Search! Find movies here...')

# sidebar for search options
st.sidebar.header('Search Options')
num_results = st.sidebar.slider('Number of Results', min_value=5, max_value=20, value=10)

# main search box
user_query = st.text_input("Enter your search query:", "")

if user_query:
    # embed the user query
    query_vector = model.encode([user_query])[0].tolist()
    
    # elasticsearch query
    try:
        response = es.search(index="imdb", body={
            "size": num_results,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            }
        })
        
        # showing results
        st.header("Results")
        results = response['hits']['hits']
        
        for result in results:
            title = result['_source']['Series_Title']
            overview = result['_source']['Overview']
            director = result['_source']['Director']
            star1 = result['_source']['Star1']
            star2 = result['_source']['Star2']
            star3 = result['_source']['Star3']
            star4 = result['_source']['Star4']
            score = result['_score']
            
            # each result block
            with st.container():
                st.subheader(f"{title} (Score: {score:.2f})")
                st.write(overview)
    
                with st.expander("See more"):
                    st.write(f"This is directed by {director} and features {star1}, {star2}, {star3}, {star4}")
    
            st.markdown("---")  # separator
    except Exception as e:
        st.error(f"An error occurred: {e}")

# footer
st.sidebar.markdown("### About This App")
st.sidebar.markdown("This will search the movie based on the IMDB dataset.")
