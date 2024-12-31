import sys
import json
import pandas as pd
import joblib
from sklearn.metrics.pairwise import linear_kernel

# Load the preprocessed dataset and TF-IDF vectorizer
df = pd.read_csv('processed_courses.csv')
tfidf = joblib.load('tfidf_vectorizer.pkl')
tfidf_matrix = joblib.load('tfidf_matrix.pkl')  # Save TF-IDF matrix for efficient computation

# Function to recommend courses based on a search query
def recommend_search_based(query, num_recommendations=5):
    query_vector = tfidf.transform([query])  # Convert query into the same vector space
    cosine_sim = linear_kernel(query_vector, tfidf_matrix).flatten()  # Compute similarity scores
    top_indices = cosine_sim.argsort()[-num_recommendations:][::-1]  # Get top matching indices
    
    return df.iloc[top_indices][['course_id', 'course_title', 'subject', 'price', 'url',
                                'num_subscribers', 'num_reviews', 'published_timestamp']].to_dict(orient='records')

try:
    # Parse the input JSON from command-line arguments
    input_data = json.loads(sys.argv[1])  # Example input: {"query": "Python programming", "num_recommendations": 5}
    
    query = input_data.get('query')
    num_recommendations = input_data.get('num_recommendations', 5)  # Default to 5 recommendations if not specified
    
    # Validate input
    if not query:
        raise ValueError("Search query is required.")
    
    # Get recommendations
    recommendations = recommend_search_based(query, num_recommendations)
    print(json.dumps(recommendations), flush=True)

except (json.JSONDecodeError, ValueError) as e:
    print(f"Error: {e}", flush=True)
    sys.exit(1)
