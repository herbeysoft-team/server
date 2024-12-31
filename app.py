from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

# Create the Flask app
app = Flask(__name__)

# Load the model and label encoder
model = joblib.load('quiz_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
# Load the preprocessed dataset
df = pd.read_csv('processed_courses.csv')
# Load the TF-IDF vectorizer and matrix (assuming these are already saved)
tfidf = joblib.load('tfidf_vectorizer.pkl')
tfidf_matrix = joblib.load('tfidf_matrix.pkl')

columns_to_include = ['course_title', 'subject', 'published_timestamp', 'course_id']
df['search_content'] = df[columns_to_include].astype(str).apply(' '.join, axis=1)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['search_content'])

# Function to recommend courses based on a single user preference
def recommend_user_preference_single(preferred_subject, num_recommendations=5):
    recommended_courses = df[df['subject'] == preferred_subject].sort_values(
        by='published_timestamp', ascending=False
    ).head(num_recommendations)
    
    if recommended_courses.empty:
        return {"error": f"No courses found for subject: {preferred_subject}"}
    
    return recommended_courses[['course_id', 'course_title', 'subject', 'price', 'url',
                                'num_subscribers', 'num_reviews', 'published_timestamp']].to_dict(orient='records')

# Function to recommend courses based on a search query
def recommend_search_based(query, num_recommendations=5):
    query_vector = tfidf.transform([query])  # Convert query into the same vector space
    cosine_sim = linear_kernel(query_vector, tfidf_matrix).flatten()  # Compute similarity scores
    top_indices = cosine_sim.argsort()[-num_recommendations:][::-1]  # Get top matching indices
     # Select top courses based on similarity
    recommended_courses = df.iloc[top_indices][['course_id', 'course_title', 'subject', 'price', 'url',
                                                  'num_subscribers', 'num_reviews', 'published_timestamp']]

    # Add similarity scores to the recommended courses
    recommended_courses['similarity_score'] = cosine_sim[top_indices]

    # Sort first by 'published_timestamp' (latest first), then by 'num_subscribers' (highest first)
    sorted_courses = recommended_courses.sort_values(by='similarity_score', ascending=False)

    # Convert the result to a dictionary, including the similarity scores
    return sorted_courses[['course_id', 'course_title', 'subject', 'price', 'url', 'num_subscribers', 
                           'num_reviews', 'published_timestamp', 'similarity_score']].to_dict(orient='records')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON input
        data = request.get_json()
        print(data)
        if not data or 'inputArray' not in data:
            return jsonify({"error": "Invalid input. Expected JSON with 'inputArray'."}), 400
        
        input_data = data['inputArray']
        print(f"Received input: {input_data}", flush=True)

        # Convert input data to a DataFrame
        input_df = pd.DataFrame([input_data], columns=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        
        # Perform prediction
        prediction = model.predict(input_df)
        decoded_prediction = label_encoder.inverse_transform(prediction)
        
        # Convert the prediction to a Python native type
        result = str(decoded_prediction[0])  # Ensure it's serializable
        
        # Send the result as JSON
        return jsonify({"result": result})

    except Exception as e:
        print(f"Error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500
    
@app.route('/preference', methods=['POST'])
def preference():
    try:
        # Parse JSON input
        data = request.get_json()
        print(data)
        if not data or 'preferred_subject' not in data:
            return jsonify({"error": "Invalid input. Expected JSON with 'preferred_subject'."}), 400
        
        preferred_subject = data['preferred_subject']
        num_recommendations = data.get('num_recommendations', 5)  # Default to 5 recommendations
        print(f"Preferred subject: {preferred_subject}, Number of recommendations: {num_recommendations}")

        # Get recommendations
        recommendations = recommend_user_preference_single(preferred_subject, num_recommendations)
        return jsonify({"recommendations": recommendations})

    except Exception as e:
        print(f"Error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500
    
@app.route('/search', methods=['POST'])
def search():
    try:
        # Parse JSON input
        data = request.get_json()
        print(data)
        if not data or 'query' not in data:
            return jsonify({"error": "Invalid input. Expected JSON with 'query'."}), 400
        
        query = data['query']
        num_recommendations = data.get('num_recommendations', 5)  # Default to 5 recommendations
        print(f"Search query: {query}, Number of recommendations: {num_recommendations}")

        # Get search-based recommendations
        recommendations = recommend_search_based(query, num_recommendations)
        return jsonify({"recommendations": recommendations})

    except Exception as e:
        print(f"Error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500
    
# For local testing (optional)
if __name__ == '__main__':
    app.run()
