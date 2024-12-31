import sys
import json
import pandas as pd

# Load the preprocessed dataset
df = pd.read_csv('processed_courses.csv')

# Function to recommend courses based on a single user preference
def recommend_user_preference_single(preferred_subject, num_recommendations=5):
    recommended_courses = df[df['subject'] == preferred_subject].sort_values(
        by='published_timestamp', ascending=False
    ).head(num_recommendations)
    
    if recommended_courses.empty:
        return {"error": f"No courses found for subject: {preferred_subject}"}
    
    return recommended_courses[['course_id', 'course_title', 'subject', 'price', 'published_timestamp']].to_dict(orient='records')

try:
    # Prompt user for input
    preferred_subject = input("Enter preferred subject: ")
    num_recommendations = int(input("Enter number of recommendations (default is 5): ") or 5)
    if not preferred_subject:
        raise ValueError("Preferred subject is required.")
    
    recommendations = recommend_user_preference_single(preferred_subject, num_recommendations)
    print(json.dumps(recommendations, indent=4), flush=True)
except ValueError as e:
    print(f"Error: {e}", flush=True)
    sys.exit(1)
