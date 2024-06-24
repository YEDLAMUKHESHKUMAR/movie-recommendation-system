# movie-recommendation-system
## Objective
* Build a movie recommendation system that suggests similar movies based on user input.
## Data Source
* The dataset used for this project is sourced from YBI Foundation's Movies Recommendation dataset.
## Import Libraries
```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
```
## Import Data
```python
# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Movies%20Recommendation.csv')
```
## Describe Data
```python
# Display first few rows of the dataset
print(df.head())

# Display data information
print(df.info())
```
## Data Visualization
* This section can include exploratory data analysis (EDA) and visualizations if applicable.
## Data Preprocessing
```python
# Select relevant features
df_features = df[['Movie_Genre', 'Movie_Keywords', 'Movie_Tagline', 'Movie_Cast', 'Movie_Director']].fillna('')

# Combine text features into a single column
df_features['combined_features'] = df_features.apply(lambda x: ' '.join(x), axis=1)

# Convert text to TF-IDF vectors
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df_features['combined_features'])

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(X)
```
## Define Target Variable (y) and Feature Variables (X)
* In this project, there's no traditional target variable (y) as it's unsupervised learning. The feature variables (X) are the combined TF-IDF vectors of movie attributes.
## Train Test Split
* Not applicable for this project as it's unsupervised.
## Modeling
* No traditional modeling; the cosine similarity is used for finding similar movies.
## Model Evaluation
* Evaluate based on user input and similarity scores.
## Prediction
```python
# Example of recommending movies
favorite_movie_name = input('Enter your favorite movie name: ')
all_movies_titles = df['Movie_Title'].tolist()
movie_recommendations = difflib.get_close_matches(favorite_movie_name, all_movies_titles)

if movie_recommendations:
    closest_match = movie_recommendations[0]
    index_of_close_match = df[df['Movie_Title'] == closest_match].index[0]
    recommendation_scores = list(enumerate(similarity_matrix[index_of_close_match]))
    sorted_similar_movies = sorted(recommendation_scores, key=lambda x: x[1], reverse=True)

    print('Top 10 Movies Suggested for You: \n')
    for i, movie in enumerate(sorted_similar_movies[:10], 1):
        similar_movie_title = df.iloc[movie[0]]['Movie_Title']
        print(f"{i}. {similar_movie_title}")
else:
    print('No close match found.')
```
## Output Example
```python
Enter your favorite movie name: Avatar
Top 10 Movies Suggested for You:

1. Avatar
2. The Girl on the Train
3. Act of Valor
4. Donnie Darko
5. Precious
6. Freaky Friday
7. The Opposite Sex
8. Heaven is for Real
9. Run Lola Run
10. Elizabethtown

```
## Explanation
* This project demonstrates how to build a basic movie recommendation system using content-based filtering and cosine similarity. It suggests movies similar to a user-provided favorite movie based on movie attributes like genre, keywords, cast, and director.
