import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Download VADER lexicon if not already present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
# Generate synthetic data for restaurant reviews and ratings
num_restaurants = 100
data = {
    'Restaurant': [f'Restaurant {i+1}' for i in range(num_restaurants)],
    'Rating': np.random.uniform(2.5, 5, num_restaurants),
    'Review': [f'This restaurant is {adjective}! The food was {quality} and the service was {service}.'
               for adjective in np.random.choice(['amazing', 'good', 'okay', 'bad', 'terrible'], num_restaurants)
               for quality in np.random.choice(['delicious', 'average', 'poor'], num_restaurants)
               for service in np.random.choice(['excellent', 'adequate', 'slow'], num_restaurants)],
    'Price_Range': np.random.choice(['$', '$$', '$$$'], num_restaurants),
    'Cuisine': np.random.choice(['Italian', 'Mexican', 'Chinese', 'American'], num_restaurants)
}
df = pd.DataFrame(data)
# --- 2. Sentiment Analysis ---
analyzer = SentimentIntensityAnalyzer()
df['Sentiment_Score'] = df['Review'].apply(lambda review: analyzer.polarity_scores(review)['compound'])
# --- 3. Data Cleaning (if needed) ---
# In a real-world scenario, this section would handle missing values, outliers, etc.
# For this synthetic data, no cleaning is explicitly needed.
# --- 4. Analysis ---
# Calculate the correlation between rating and sentiment score
correlation = df['Rating'].corr(df['Sentiment_Score'])
print(f"Correlation between Rating and Sentiment Score: {correlation}")
# Group by cuisine and calculate average rating and sentiment
cuisine_stats = df.groupby('Cuisine')[['Rating', 'Sentiment_Score']].mean()
print("\nAverage Rating and Sentiment Score by Cuisine:")
print(cuisine_stats)
# --- 5. Visualization ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Sentiment_Score', y='Rating', data=df, hue='Price_Range')
plt.title('Restaurant Rating vs. Sentiment Score')
plt.xlabel('Sentiment Score')
plt.ylabel('Rating')
plt.grid(True)
plt.tight_layout()
plt.savefig('sentiment_rating_scatter.png')
print("Plot saved to sentiment_rating_scatter.png")
plt.figure(figsize=(10, 6))
sns.barplot(x=cuisine_stats.index, y=cuisine_stats['Rating'])
plt.title('Average Rating by Cuisine')
plt.xlabel('Cuisine')
plt.ylabel('Average Rating')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('average_rating_cuisine.png')
print("Plot saved to average_rating_cuisine.png")