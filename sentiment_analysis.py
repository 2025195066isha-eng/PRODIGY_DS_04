import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# 1. Load Dataset
url = "https://raw.githubusercontent.com/Prodigy-InfoTech/data-science-datasets/main/Task%204/twitter_training.csv"
column_names = ['tweet_id', 'entity', 'sentiment_label', 'tweet_text']

df = pd.read_csv(url, header=None, names=column_names)

# Clean missing values
df.dropna(subset=['tweet_text'], inplace=True)
df['tweet_text'] = df['tweet_text'].astype(str)

# 2. Compute Sentiment Scores using VADER
def get_vader_sentiment(text):
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['vader_sentiment'] = df['tweet_text'].apply(get_vader_sentiment)

# 3. Visualization
sns.set_theme(style="whitegrid")
plt.figure(figsize=(14, 6))

# Plot 1: Overall Sentiment Distribution
plt.subplot(1, 2, 1)
sns.countplot(data=df, x='vader_sentiment', order=['Positive', 'Neutral', 'Negative'], palette='viridis')
plt.title('Overall Sentiment Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Sentiment Class')
plt.ylabel('Count')

# Plot 2: Sentiment Patterns across Top Entities/Topics
top_entities = df['entity'].value_counts().head(5).index
filtered_df = df[df['entity'].isin(top_entities)]

plt.subplot(1, 2, 2)
sns.countplot(data=filtered_df, x='entity', hue='vader_sentiment', palette='viridis')
plt.title('Sentiment Pattern by Top Topics/Brands', fontsize=14, fontweight='bold')
plt.xlabel('Topic / Brand')
plt.ylabel('Count')
plt.xticks(rotation=30)
plt.legend(title='Sentiment')

plt.tight_layout()
plt.show()
