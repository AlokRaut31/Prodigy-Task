import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources (only once needed)
nltk.download('stopwords')
nltk.download('wordnet')

# Step 1: Load Dataset
file_path = r"C:\Users\Alok Kumar\Desktop\DATA SCIENCE\PYTHON\twitter.csv"
df = pd.read_csv(file_path, usecols=[0, 1, 2, 3], names=['ID', 'Topic', 'Sentiment', 'Text'], header=0)

# Drop rows with missing 'Text'
df.dropna(subset=['Text'], inplace=True)

# Step 2: Clean Text
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"http\S+|@\S+|#[A-Za-z0-9_]+", "", str(text))  # Remove URLs, mentions, hashtags
    text = re.sub(r"[^a-zA-Z ]", " ", text)  # Remove non-letter characters
    words = text.lower().split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

df['Cleaned_Text'] = df['Text'].apply(clean_text)

# Step 3: Plot Sentiment Distribution (fixed palette warning by adding hue)
plt.figure(figsize=(8, 5))
sns.countplot(x='Sentiment', data=df, hue='Sentiment', palette='pastel', legend=False)
plt.title("Sentiment Distribution", fontsize=14)
plt.xlabel("Sentiment", fontsize=12)
plt.ylabel("Tweet Count", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# Step 4: Word Cloud of All Cleaned Text
all_words = " ".join(df['Cleaned_Text'])

wordcloud = WordCloud(
    width=1000, 
    height=500, 
    background_color='white', 
    colormap='viridis',
    max_words=200
).generate(all_words)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Most Common Words in Tweets", fontsize=16)
plt.tight_layout()
plt.show()
