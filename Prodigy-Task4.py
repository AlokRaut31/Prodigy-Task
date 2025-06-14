import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

file_path = r"C:\Users\Alok Kumar\Desktop\DATA SCIENCE\PYTHON\twitter.csv"



# -------- Step 1: Load CSV --------
df = pd.read_csv(file_path, usecols=[0, 1, 2, 3], names=['ID', 'Topic', 'Sentiment', 'Text'], header=0)

# Drop rows with missing text
df.dropna(subset=['Text'], inplace=True)

# -------- Step 2: Clean Text --------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"http\S+|@\S+|#[A-Za-z0-9_]+", "", str(text))  # Remove URLs, mentions, hashtags
    text = re.sub(r"[^a-zA-Z ]", "", text)  # Remove non-letter characters
    words = text.lower().split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

df['Cleaned_Text'] = df['Text'].apply(clean_text)

# -------- Step 3: Plot Sentiment Distribution --------
plt.figure(figsize=(8, 5))
sns.countplot(x='Sentiment', data=df, palette='pastel')
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# -------- Step 4: Generate Word Cloud --------
all_words = " ".join(text for text in df['Cleaned_Text'] if text)

wordcloud = WordCloud(width=1000, height=500, background_color='white').generate(all_words)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Most Common Words in Tweets")
plt.tight_layout()
plt.show()
