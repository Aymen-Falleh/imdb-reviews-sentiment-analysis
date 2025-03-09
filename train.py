import pandas as pd
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Download necessary NLTK resources
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('IMDB Dataset.csv')

# Preprocessing function
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')

def preprocess_text(text):
    text = text.lower()
    text = re.sub('<br />', '', text)
    text = re.sub(r'\W', ' ', text)
    tokens = tokenizer.tokenize(text)  # Using RegexpTokenizer instead of word_tokenize
    return " ".join([w for w in tokens if w not in stop_words])

# Apply preprocessing
df['review'] = df['review'].apply(preprocess_text)
df['sentiment'] = df['sentiment'].replace({'positive': 1, 'negative': 0})

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['review'])
y = df['sentiment']

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression()
model.fit(x_train, y_train)

# Evaluate model
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))


# Save the trained Logistic Regression model
joblib.dump(model, 'sentiment_model.pkl')

# Save the TF-IDF vectorizer
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("Model and vectorizer saved successfully!")
