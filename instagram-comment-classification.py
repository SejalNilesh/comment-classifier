import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset (assuming you have a CSV file with 'Comment' and 'Category' columns)
# You would need to create this CSV file from your Instagram data
dataset = pd.read_csv('Instagram_Comments.csv')

# Text cleaning function
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    text = re.sub(r'\@\w+|\#','', text)
    # Remove non-alphabetic characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    text = text.split()
    # Remove stopwords and lemmatize
    lemmatizer = WordNetLemmatizer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    text = [lemmatizer.lemmatize(word) for word in text if word not in set(all_stopwords)]
    # Join words back into a string
    return ' '.join(text)

# Clean all comments
dataset['cleaned_comment'] = dataset['Comment'].apply(clean_text)

# Create TF-IDF model
tfidf = TfidfVectorizer(max_features=1500)
X = tfidf.fit_transform(dataset['cleaned_comment']).toarray()

# Encode categories
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(dataset['Category'])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Train the Multinomial Naive Bayes model
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predict the Test set results
y_pred = classifier.predict(X_test)

# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Function to classify new comments
def classify_comment(comment):
    cleaned_comment = clean_text(comment)
    comment_vector = tfidf.transform([cleaned_comment]).toarray()
    prediction = classifier.predict(comment_vector)
    return le.inverse_transform(prediction)[0]

# Example usage
new_comment = "💀💀"
print(f"\nClassification for '{new_comment}': {classify_comment(new_comment)}")
