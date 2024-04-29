# Daniel Chudý
# MUNI FI, Brno

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Tokenize text
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and non-alphabetic tokens, and lemmatize
    preprocessed_text = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    
    # Join tokens into a single string
    return ' '.join(preprocessed_text)

def tf_idf_cosine_rec(df, category, title_string, mode="live"):

    if not mode == "eval":
        # Filter DataFrame by category
        df = df[df['Category'] == category]
    
    # Preprocess input title
    preprocessed_input_title = preprocess(title_string)
    
    # Preprocess titles in the DataFrame
    df['preprocessed_title'] = df['Title'].apply(preprocess)
    
    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    
    # Compute TF-IDF matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['preprocessed_title'])
    
    # Compute TF-IDF vector for input title
    input_tfidf = tfidf_vectorizer.transform([preprocessed_input_title])
    
    # Calculate cosine similarity between input title and all titles in the DataFrame
    # We have to explicitly set the dtype to 'float32' because of the deletion of values == 1.0 later on
    cosine_similarities = np.array(cosine_similarity(input_tfidf, tfidf_matrix).flatten(), dtype='float32')

    if mode == "eval":
        # In case there are the same articles in history and impression column
        cosine_similarities = cosine_similarities[cosine_similarities != 1]
        return np.sum(cosine_similarities)/len(cosine_similarities)

    # Get indices of top recommendations
    top_indices = cosine_similarities.argsort()[:-1][::-1]
    
    top_five = top_indices[:5]

    # Return recommended article titles
    recommended_titles = df.iloc[top_five]['Title'].tolist()
    return recommended_titles


if __name__ == "__main__":

    # Sample DataFrame
    data = {
        'News_id': ['N19639', 'N19640', 'N19641', 'N19642', 'N19643'],
        'Category': ['health', 'health', 'health', 'health', 'health'],
        'Subcategory': ['weightloss', 'weightloss', 'weightloss', 'weightloss', 'weightloss'],
        'Title': [
            "50 Worst Habits For Belly Fat",
            "10 guys that have shredded abs",
            "How to Reduce Belly Fat Naturally",
            "5 Foods The That Burn Belly Fat",
            "Yoga Poses for a Flat Belly"
        ]
    }

    df = pd.DataFrame(data)

    # Example usage
    mode = "live"
    recommended_titles = tf_idf_cosine_rec(df, category='health', title_string='50 Worst Habits For Belly Fat', mode=mode)
    if mode == "live":
        print("Recommended Articles:")
        for title in recommended_titles:
            print(title)
    print(recommended_titles)
