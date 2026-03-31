import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# Simple text augmentation techniques
def random_swap(words, n=1):
    if len(words) < 2:
        return words
    new_words = words.copy()
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return new_words

def random_deletion(words, p=0.1):
    if len(words) == 1:
        return words
    new_words = []
    for word in words:
        if random.uniform(0, 1) > p:
            new_words.append(word)
    if len(new_words) == 0:
        return [words[random.randint(0, len(words)-1)]]
    return new_words

def augment_text(text, num_augments=1):
    words = str(text).split()
    augmented = []
    for _ in range(num_augments):
        choice = random.choice(['swap', 'delete', 'none'])
        if choice == 'swap':
            aug_words = random_swap(words, n=max(1, len(words)//10))
        elif choice == 'delete':
            aug_words = random_deletion(words, p=0.15)
        else:
            aug_words = words # keep original as a copy
        augmented.append(" ".join(aug_words))
    return augmented

def main():
    print("Loading original data...")
    df = pd.read_csv('Combined Data.csv')
    df = df[['statement', 'status']].dropna()
    
    print(f"Original dataset size: {len(df)}")
    
    # We will augment ONLY the training data so we don't leak duplicated patterns into our validation tests
    X = df['statement']
    y = df['status']
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Augmenting training data to synthetically increase the dataset size...")
    augmented_X = []
    augmented_y = []
    
    for text, label in zip(X_train, y_train):
        # 1. Keep the original
        augmented_X.append(text)
        augmented_y.append(label)
        
        # 2. Add 2 more synthetically augmented variants (total training dataset size grows by ~3x)
        augs = augment_text(text, num_augments=2)
        for a in augs:
            augmented_X.append(a)
            augmented_y.append(label)
            
    print(f"Training dataset size AFTER augmentation: {len(augmented_X)} statements!")
    print(f"Test dataset size remains strictly: {len(X_test)} statements")
    
    print("Building and training pipeline...")
    # Increase the values we capture by utilizing unigrams + bigrams and upgrading max_features
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=25000, ngram_range=(1,2), stop_words='english')),
        ('clf', LogisticRegression(max_iter=2000, n_jobs=-1, C=2.0))
    ])
    
    print("Training expanded model (this will take longer now because data is 3x larger)...")
    pipeline.fit(augmented_X, augmented_y)
    
    print("Evaluating new improved model...")
    y_pred = pipeline.predict(X_test)
    print("\nClassification Report (Augmented Dataset):")
    print(classification_report(y_test, y_pred))
    
    model_path = 'mental_health_model_large.pkl'
    print(f"Saving large model to {model_path}...")
    joblib.dump(pipeline, model_path)
    print("Complete!")

if __name__ == "__main__":
    main()
