import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

def main():
    print("Loading data...")
    try:
        df = pd.read_csv('Combined Data.csv')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Keep only the relevant columns and drop NaNs
    df = df[['statement', 'status']].dropna()

    X = df['statement']
    y = df['status']

    print(f"Dataset shape after dropping nulls: {df.shape}")
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creating a pipeline with TF-IDF Vectorizer and Logistic Regression
    print("Building and training pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, n_jobs=-1))
    ])

    pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    model_path = 'mental_health_model.pkl'
    print(f"Saving model to {model_path}...")
    joblib.dump(pipeline, model_path)
    print("Complete!")

if __name__ == "__main__":
    main()
