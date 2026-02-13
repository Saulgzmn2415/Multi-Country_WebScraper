import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import joblib


def train_model():
    # Load enriched dataset
    df = pd.read_csv('data/processed/partners_enriched.csv')
    
    # Create success target
    df['success'] = (
        (df['kaycore_fit_score'] >= 7) &
        (df['revenue_usd'] >= 300000)
    ).astype(int)
    
    # Features for model
    features = [
        'employees',
        'clutch_rating',
        'revenue_usd',
        'kaycore_fit_score'
    ]
    
    X = df[features]
    y = df['success']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Evaluate model
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print("Model Accuracy:", acc)
    
    # Save model
    joblib.dump(model, 'data/models/rf_model.pkl')
    print("Model saved successfully!")

    # -----------------------------------
    # K-MEANS CLUSTERING (Day 3)
    # -----------------------------------
    cluster_features = df[[
        'employees',
        'clutch_rating',
        'revenue_usd',
        'kaycore_fit_score'
    ]]

    kmeans = KMeans(n_clusters=4, random_state=42)
    df['cluster'] = kmeans.fit_predict(cluster_features)

    # -----------------------------------
    # SAVE FINAL DATASET FOR DAY 4
    # -----------------------------------
    df.to_csv('data/processed/partners_with_clusters.csv', index=False)
    print("partners_with_clusters.csv created successfully!")


if __name__ == "__main__":
    train_model()
