import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import ast  # for safe parsing of services list


def parse_services(services_str):
    """Extract top service category and WP presence as features"""
    if pd.isna(services_str) or not services_str:
        return 0, 0, 0, False  # web %, mobile %, custom %, is_wp

    try:
        services = ast.literal_eval(services_str)
        if not isinstance(services, list):
            services = []
    except:
        services = []

    web_pct = 0
    mobile_pct = 0
    custom_pct = 0
    is_wp = any('wordpress' in s.lower() or 'wp ' in s.lower() for s in services)

    for item in services:
        try:
            pct = float(item.split('%')[0].strip())
            service = item.split('%')[1].strip().lower()
            if 'web development' in service or 'web design' in service:
                web_pct = max(web_pct, pct)
            if 'mobile' in service:
                mobile_pct = max(mobile_pct, pct)
            if 'custom software' in service:
                custom_pct = max(custom_pct, pct)
        except:
            continue

    return web_pct, mobile_pct, custom_pct, is_wp


def train_model():
    df = pd.read_csv('data/processed/partners_enriched.csv')

    # Parse services into useful numeric features
    df[['web_pct', 'mobile_pct', 'custom_pct', 'services_wp']] = df['services'].apply(
        lambda x: pd.Series(parse_services(x))
    )

    # Fill NaN
    df.fillna({
        'employees': 0,
        'clutch_rating': 0,
        'revenue_usd': 0,
        'kaycore_fit_score': 0,
        'web_pct': 0,
        'mobile_pct': 0,
        'custom_pct': 0,
        'services_wp': False
    }, inplace=True)

    # Better success proxy (relaxed until you have real labels)
    df['success'] = (
        (df['kaycore_fit_score'] >= 5) &           # lowered threshold
        (df['revenue_usd'] >= 1000000) &           # at least ~$1M
        (df['clutch_rating'] >= 4.5) &
        (df['employees'] >= 10)
    ).astype(int)

    print("Success class distribution:\n", df['success'].value_counts(normalize=True))

    # Features — now include service percentages!
    features = [
        'employees',
        'clutch_rating',
        'revenue_usd',
        'kaycore_fit_score',
        'web_pct',
        'mobile_pct',
        'custom_pct',
        'services_wp'          # boolean → will be treated as 0/1
    ]

    X = df[features]
    y = df['success']

    # Scale features (very important for clustering and model stability)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y if y.sum() > 0 else None
    )

    # Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"RandomForest Accuracy: {acc:.3f}")
    print("Feature importances:", dict(zip(features, model.feature_importances_)))

    joblib.dump(model, 'data/models/rf_model.pkl')
    joblib.dump(scaler, 'data/models/scaler.pkl')  # save scaler too!

    # KMeans clustering on scaled features
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # Optional: add cluster description (run after clustering)
    cluster_summary = df.groupby('cluster')[features].mean().round(2)
    print("\nCluster centroids (mean values):\n", cluster_summary)

    # Final output columns
    final_cols = [
        'name', 'country', 'location_city', 'website',
        'employees', 'revenue_usd', 'clutch_rating',
        'min_project_size_usd', 'is_wp_specialist',
        'kaycore_fit_score', 'partnership_priority',
        'web_pct', 'mobile_pct', 'custom_pct',
        'cluster'
    ]

    df[final_cols].to_csv('data/processed/partners_with_clusters.csv', index=False)
    print("partners_with_clusters.csv created!")


class PartnershipMLModels:
    def __init__(self):
        self.rf_model = joblib.load('data/models/rf_model.pkl')
        self.scaler = joblib.load('data/models/scaler.pkl')
        self.kmeans = None  # you can load or re-fit if needed

    def predict_new_partner(self, partner_dict):
        df_new = pd.DataFrame([partner_dict])
        # Assume partner_dict has same features or you fill missing
        X_new = df_new[self.rf_model.feature_names_in_]  # safer
        X_scaled = self.scaler.transform(X_new)

        success_prob = self.rf_model.predict_proba(X_scaled)[0][1]

        cluster = 0  # placeholder; can add kmeans.predict if you save kmeans

        return {
            'success_probability': round(float(success_prob), 3),
            'predicted_revenue_usd': partner_dict.get('revenue_usd', 0),
            'cluster': int(cluster)
        }


if __name__ == "__main__":
    train_model()
