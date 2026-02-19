import pandas as pd

def estimate_revenue(row):
    revenue = row['employees'] * 100000

    if row['clutch_rating'] >= 4.5:
        revenue *= 1.2

    if row['country'] == 'US':
        revenue *= 1.3
    elif row['country'] == 'UK':
        revenue *= 1.1
    elif row['country'] == 'SA':
        revenue *= 0.7

    return round(revenue, 2)


def calculate_fit_score(row):
    score = 0

    if row['is_wp_specialist']:
        score += 3

    if 25 <= row['employees'] <= 100:
        score += 2

    if row['clutch_rating'] >= 4.5:
        score += 2

    if row['country'] in ['US', 'UK', 'AU']:
        score += 1

    return min(score, 10)


def enrich():
    df = pd.read_csv('data/raw/global_partners_raw.csv')

    # Feature engineering
    df['revenue_usd'] = df.apply(estimate_revenue, axis=1)
    df['kaycore_fit_score'] = df.apply(calculate_fit_score, axis=1)

    df['partnership_priority'] = df['kaycore_fit_score'].apply(
        lambda x: 'High' if x >= 7 else 'Medium'
    )

    # Preserve all important metadata from the scraper
    keep_cols = [
        'name',
        'country',
        'location_city',
        'website',
        'employees',
        'clutch_rating',
        'min_project_size_usd',
        'services',
        'is_wp_specialist',
        'revenue_usd',
        'kaycore_fit_score',
        'partnership_priority'
    ]

    df = df[keep_cols]

    df.to_csv('data/processed/partners_enriched.csv', index=False)
    print("Enrichment complete!")


if __name__ == "__main__":
    enrich()
