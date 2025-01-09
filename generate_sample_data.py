import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_data():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate sample data
    n_customers = 1000

    # Generate customer IDs
    customer_ids = range(1, n_customers + 1)

    # Generate dates within the last year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_customers)]
    
    # Convert dates to string format
    last_purchase_dates = [date.strftime('%Y-%m-%d') for date in dates]

    # Generate other features
    data = {
        'customer_id': customer_ids,
        'last_purchase_date': last_purchase_dates,
        'total_purchases': np.random.randint(1, 100, n_customers),
        'avg_order_value': np.random.uniform(20, 200, n_customers).round(2),
        'frequency': np.random.randint(1, 52, n_customers),  # purchases per year
        'monetary_value': np.random.uniform(100, 5000, n_customers).round(2),
        'customer_service_calls': np.random.randint(0, 10, n_customers),
        'customer_segment': np.random.choice(['Premium', 'Standard', 'Basic'], n_customers),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Home & Garden', 'Books'], n_customers)
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Calculate recency (days since last purchase)
    df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
    recency = (end_date - df['last_purchase_date']).dt.days
    
    # Convert back to string format for storage
    df['last_purchase_date'] = df['last_purchase_date'].dt.strftime('%Y-%m-%d')

    # Calculate churn based on multiple factors
    normalized_calls = df['customer_service_calls'] / df['customer_service_calls'].max()
    normalized_frequency = 1 - (df['frequency'] / df['frequency'].max())
    normalized_recency = recency / recency.max()

    churn_score = (normalized_calls + normalized_frequency + normalized_recency) / 3
    df['churned'] = (churn_score > 0.7).astype(int)

    # Save to CSV
    output_path = os.path.join('data', 'customer_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Sample data generated and saved to '{output_path}'")
    print(f"Total customers: {n_customers}")
    print(f"Churn rate: {(df['churned'].mean() * 100):.2f}%")

    # Display first few rows and data info
    print("\nFirst few rows of the dataset:")
    print(df.head())
    print("\nDataset information:")
    print(df.info())

if __name__ == "__main__":
    generate_sample_data()



