import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        
    def load_and_preprocess_data(self, filepath):
        """
        Load and preprocess the customer data
        """
        # Ensure the filepath is correct
        filepath = os.path.join('data', filepath)
        
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found at: {filepath}")
            
        # Load data
        print(f"Loading data from: {filepath}")
        df = pd.read_csv(filepath)
        
        # Convert date to datetime and calculate recency
        df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
        df['recency'] = (datetime.now() - df['last_purchase_date']).dt.days
        
        # Encode categorical variables
        categorical_columns = ['customer_segment', 'product_category']
        for col in categorical_columns:
            self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col])
        
        return df
    
    def prepare_features(self, df):
        """
        Prepare features for model training
        """
        # Select features for model
        self.feature_columns = ['recency', 'frequency', 'monetary_value', 
                              'total_purchases', 'avg_order_value', 
                              'customer_service_calls', 'customer_segment', 
                              'product_category']
        
        X = df[self.feature_columns]
        y = df['churned']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train_model(self, X, y):
        """
        Train the Random Forest model
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Model Accuracy: {accuracy * 100:.2f}%')
        
        # Print detailed classification report
        print('\nClassification Report:')
        print(classification_report(y_test, y_pred))
        
        return X_test, y_test, y_pred
    
    def visualize_results(self, X_test, y_test, y_pred):
        """
        Create visualizations for model performance
        """
        # Create visualizations directory if it doesn't exist
        os.makedirs('visualizations', exist_ok=True)
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join('visualizations', 'confusion_matrix.png'))
        plt.close()
        
        # Feature Importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join('visualizations', 'feature_importance.png'))
        plt.close()
    
    def save_model(self, filename):
        """
        Save the trained model
        """
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        model_path = os.path.join('models', filename)
        with open(model_path, 'wb') as file:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_columns': self.feature_columns
            }, file)
        print(f"Model saved to: {model_path}")

def main():
    # Initialize predictor
    predictor = ChurnPredictor()
    
    try:
        # Load and preprocess data
        df = predictor.load_and_preprocess_data('customer_data.csv')
        
        # Prepare features
        X, y = predictor.prepare_features(df)
        
        # Train model and get predictions
        X_test, y_test, y_pred = predictor.train_model(X, y)
        
        # Visualize results
        predictor.visualize_results(X_test, y_test, y_pred)
        
        # Save model
        predictor.save_model('churn_model.pkl')
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()



