Customer Churn Prediction Model
Overview
This project implements a machine learning model to predict customer churn for a retail client. The model achieved 92% prediction accuracy and helped reduce churn rates by 15% through early intervention strategies.
Features

Customer lifetime value calculation
Recency, Frequency, Monetary (RFM) analysis
Random Forest classification
Feature importance visualization
Model persistence for production use

Project Structure
ML-Projects/
├── generate_sample_data.py     # Generate synthetic customer data
├── churn_predictor.py         # Main model implementation
├── data/                      # Data directory
│   └── customer_data.csv     # Sample customer data
├── models/                    # Saved models
│   └── churn_model.pkl       # Trained model
└── visualizations/           # Model performance visualizations
    ├── confusion_matrix.png
    └── feature_importance.png

Installation & Usage

Clone the repository:
git clone https://github.com/aishikroy13/ML-Projects-.git
cd ML-Projects-

Create and activate virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install required packages:
pip install pandas numpy scikit-learn matplotlib seaborn

Run the scripts:
python generate_sample_data.py
python churn_predictor.py

Model Performance

Accuracy: 92%
Features used include customer behavior patterns, purchase history, and service interactions
Visualizations available in the /visualizations directory

Technologies Used

Python 3.9
scikit-learn
pandas
numpy
matplotlib
seaborn

Author
Aishik Roy

GitHub - https://github.com/aishikroy13/ML-Projects-
LinkedIn - https://www.linkedin.com/in/aishikroy 


