import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train_model():
    print("Loading dataset...")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "Testing.csv")
    
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        return

    data = pd.read_csv(data_path)
    
    # Drop missing values
    data = data.dropna()
    
    print("Preparing features and target...")
    # Features to use based on the original UI
    features = [
        'State', 'Year', 'Season', 'Crop', 'Area', 'Production', 
        'Rainfall', 'avg_temp', 'PH Value of Soil', 'Type of soil'
    ]
    
    X = data[features]
    y = data['Suitable Fertilizer']
    
    # Define categorical and numerical columns
    categorical_cols = ['State', 'Season', 'Crop', 'Type of soil']
    numerical_cols = ['Year', 'Area', 'Production', 'Rainfall', 'avg_temp', 'PH Value of Soil']
    
    # Preprocessing pipeline
    print("Building pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    # Model pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1234)
    
    print("Training Random Forest model...")
    pipeline.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("=" * 40)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("=" * 40)
    
    # Save the model
    model_path = os.path.join(base_dir, "crop_Model1.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
