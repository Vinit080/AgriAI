import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

print("Loading Testing.csv...")
df = pd.read_csv("Testing.csv")

print("Generating synthetic N, P, K target values...")
np.random.seed(42)
def map_npk(fert):
    if pd.isna(fert):
        return pd.Series({'N': 25.0, 'P': 25.0, 'K': 25.0})
    fert = str(fert)
    if "Nitrogen" in fert:
        return pd.Series({'N': np.random.uniform(40, 60), 'P': np.random.uniform(10, 20), 'K': np.random.uniform(10, 20)})
    elif "Phosphorus" in fert:
        return pd.Series({'N': np.random.uniform(10, 20), 'P': np.random.uniform(40, 60), 'K': np.random.uniform(10, 20)})
    elif "Potassium" in fert:
        return pd.Series({'N': np.random.uniform(10, 20), 'P': np.random.uniform(10, 20), 'K': np.random.uniform(40, 60)})
    else:
        return pd.Series({'N': np.random.uniform(20, 30), 'P': np.random.uniform(20, 30), 'K': np.random.uniform(20, 30)})

npk_df = df['Suitable Fertilizer'].apply(map_npk)
df = pd.concat([df, npk_df], axis=1)

X = df[['State', 'Year', 'Season', 'Crop', 'Area', 'Production', 'Rainfall', 'avg_temp', 'PH Value of Soil', 'Type of soil']]
y = df[['N', 'P', 'K']]

num_features = ['Year', 'Area', 'Production', 'Rainfall', 'avg_temp', 'PH Value of Soil']
cat_features = ['State', 'Season', 'Crop', 'Type of soil']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', MultiOutputRegressor(RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)))
])

print("Training MultiOutputRegressor. This might take a minute...")
pipeline.fit(X, y)

joblib.dump(pipeline, 'npk_model.joblib')
print("NPK MultiOutputRegressor trained and saved successfully to npk_model.joblib.")
