#Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

# Load Boston housing dataset
import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("altavish/boston-housing-dataset")
housing_csv = os.path.join(path, "HousingData.csv")
print("Path to dataset files:", path)
df_housing = pd.read_csv(housing_csv)
print(df_housing)

# Separate features and target
data = pd.read_csv(housing_csv)
# Convert all column names to lowercase
data.columns = data.columns.str.lower()
X = data.drop("medv", axis=1)
y = data["medv"]

# Define categorical and numerical features
categorical_features = ['chas']
numerical_features = [col for col in X.columns if col not in categorical_features]

# Preprocessor – OneHotEncoder only for 'chas'
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(), categorical_features)
], remainder='passthrough')

# Create pipeline with XGBoost model

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1,
                               max_depth=4, random_state=42))
])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict and evaluate performance
predictions = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, predictions))

# Feature importance analysis
feature_names = list(model.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out()) + numerical_features
importances = model.named_steps['regressor'].feature_importances_
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Visualization of feature importance
sns.barplot(x="Importance", y="Feature", data=importance_df)
plt.title("Feature Importance in XGBoost Model")
plt.tight_layout()
plt.show()