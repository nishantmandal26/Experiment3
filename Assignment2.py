import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


data = {
    "Age": [22, 25, None, 30, 28],
    "Salary": [25000, 35000, 40000, None, 45000],
    "Department": ["IT", "HR", "IT", "Finance", "HR"],
    "Years_of_Experience": [1, 3, 5, None, 4]
}

df = pd.DataFrame(data)

X = df.copy()

numeric_features = ["Age", "Salary", "Years_of_Experience"]
categorical_features = ["Department"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])


categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


X_processed = preprocessor.fit_transform(X)

print("--- Original Dataset ---")
print(df)

print("\n--- Processed Data using MinMaxScaler ---")
print(X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed)

print("\n--- Processed Data Shape ---")
print(X_processed.shape)