import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


data = {
    "Age": [22, 25, 27, 30, 28],
    "Salary": [25000, 35000, 40000, 30000, 45000],
    "Years_of_Experience": [1, 3, 5, 2, 4]
}

df = pd.DataFrame(data)


X = df[["Age", "Salary", "Years_of_Experience"]]


standard_scaler = StandardScaler()
X_standard = standard_scaler.fit_transform(X)


minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(X)


print("--- Original Data ---")
print(X)


print("\n--- StandardScaler Output ---")
print(X_standard)

print("\nStandardScaler Minimum:", X_standard.min())
print("StandardScaler Maximum:", X_standard.max())


print("\n--- MinMaxScaler Output ---")
print(X_minmax)

print("\nMinMaxScaler Minimum:", X_minmax.min())
print("MinMaxScaler Maximum:", X_minmax.max())