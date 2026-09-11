from sklearn.datasets import load_iris
import pandas as pd
import os

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["target"] = iris.target

os.makedirs("data", exist_ok=True)

df.to_csv("data/iris.csv", index=False)

print("Dataset created successfully!")
print(df.head())