from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


print("Python Titanic model started.")

script_dir = Path(__file__).resolve().parent
candidate_data_dirs = [
    script_dir.parent / "data",
    Path("/app/data"),
]

data_dir = next(
    (path for path in candidate_data_dirs if (path / "train.csv").exists()),
    None,
)

if data_dir is None:
    raise FileNotFoundError("Could not find train.csv in src/data or /app/data.")

print(f"Using data folder: {data_dir}")

train_df = pd.read_csv(data_dir / "train.csv")
test_df = pd.read_csv(data_dir / "test.csv")

print(f"Loaded train.csv with shape: {train_df.shape}")
print(f"Loaded test.csv with shape: {test_df.shape}")

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
]

target = "Survived"
numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Sex", "Embarked"]

X = train_df[features].copy()
y = train_df[target].copy()
X_kaggle_test = test_df[features].copy()

print(f"Selected features: {features}")
print("Missing values before preprocessing:")
print(X.isna().sum())

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features,
        ),
        (
            "categorical",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_features,
        ),
    ]
)

print(
    "Data changes: filled numeric missing values with medians, "
    "filled categorical missing values with the most frequent values, "
    "scaled numeric features, and one-hot encoded categorical features."
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"Training rows: {len(X_train)}")
print(f"Held-out rows: {len(X_valid)}")

model.fit(X_train, y_train)

training_predictions = model.predict(X_train)
validation_predictions = model.predict(X_valid)

training_accuracy = accuracy_score(y_train, training_predictions)
held_out_accuracy = accuracy_score(y_valid, validation_predictions)

print(f"Training accuracy: {training_accuracy:.4f}")
print(f"Held-out test accuracy: {held_out_accuracy:.4f}")

model.fit(X, y)
kaggle_predictions = model.predict(X_kaggle_test)

print(f"Generated predictions for {len(kaggle_predictions)} Kaggle test passengers.")
print(f"Predicted survival rate: {kaggle_predictions.mean():.4f}")

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Survived": kaggle_predictions,
    }
)

output_path = script_dir / "predictions.csv"
submission.to_csv(output_path, index=False)

print(f"Saved predictions to: {output_path}")
print(submission.head())
print("Python Titanic model completed successfully.")

