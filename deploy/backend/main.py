from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
import joblib
from collections import defaultdict
import warnings

from fastapi.middleware.cors import CORSMiddleware

# Load feature column order
feature_columns = joblib.load("feature_columns.joblib")

# Suppress warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Initialize app
app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["*"] if testing broadly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
model_root = joblib.load("random_forest_root_failure.joblib")
model_stem = joblib.load("random_forest_stem_failure.joblib")
model_branch = joblib.load("random_forest_branch_failure.joblib")

# Load training data for one-hot encoding reference
df = pd.read_csv("treedata.csv")
df = df.drop(columns=["ObjectID", "Other.1", "Length of Failed Part", "Diameter of Failed Part"], axis=1)
df = df.fillna("None")

categorical_cols = [
    "Tree Species", "Condition", "Site Factors", "Type of Soil", "Weather factors",
    "Failed Part", "Root Failure", "Stem Failure", "Branch Failure",
    "Location and Percentage of Decay", "Decay Present"
]

# Generate one-hot encoding structure
all_unique_labels = defaultdict(set)

for col in categorical_cols:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].apply(lambda x: [item.strip() for item in x.split(",") if item.strip() != ""])
    for items in df[col]:
        all_unique_labels[col].update(items)

all_one_hot_columns = []
for col in categorical_cols:
    for label in all_unique_labels[col]:
        all_one_hot_columns.append(f"{col}_{label}")

# Expected input keys and mapping to training feature names
field_map = {
    "treeSpecies": "Tree Species",
    "condition": "Condition",
    "siteFactors": "Site Factors",
    "soilType": "Type of Soil",
    "weatherFactors": "Weather factors",
    "rootFailure": "Root Failure",
    "stemFailure": "Stem Failure",
    "branchFailure": "Branch Failure",
    "locationOfDecay": "Location and Percentage of Decay",
    "decayAmount": "Decay Present",
}

numerical_fields = ["diameter", "height"]

@app.post("/api/evaluate_tree")
async def evaluate_tree(tree_data: Request):
    data = await tree_data.json()

    # Start forming input row
    input_row = {}

    # Numerical features
    for field in numerical_fields:
        input_row[field] = float(data.get(field, 0))
    
    # Diameter of Tree and Height of Tree 
    input_row["Diameter of Tree"] = input_row["diameter"]
    input_row["Height of Tree"] = input_row["height"]

    # Categorical one-hot encoding
    for field_key, col_name in field_map.items():
        raw_val = data.get(field_key, "")
        
        if isinstance(raw_val, list):  # Handle list inputs
            present_labels = [item.strip() for item in raw_val if item.strip() != ""]
        else:  # Handle single string inputs
            present_labels = [item.strip() for item in raw_val.split(",") if item.strip() != ""]

        for label in all_unique_labels[col_name]:
            col = f"{col_name}_{label}"
            input_row[col] = int(label in present_labels)

    # Fill missing one-hot columns with 0
    for col in all_one_hot_columns:
        if col not in input_row:
            input_row[col] = 0

    # Convert to DataFrame
    X_input = pd.DataFrame([input_row])
    
    # Ensure correct order and only expected columns
    X_input = X_input.reindex(columns=feature_columns, fill_value=0)

    # Predict
    pred_root = model_root.predict_proba(X_input)[0][1]
    pred_stem = model_stem.predict_proba(X_input)[0][1]
    pred_branch = model_branch.predict_proba(X_input)[0][1]

    return {
        "rootFailureProbability": round(pred_root, 4),
        "stemFailureProbability": round(pred_stem, 4),
        "branchFailureProbability": round(pred_branch, 4)
    }
