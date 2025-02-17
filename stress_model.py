import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

# Load the synthetic dataset
dataset = pd.read_csv('synthetic_health_data.csv')

# Define features and target
# Let's assume we want to predict 'Stress_Level' (you can change this to 'Depression_Level' or another target)
features = ['Age_Group', 'Sleep_Duration', 'Skin_Conductance', 'Blood_Pressure_Systolic', 
            'Blood_Pressure_Diastolic', 'Body_Temperature', 'Oxygen_Saturation']
target = 'Stress_Level'

# Convert categorical features to strings (CatBoost requires this)
dataset[features[0]] = dataset[features[0]].astype(str)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset[features], dataset[target], test_size=0.2, random_state=42)

# Create CatBoost Pool objects for training and testing
train_pool = Pool(data=X_train, label=y_train, cat_features=[0])  # 'Age_Group' is categorical
test_pool = Pool(data=X_test, label=y_test, cat_features=[0])

# Initialize the CatBoost classifier
model = CatBoostClassifier(
    iterations=100,  # Number of trees
    learning_rate=0.1,
    depth=6,
    loss_function='MultiClass',  # For multi-class classification
    verbose=10  # Print progress every 10 iterations
)

# Train the model
model.fit(train_pool, eval_set=test_pool)

# Evaluate the model
accuracy = model.score(test_pool)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model to a file
model.save_model('stress_level_model.cbm')

print("Model trained and saved as 'stress_level_model.cbm'.")
