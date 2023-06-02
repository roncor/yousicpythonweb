import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Data preprocessing
features = ['Age', 'Occupation', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
            'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', "Delay_from_due_date",
            "Changed_Credit_Limit", 'Num_Credit_Inquiries', 'Credit_Mix', "Outstanding_Debt",
            "Credit_Utilization_Ratio", "Payment_of_Min_Amount", "Total_EMI_per_month",
            "Amount_invested_monthly", "Payment_Behaviour", "Monthly_Balance"]
categorical_features = ['Occupation', 'Payment_Behaviour',
                        "Payment_of_Min_Amount", "Credit_Mix"]
target_variable = 'Credit_Score'
k_features = 10  # Number of top features to select

# Load the dataset
print("Loading the dataset...")
data = pd.read_csv('cleaned_df.csv')

# Separate features and target variable
X = data[features]
y = data[target_variable]

# Convert categorical features into dummy variables
X_encoded = pd.get_dummies(X, columns=categorical_features)

# Split the dataset into training and validation sets
print("Splitting the dataset into training and validation sets...")
X_train, X_valid, y_train, y_valid = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Perform feature selection
print("Performing feature selection...")
selector = SelectKBest(f_classif, k=k_features)
X_train_selected = selector.fit_transform(X_train, y_train)
X_valid_selected = selector.transform(X_valid)

# Get the selected feature indices
selected_feature_indices = selector.get_support(indices=True)

# Get the selected feature names
selected_features = X_encoded.columns[selected_feature_indices]

print("Selected features:", selected_features)

# Train the Random Forest classifier using selected features
print("Training the Random Forest classifier...")
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train_selected, y_train)

# Evaluate the classifier on the validation set
y_pred = rf_classifier.predict(X_valid_selected)
accuracy = accuracy_score(y_valid, y_pred)
print(f"Random Forest Accuracy: {accuracy}")

# Load the new dataset for prediction
print("Loading the new dataset for prediction...")
new_data = pd.read_csv('cleaned_test_df.csv')

# Check for missing values in the new dataset
if new_data.isnull().values.any():
    # Remove rows with missing values
    new_data = new_data.dropna()

# Apply the same preprocessing and feature selection steps on the new dataset
print("Preprocessing the new dataset and performing feature selection...")
X_new = new_data[features]
X_new_encoded = pd.get_dummies(X_new, columns=categorical_features)
X_new_selected = selector.transform(X_new_encoded)

# Make predictions on the new dataset using the selected features and trained model
print("Making predictions on the new dataset...")
predictions = rf_classifier.predict(X_new_selected)

# Create a DataFrame with the original test dataset and predictions
predictions_df = pd.DataFrame(new_data)
predictions_df['Predicted_Credit_Score'] = predictions

# Save the original test dataset and predictions as a CSV file
print("Saving the predictions as a CSV file...")
predictions_df.to_csv('test_predictions.csv', index=False)

print("Prediction process completed successfully.")
