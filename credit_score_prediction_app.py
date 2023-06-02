import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

# Define the Streamlit application
def main():
    # Set page title and styles
    st.set_page_config(page_title="Data Analysis with Python Credit Profile", page_icon="📊")
    st.markdown(
        """
        <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 20px;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Display the title
    st.markdown("<h1 class='title'>Data Analysis with Python</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='title'>Credit profile classification</h2>", unsafe_allow_html=True)
    st.write("Welcome to the data analysis project!")

    # Load the dataset
    st.sidebar.title("Dataset")
    dataset_path = st.sidebar.file_uploader("Upload the dataset", type="csv")
    if dataset_path is not None:
        data = pd.read_csv(dataset_path)
        st.write("Data Loaded:")
        st.dataframe(data)

        # Add headers to the tables
        data.columns = ["ID"] + data.columns[1:].tolist()

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

        # Separate features and target variable
        X = data[features]
        y = data[target_variable]

        # Convert categorical features into dummy variables
        X_encoded = pd.get_dummies(X, columns=categorical_features)

        # Split the dataset into training and validation sets
        X_train, X_valid, y_train, y_valid = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        # Perform feature selection
        selector = SelectKBest(f_classif, k=k_features)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_valid_selected = selector.transform(X_valid)

        # Get the selected feature indices
        selected_feature_indices = selector.get_support(indices=True)

        # Get the selected feature names
        selected_features = X_encoded.columns[selected_feature_indices]

        st.write("Feature Selection:")
        # Create a DataFrame to display the selected features
        feature_selection_table = pd.DataFrame({"Name": selected_features})
        st.table(feature_selection_table[["Name"]])

        # Define the models to train
        models = {
            "Random Forest": RandomForestClassifier(),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Naive Bayes": GaussianNB()
        }

        best_model = None
        best_accuracy = 0.0
        best_precision = 0.0
        best_recall = 0.0
        best_f1 = 0.0

        # Train and evaluate each model using the selected features
        st.write("Model Training and Evaluation:")
        evaluation_results = []
        for model_name, model in models.items():
            st.write(f"Training {model_name}...")
            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_valid_selected)
            accuracy = accuracy_score(y_valid, y_pred)
            precision = precision_score(y_valid, y_pred, average='weighted')  # Update average parameter
            recall = recall_score(y_valid, y_pred, average='weighted')  # Update average parameter
            f1 = f1_score(y_valid, y_pred, average='weighted')  # Update average parameter

            evaluation_results.append({
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            })

            if accuracy > best_accuracy:
                best_model = model
                best_accuracy = accuracy
                best_precision = precision
                best_recall = recall
                best_f1 = f1

        evaluation_table = pd.DataFrame(evaluation_results)
        st.table(evaluation_table)

        st.write(f"The best model is {type(best_model).__name__} with accuracy: {round(best_accuracy * 100, 2)}%")
        st.write(f"Precision: {round(best_precision, 2)}")
        st.write(f"Recall: {round(best_recall, 2)}")
        st.write(f"F1 Score: {round(best_f1, 2)}")

        # Calculate the confusion matrix for the best model
        y_pred = best_model.predict(X_valid_selected)
        cm = confusion_matrix(y_valid, y_pred)

        st.write("Confusion Matrix:")
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        st.pyplot()

        # Prediction on new data
        st.sidebar.title("Prediction on New Data")
        new_data = st.sidebar.file_uploader("Upload new data for prediction", type="csv")
        if new_data is not None:
            new_data = pd.read_csv(new_data)

            # Apply the same preprocessing and feature selection steps on the new dataset
            X_new = new_data[features]
            X_new_encoded = pd.get_dummies(X_new, columns=categorical_features)
            X_new_selected = selector.transform(X_new_encoded)

            # Make predictions on the new dataset using the selected features and the best model
            predictions = best_model.predict(X_new_selected)

            # Create a DataFrame with the selected columns and predictions
            predictions_df = pd.DataFrame(new_data, columns=["Name", "Age", "Occupation", "Num_Bank_Accounts",
                                                             "Num_Credit_Card", "Interest_Rate",
                                                             "Delay_from_due_date", "Num_Credit_Inquiries",
                                                             "Credit_Mix", "Payment_of_Min_Amount"])
            predictions_df['Predicted_Credit_Score'] = predictions

            st.write("Prediction Results:")
            predictions_table = predictions_df.head(10)  # Display 10 predictions per page

            # Pagination for the prediction results
            current_page = st.session_state.get("page_number", 0)
            if st.button("Previous Page") and current_page > 0:
                current_page -= 1
            elif st.button("Next Page") and current_page < (len(predictions_df) // 10):
                current_page += 1

            start_index = current_page * 10
            end_index = start_index + 10
            predictions_table = predictions_df[start_index:end_index]
            st.table(predictions_table)

            # Save the original test dataset and predictions as a CSV file
            predictions_file = "test_predictions.csv"
            st.button("Save Predictions", on_click=save_predictions, args=(predictions_df, predictions_file))

            # Update the session state with the current page number
            st.session_state["page_number"] = current_page


def save_predictions(predictions_df, filename):
    predictions_df.to_csv(filename, index=False)
    st.success(f"Predictions saved successfully as {filename}")


# Run the Streamlit application
if __name__ == "__main__":
    main()
