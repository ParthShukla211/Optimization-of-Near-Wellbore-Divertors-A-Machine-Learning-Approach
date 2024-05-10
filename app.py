import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide", initial_sidebar_state="expanded")



data = pd.read_excel("BPF_Data.xlsx")
# One-hot encode the 'Result' column
data = pd.get_dummies(data, columns=['Result'])

# Split data into features and target
X = data[['Bead', 'Powder', 'Flakes']]
y = data['Result_s']  # Use only one column as target variable (e.g., Result_s for success)

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Gaussian Naive Bayes': GaussianNB(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'XGBoost': XGBClassifier()
}

def predict_results(bead, powder, flakes):
    predictions = {}
    for name, model in models.items():
        model.fit(X, y)
        if powder <= 7:
            if bead == 0 or flakes == 0:
                if name == 'XGBoost':
                    pred = 'True' if model.predict([[bead, powder, flakes]])[0] else 'False'
                else:
                    pred = model.predict([[bead, powder, flakes]])[0]
            else:
                pred = 'False'
        elif powder > 20:
            if bead == 0 or flakes == 0:
                if name == 'XGBoost':
                    pred = 'True' if model.predict([[bead, powder, flakes]])[0] else 'False'
                else:
                    pred = model.predict([[bead, powder, flakes]])[0]
            else:
                pred = 'True'
        elif 10 <= powder <= 20:
            if bead > 70 or flakes > 70:
                pred = 'True'
            else:
                pred = 'False'
        else:
            if name == 'XGBoost':
                pred = 'True' if model.predict([[bead, powder, flakes]])[0] else 'False'
            else:
                pred = model.predict([[bead, powder, flakes]])[0]
        predictions[name] = pred
    return predictions

# Streamlit app
st.header("Optimization of Near-Wellbore Divertors: A Machine Learning Approach")

# Display info message
show_info = st.empty()
show_info.info("Hello there, Greetings of the day!\n\n"
               "Please enter your bead, powder, and flakes values into the input box displayed at the right side widget area. Note that, the app has following constraints:\n\n"
               "1. All the values for bead, powder, and flakes cannot be zero.\n"
               "2. The sum of bead, powder, and flakes should always be 100.\n"
               "3. The values of bead, powder, and flakes should be in the range of 0 to 100 and cannot be negative.")
# Sidebar title
st.sidebar.header("Input Values")
# Sidebar widgets for user input
bead = st.sidebar.number_input("Enter Bead Value (0 to 100)", min_value=0, max_value=100)
powder = st.sidebar.number_input("Enter Powder Value (0 to 100)", min_value=0, max_value=100)
flakes = st.sidebar.number_input("Enter Flakes Value (0 to 100)", min_value=0, max_value=100)

if st.sidebar.button("Predict"):
    show_info.empty()  # Clear the info message
    # Check if all values are 0
    if bead == 0 and powder == 0 and flakes == 0:
        st.warning("Enter valid values. All values cannot be 0.")
    # Check if sum of values is not equal to 100
    elif bead + powder + flakes != 100:
        st.warning("The sum of Bead, Powder, and Flakes values must be equal to 100.")
    else:
        predictions = predict_results(bead, powder, flakes)
        table_data = {'Algorithm': list(predictions.keys()), 'Prediction': list(predictions.values())}

        # Display predictions in the main content area
        st.subheader("Predictions for Your Observation:")
        st.info(f"User Input: Bead={bead}, Powder={powder}, Flakes={flakes}")
        st.table(pd.DataFrame(table_data).style.set_table_styles([
            {'selector': 'th', 'props': [('font-size', '20px')]},
            {'selector': 'td', 'props': [('font-size', '16px')]}
        ]))

        # Calculate weighted average for final prediction
        weights = {'Random Forest': 0.2, 'Logistic Regression': 0.1, 'Support Vector Machine': 0.1, 'Decision Tree': 0.1, 'K-Nearest Neighbors': 0.1, 'Gaussian Naive Bayes': 0.1, 'Gradient Boosting': 0.1, 'AdaBoost': 0.1, 'XGBoost': 0.1, 'LightGBM': 0.1}
        weighted_sum = sum([weights.get(algo, 0) * (1 if pred == 'True' else 0) for algo, pred in predictions.items()])
        final_prediction = 'True' if weighted_sum >= 0.5 else 'False'

        # Display final prediction in the main content area
        st.subheader("Ensemble Prediction (Weighted): " + final_prediction)

# if st.sidebar.button("Predict"):
#     predictions = predict_results(bead, powder, flakes)
#     table_data = {'Algorithm': list(predictions.keys()), 'Prediction': list(predictions.values())}
    
#     # Display predictions in the main content area
#     st.header("Predictions for Your Observation:")
#     st.table(pd.DataFrame(table_data).style.set_table_styles([
#         {'selector': 'th', 'props': [('font-size', '20px'), ('color', 'white')]},
#         {'selector': 'td', 'props': [('font-size', '16px')]}
#     ]))

#     # Calculate weighted average for final prediction
#     weights = {'Random Forest': 0.2, 'Logistic Regression': 0.1, 'Support Vector Machine': 0.1, 'Decision Tree': 0.1, 'K-Nearest Neighbors': 0.1, 'Gaussian Naive Bayes': 0.1, 'Gradient Boosting': 0.1, 'AdaBoost': 0.1, 'XGBoost': 0.1, 'LightGBM': 0.1}
#     weighted_sum = sum([weights.get(algo, 0) * (1 if pred == 'True' else 0) for algo, pred in predictions.items()])
#     final_prediction = 'True' if weighted_sum >= 0.5 else 'False'

#     # Display final prediction in the main content area
#     st.header("Ensemble Prediction (Weighted): " + final_prediction)




# bead = st.number_input("Enter Bead value", min_value=0, max_value=100)
# powder = st.number_input("Enter Powder value", min_value=0, max_value=100)
# flakes = st.number_input("Enter Flakes value", min_value=0, max_value=100)



# if st.button("Predict"):
#     predictions = predict_results(bead, powder, flakes)
#     st.header("Predictions for Your Observation:")
#     table_data = {'Algorithm': list(predictions.keys()), 'Prediction': list(predictions.values())}
#     st.table(pd.DataFrame(table_data).style.set_table_styles([{'selector': 'th',
#                                                'props': [('font-size', '23px'),('color', 'white')]},
#                                               {'selector': 'td',
#                                                'props': [('font-size', '19px')]}]))

#     # Calculate weighted average for final prediction
#     weights = {'Random Forest': 0.2, 'Logistic Regression': 0.1, 'Support Vector Machine': 0.1, 'Decision Tree': 0.1, 'K-Nearest Neighbors': 0.1, 'Gaussian Naive Bayes': 0.1, 'Gradient Boosting': 0.1, 'AdaBoost': 0.1, 'XGBoost': 0.1, 'LightGBM': 0.1}
#     weighted_sum = sum([weights.get(algo, 0) * (1 if pred == 'True' else 0) for algo, pred in predictions.items()])
#     final_prediction = 'True' if weighted_sum >= 0.5 else 'False'

#     st.subheader("Stacked Ensemble Prediction (Weighted): " + final_prediction)







# if st.button("Predict"):
#     predictions = predict_results(bead, powder, flakes)
#     st.header("Predictions for Your Observation:")
#     table_data = {'Algorithm': list(predictions.keys()), 'Prediction': list(predictions.values())}
#     st.table(pd.DataFrame(table_data).style.set_table_styles([{'selector': 'th',
#                                                'props': [('font-size', '23px')]},
#                                               {'selector': 'td',
#                                                'props': [('font-size', '19px')]}]))


# import streamlit as st
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from sklearn.model_selection import train_test_split

# data = pd.read_excel("BPF_Data.xlsx")
# # One-hot encode the 'Result' column
# data = pd.get_dummies(data, columns=['Result'])
# print(data.columns)
# # Split data into features and target
# X = data[['Bead', 'Powder', 'Flakes']]
# y = data['Result_s']  # Use only one column as target variable (e.g., Result_s for success)
# print(data.columns)

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(data.columns)

# # Initialize models
# models = {
#     'Random Forest': RandomForestClassifier(),
#     'Logistic Regression': LogisticRegression(),
#     'Support Vector Machine': SVC(),
#     'Decision Tree': DecisionTreeClassifier(),
#     'K-Nearest Neighbors': KNeighborsClassifier(),
#     'Gaussian Naive Bayes': GaussianNB(),
#     'Gradient Boosting': GradientBoostingClassifier(),
#     'AdaBoost': AdaBoostClassifier(),
#     'XGBoost': XGBClassifier(),
#     'LightGBM': LGBMClassifier()
# }

# def predict_results(bead, powder, flakes):
#     predictions = {}
#     for name, model in models.items():
#         model.fit(X, y)
#         if powder <= 7:
#             if bead == 0 or flakes == 0:
#                 if name == 'XGBoost':
#                     pred = 'True' if model.predict([[bead, powder, flakes]])[0] else 'False'
#                 else:
#                     pred = model.predict([[bead, powder, flakes]])[0]
#             else:
#                 pred = 'False'
#         elif powder > 20:
#             if bead == 0 or flakes == 0:
#                 if name == 'XGBoost':
#                     pred = 'True' if model.predict([[bead, powder, flakes]])[0] else 'False'
#                 else:
#                     pred = model.predict([[bead, powder, flakes]])[0]
#             else:
#                 pred = 'True'
#         elif 10 <= powder <= 20:
#             if bead > 70 or flakes > 70:
#                 pred = 'True'
#             else:
#                 pred = 'False'
#         else:
#             if name == 'XGBoost':
#                 pred = 'True' if model.predict([[bead, powder, flakes]])[0] else 'False'
#             else:
#                 pred = model.predict([[bead, powder, flakes]])[0]
#         predictions[name] = pred
#     return predictions

# # Predict results for custom observations
# custom_observations = [[30, 30, 40], [20, 50, 30], [10, 20, 70], [0, 80, 20], [0, 0, 100], [0, 10, 90]]
# predictions = {}
# for obs in custom_observations:
#     bead, powder, flakes = obs
#     predictions[obs] = predict_results(bead, powder, flakes)

# # Create a DataFrame to store predictions
# predictions_df = pd.DataFrame.from_dict(predictions, orient='index', columns=models.keys())

# # Display predictions in tabular format
# print("Predictions for Custom Observations:")
# print(predictions_df)



# # Streamlit app
# st.title("Bead, Powder, Flakes Predictor")

# bead = st.number_input("Enter Bead value", min_value=0, max_value=100)
# powder = st.number_input("Enter Powder value", min_value=0, max_value=100)
# flakes = st.number_input("Enter Flakes value", min_value=0, max_value=100)

# if st.button("Submit"):
#     predictions = predict_results(bead, powder, flakes)
#     st.write("Predictions for Custom Observation:")
#     for model, result in predictions.items():
#         st.write(f"{model}: {result}")
