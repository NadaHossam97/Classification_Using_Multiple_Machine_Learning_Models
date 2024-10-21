import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb

#Load Dataframe
df = pd.read_csv(".\\Credit_Score.csv")

# Splitting the data into features and target
X = df.drop('Credit Score', axis=1)
y = df['Credit Score']

#Encoding categorical Variables
X = pd.get_dummies(X, drop_first=True)

# Convert the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

#Split dataset to train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models that require scaling
scaled_classifiers = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# Models that don't require scaling
non_scaled_classifiers = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": xgb.XGBClassifier()
}

# Dictionary to store model performance
model_performance = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': []
}

# Function to evaluate each model
def evaluate_model(model_name, model, X_train, X_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Append the results to the performance dictionary
    model_performance['Model'].append(model_name)
    model_performance['Accuracy'].append(accuracy)
    model_performance['Precision'].append(precision)
    model_performance['Recall'].append(recall)
    model_performance['F1 Score'].append(f1)

# Apply scaling only to models that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Evaluate models that require scaling
for model_name, model in scaled_classifiers.items():
    evaluate_model(model_name, model, X_train_scaled, X_test_scaled)

# Evaluate models that do not require scaling
for model_name, model in non_scaled_classifiers.items():
    evaluate_model(model_name, model, X_train, X_test)

# Convert the dictionary to a Pandas DataFrame
comparison_df = pd.DataFrame(model_performance)

# Display the comparison table
print(comparison_df)







