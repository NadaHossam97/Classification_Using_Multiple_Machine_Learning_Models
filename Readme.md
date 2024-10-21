### Credit Score Classification Using Multiple Machine Learning Models

#### 1. **Introduction**

In this project, we aim to build a classification model that predicts credit scores based on various demographic features. A credit score is a critical metric used by financial institutions to evaluate a person's creditworthiness. By accurately predicting credit scores, lenders can assess the likelihood that a borrower will repay their debts, thus minimizing risks in lending decisions.

We will fit several machine learning models to this data, compare their performance based on accuracy, precision, recall, and F1-score, and select the best model for credit score prediction. One key aspect of this project is to ensure that the models are generalizing well and are not overfitting to the training data.

#### 2. **About the Dataset**

The dataset for this analysis is obtained from Kaggle.This dataset contains information about a sample of over 100 individuals from around the world. The features included in the dataset are:
- **Age**: The age of the individual.
- **Gender**: Male or Female.
- **Income**: The individual's annual income.
- **Education**: The highest level of education attained by the individual.
- **Marital Status**: Whether the individual is single, married, or otherwise.
- **Number of Children**: The number of children the individual has.
- **Home Ownership**: Whether the individual owns a home.
- **Credit Score**: The credit score category of the individual, which is the target variable (e.g., "Low", "Average", "High").

#### 3. **Project Workflow and Explanation of Steps**

1. **Handling Categorical variables**:
   - Categorical variables (such as gender, education, and marital status) are transformed into numerical representations using pd.get_dummies(). Allowing the models to interpret the categorical data correctly.
2. **Target Variable Encoding**:
   - The target variable (credit score) is a categorical variable with labels such as "Low", "Average", and "High". We use **`LabelEncoder`** to convert these categories into numerical values (e.g., 0, 1, 2) to make them suitable for classification algorithms.

3. **Splitting the Data**:
   - The dataset is split into features (`X`) and the target (`y`), and then further split into training and test sets (80% training, 20% testing) using **`train_test_split()`**.

4. **Model Selection and Feature Scaling**:
   - Multiple machine learning models are chosen, including:
     - **Logistic Regression**
     - **Support Vector Machine (SVM)**
     - **K-Nearest Neighbors (KNN)**
     - **Decision Tree**
     - **Random Forest**
     - **Gradient Boosting**
     - **XGBoost**
   - Models like Logistic Regression, SVM, and KNN require scaled features, so **`StandardScaler`** is applied to standardize the dataset. Models like Decision Tree and Random Forest do not need scaling, so we train these models on unscaled data.

5. **Training and Evaluation**:
   - Each model is trained on the training data and evaluated on the test data using accuracy, precision, recall, and F1-score. These metrics are calculated to provide a comprehensive view of how well each model performs.



#### 4. **Model Comparison Table**

| Model                 | Accuracy | Precision | Recall | F1 Score |
|-----------------------|----------|-----------|--------|----------|
| Logistic Regression    | 0.9697   | 0.9709    | 0.9697 | 0.9683   |
| SVM                   | 0.9394   | 0.9407    | 0.9394 | 0.9380   |
| K-Nearest Neighbors    | 0.9394   | 0.9394    | 0.9394 | 0.9394   |
| Decision Tree          | 0.9697   | 0.9747    | 0.9697 | 0.9707   |
| Random Forest          | 0.9394   | 0.9394    | 0.9394 | 0.9394   |
| Gradient Boosting      | 0.9697   | 0.9747    | 0.9697 | 0.9707   |
| XGBoost (excluded)     | 1.0000   | 1.0000    | 1.0000 | 1.0000   |

#### 5. **Comments**

The comparison table reveals that:
- **Logistic Regression**, **Decision Tree**, and **Gradient Boosting** are the top-performing models with an accuracy of **96.97%**. These models also score high in terms of precision, recall, and F1-score, indicating that they perform well at identifying and correctly predicting credit scores.
- **SVM**, **KNN**, and **Random Forest** achieve slightly lower accuracy at **93.94%**, but they are still strong contenders for this classification task.
- **XGBoost** achieved perfect accuracy (100%), which is a likely sign of overfitting. Since a perfect score is rarely expected in real-world scenarios, we excluded XGBoost from the final model selection.

#### 6. **Conclusion**

In this project, we successfully built and evaluated several machine learning models for credit score classification. **Logistic Regression**, **Decision Tree**, and **Gradient Boosting** were the most accurate models, each achieving high performance across multiple metrics. **XGBoost**, while achieving perfect accuracy, was excluded due to signs of overfitting.

#### 7. **Recommendations**

Model performance should be continuously monitored over time to detect any changes in data distribution, as the dataset may evolve and affect prediction accuracy.