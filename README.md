# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.
6. Assign the train dataset and test dataset.
7. From sklearn.tree import DecisionTreeClassifier.
8. Use criteria as entropy.
9. From sklearn import metrics.
10. Find the accuracy of our model and predict the require values.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: HARSHITHA V
RegisterNumber:  212223230074
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv(r"C:\Users\admin\Downloads\Employee.csv")

# Check for missing values
print("Missing values in each column:\n", data.isnull().sum())

# Encode the 'salary' categorical feature
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])

# Features and target variable
X = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", 
           "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
y = data["left"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Initialize and train the Decision Tree Classifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(X_train, y_train)

# Make predictions
y_pred = dt.predict(X_test)

# Calculate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Example prediction
input_data = [[0.5, 0.8, 9, 260, 6, 0, 1, 2]]  # Make sure the input matches the features' order
prediction = dt.predict(input_data)
print("Prediction for input data:", prediction)

```
## Output:
![image](https://github.com/user-attachments/assets/78a757cd-2de0-44e1-8cbc-307c7b2f6b56)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
