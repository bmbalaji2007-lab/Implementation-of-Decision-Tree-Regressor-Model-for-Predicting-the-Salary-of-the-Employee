# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and create/load the employee salary dataset.
2. Split the dataset into training and testing data.
3. Train the Decision Tree Regressor model using the training data.
4. Predict salary values and evaluate the model using MAE, MSE, and R2 score.

## Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: BALAJI B M
RegisterNumber: 212225230029
```
```
# Program: Decision Tree Regressor for Predicting Employee Salary

# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Create Sample Dataset
data = {
    'Experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Age': [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    'Salary': [25000, 30000, 35000, 40000, 50000,
               60000, 65000, 70000, 80000, 90000]
}

df = pd.DataFrame(data)

# Step 3: Split Features and Target
X = df[['Experience', 'Age']]
y = df['Salary']

# Step 4: Split Dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 5: Create Decision Tree Regressor Model
model = DecisionTreeRegressor(random_state=42)

# Step 6: Train the Model
model.fit(X_train, y_train)

# Step 7: Predict the Salary
y_pred = model.predict(X_test)

# Step 8: Display Predicted Values
print("Predicted Salary Values:")
print(y_pred)

# Step 9: Evaluate the Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error:")
print(mae)

print("\nMean Squared Error:")
print(mse)

print("\nR2 Score:")
print(r2)

```

## Output:
<img width="322" height="173" alt="image" src="https://github.com/user-attachments/assets/d4b93a80-ea2b-42e1-aee6-454dc82ea6f2" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
