# Import the required libraries:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#######################################################################################################################

# Import the raw dataset:
raw_data = pd.read_csv("Total_data.csv")

# Drop MLd if you want:
raw_data = raw_data.drop('MLD', axis=1)

# Separate predictor and target variables:
Y = raw_data['TEP']
x = raw_data.drop('TEP', axis=1)  # Predictor array

#######################################################################################################################

# Handling categorical variables - if necessary: ue the pd.get_dummies() function:
Seasons = pd.get_dummies(x, drop_first=True)

x = x.drop('Season', axis=1)  # dropping extra column
x = pd.concat([x, Seasons], axis=1)  # concatenation of independent variables and new categorical variable.

#######################################################################################################################

# Splitting the data: Training and validation:
x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size=0.2, random_state=13)  # Set seed (13, 31, 99)

#######################################################################################################################

# APPLYING THE MULTIVARIATE REGRESSION

# Create object of LinearRegression class:
LR = LinearRegression()

# Fit the training data:
LR.fit(x_train, y_train)  # The model has now been trained on the training data!

# Make predictions on the testing sets:
y_prediction = LR.predict(x_test)

# Make predictions for the training set for reference later:
y_train_prediction = LR.predict(x_train)

#######################################################################################################################

# EVALUATING THE MODEL

# Use r2_score from sklearn: Closely related ot MSE
score = r2_score(y_test, y_prediction)

print('r2 score is', score)
print('mean_sqrd_error is==', mean_squared_error(y_test, y_prediction))
print('root_mean_squared error of is==', np.sqrt(mean_squared_error(y_test, y_prediction)))

# Plot the results of the cross validation:
plt.axes(aspect='equal')
plt.scatter(y_prediction, y_test)  # plot the validation results
plt.scatter(y_train_prediction, y_train)  # visualize the training aspects of the model alongside the validation results
plt.xlabel('True Values [TEP]')
plt.ylabel('Predictions [TEP]')
lims = [0, 150]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)  # Add a 1:1 line through the graph for comparison

