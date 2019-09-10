# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 20:13:01 2018

@author: miru
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# create a data fram
df = pd.read_csv('USA_Housing.csv')
# access head of the dataframe
#print(df.head())
#check out info
#print(df.info())
#print(df.describe())
info = df.describe()
print(df.info())
# we can create all the plot with seaborn if the data is not too large 
#with pairplot

# for now we do scatter plot
#sns.pairplot(df)

# to check out the distribution of the house price
sns.distplot(df['Price'])
# to see the correlcation matrix of the dataframe
corr_mat = df.corr()
# to see correlation heat map
sns.heatmap(df.corr(),annot = True)

# now we want to split our data into training set and test set

'''
X_array will contain feautures to train and test on
Y_array will be the target variable; in this case the price column
we can toss out the address column since it contains only strings
natural language processing will later on allow us to use text info
'''

# for convenience, we will print the column names to the console
print(df.columns)
X = df[['Avg. Area Income', 'Avg. Area House Age',
       'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms',
       'Area Population']]
X = (X - np.mean(X)) / np.std(X)
X_good = df.iloc[:,0:5]
print(X == X_good)
y = df['Price']
first_row = X.iloc[1,:]
numfeat = len(X.columns)
y_1 = y[1]
lengthhehe = len(X.index)
# now we do a train test split on our data
'''

'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.4, random_state = 101 )

# now instantiate the object

lm = LinearRegression()
 # we want to fit the training set 
lm.fit(X_train,y_train)
# the y intercept
print(lm.intercept_)
intercept = lm.intercept_
# to print the coefficients w.r.t each column name
print(lm.coef_)
print(X.columns)
# for convenience, make a data frame
coeff = pd.DataFrame(lm.coef_, X.columns, columns = ['coeffs'])

# now we want to test with our test set

predictions = lm.predict(X_test)

# so far summary

# we have declared and instantiated our choice of model
# we have split the dataset into training and test set
# we have "trained the model" on training set by calling 'lm.fit(X_train, y_train)'
# we have derived the coefficients
# we have then tested our model on the test set by calling 'predictions = lm.predict(X_test)'

# so now we want to see how accurate we are; we know the values of y_test
plt.scatter(y_test, predictions)
# we want a nice distribution of points that clearly is fittable by a line
plt.show()
# now we want to check the residuals
# we want a nice bellshape, if it is not, LM was not the right choice of h in H
sns.distplot((y_test - predictions))
# regression evaluation metrics
'''
    1. mean absolute error
    2. mean squared error
    3.root mean squared error
    
    # above are loss functions and 2 and 3 are preferrable 2 esp. for punishing large errors
'''
# we need to import metrics for these loss functions
from sklearn import metrics
print('MAE: ', metrics.mean_absolute_error(y_test, predictions))
print('MSE: ', metrics.mean_squared_error(y_test, predictions))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

