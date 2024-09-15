# %%
# importing required lib

import pandas as pd                  # to work on data sets
import numpy as np                   # for scientific and math calculation
import seaborn as sns                # to visualize data using matplot
import matplotlib.pyplot as plt      # make visuals of data such as graph, bar graph, scatter plot etc...

# remove the comment while running
# %matplotlib inline

# %%
# reading the data for training the model
HouseDF = pd.read_csv("boston-housing/train.csv")

# %%
# top 5 element/row of the training data
HouseDF.head()

# %%
# information about the colums of the data set
HouseDF.info()
# we did not found any null values so we can proceed withoud removing null values

# %%
# describe data set and values like mean, deviation, min, max etc...
HouseDF.describe()

# %%
HouseDF.columns

# %%
sns.pairplot(HouseDF)

# %% [markdown]
# 

# %%
sns.heatmap(HouseDF.corr(), annot=True)

# %%
X = HouseDF[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad',
       'tax', 'ptratio', 'black', 'lstat']]

y = HouseDF['medv']

# %%
# importing moduls to split the data to train and test
from sklearn.model_selection import train_test_split

# %%
# splitting data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=101)

# %%
#importing linear regression model to train our model
from sklearn.linear_model import LinearRegression

# %%
lm = LinearRegression()

# %%
# fitting the model
lm.fit(X_train, y_train)

# %%
coeff_DF = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])

# %%
coeff_DF

# %%
# predicting the data
predictions =  lm.predict(X_test)

# %%
plt.scatter(y_test, predictions)

# %%
sns.displot((y_test-predictions), bins=20)

# %%



