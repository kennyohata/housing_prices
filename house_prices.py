# Import all the libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import norm, skew
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer

import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

# Import the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Check it out
train.head(5)

# Check out the size and shape of the data set
train.shape
test.shape

# Save the 'Id' column and drop the "Id" column because it is not necessary moving forward
train_id = train['Id']
test_id = test['Id']

train.drop('Id', axis = 1, inplace = True)
test.drop('Id', axis = 1, inplace = True)

# Check out the data one more time
train.shape
test.shape

# Clean the data, explore any outliers
# Check out 'GrLivArea'
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

# Check out 'TotalBsmtSF'
fig, ax = plt.subplots()
ax.scatter(x = train['TotalBsmtSF'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('TotalBsmtSF', fontsize=13)
plt.show()

# Delete outliers. In GrLivArea, (['GrLivArea']>4000) & (['SalePrice']<300000) and in TotalBsmtSF, (TotalBsmtSF > 4000)
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
train = train.drop(train[(train['TotalBsmtSF']>4000)].index)

# Check out 'GrLivArea'
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

# Check out 'TotalBsmtSF'
fig, ax = plt.subplots()
ax.scatter(x = train['TotalBsmtSF'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('TotalBsmtSF', fontsize=13)
plt.show()

# Let's do some exploratory data analysis
train['SalePrice'].describe()

# Plot the SalesPrice data
sns.distplot(train['SalePrice'])

# Build a botplot comparing overall quality and salesprice
overall_quality = 'OverallQual'
data = pd.concat([train['SalePrice'], train[overall_quality]], axis=1)
f, ax = plt.subplots(figsize=(10, 8))
fig = sns.boxplot(x=overall_quality, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);

year_built = 'YearBuilt'
data = pd.concat([train['SalePrice'], train[year_built]], axis=1)
f, ax = plt.subplots(figsize=(1, 10))
fig = sns.boxplot(x=year_built, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);


# Build a correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(13, 10))
sns.heatmap(corrmat, vmax=1, square=True);

# Plot a scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 3)
plt.show();

# Check for missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=percent.index, y=percent)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)

# Delete unnecessary missing data
train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)
train = train.drop(train.loc[train['Electrical'].isnull()].index)
train.isnull().sum().max()

# Search for normality. Create a histogram and normal probability plot
sns.distplot(train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)

# Applying the log transformation and replot 
train['SalePrice'] = np.log(train['SalePrice'])
sns.distplot(train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)

# Apply the similar histogram and normal probability plot for 'GrLivArea'
sns.distplot(train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['GrLivArea'], plot=plt)

train['GrLivArea'] = np.log(train['GrLivArea'])
sns.distplot(train['GrLivArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(train['GrLivArea'], plot=plt)

# Apply the similar histogram and normal probability plot for 'TotalBsmtSF'
sns.distplot(train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['TotalBsmtSF'], plot=plt)

# Create a column for new variable 
# If area>0 it gets 1, for area==0 it gets 0
train['HasBsmt'] = pd.Series(len(train['TotalBsmtSF']), index=train.index)
train['HasBsmt'] = 0 
train.loc[train['TotalBsmtSF']>0,'HasBsmt'] = 1

# Applying the log transformation and replot 
train.loc[train['HasBsmt']==1,'TotalBsmtSF'] = np.log(train['TotalBsmtSF'])
sns.distplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)

# Create a scatter plot for GrLivArea
plt.scatter(train['GrLivArea'], train['SalePrice']);

# Convert categorical variable into dummy
train = pd.get_dummies(train)

# Start modeling 
train.shape

# Split the data to train the model
X_train,X_test,y_train,y_test = train_test_split(train,train.index,test_size = 0.3,random_state= 0)

X_train.shape,X_test.shape,y_train.shape,y_test.shape

train.head(5)

from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold

n_folds = 5
scorer = make_scorer(mean_squared_error,greater_is_better = False)

def rmse_CV_train(model):
    kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model,X_train,y_train,scoring ="neg_mean_squared_error",cv=kf))
    return (rmse)

def rmse_CV_test(model):
    kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model,X_test,y_test,scoring ="neg_mean_squared_error",cv=kf))
    return (rmse)

lr = LinearRegression()
lr.fit(X_train,y_train)
test_pred = lr.predict(X_test)
train_pred = lr.predict(X_train)
print('rmse on train',rmse_CV_train(lr).mean())
print('rmse on test',rmse_CV_test(lr).mean())

# Plot between predicted values and residuals
plt.scatter(train_pred, train_pred - y_train, c = 'green',  label = 'Training data')
plt.scatter(test_pred,test_pred - y_test, c = 'blue',  label = 'Validation data')
plt.title('Linear regression')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc = 'best')
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = 'red')
plt.show()

# Plot predictions - Real values
plt.scatter(train_pred, y_train, c = 'green',  label = 'Training data')
plt.scatter(test_pred, y_test, c = 'blue',  label = 'Validation data')
plt.title('Linear regression')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc = 'best')
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = 'red')
plt.show()
