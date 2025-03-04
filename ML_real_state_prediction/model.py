import matplotlib_inline
import numpy as np 
import pandas as pd 

housing = pd.read_csv('data.csv')
#print( f"housing data : {housing.head()}")
#print( f"housing info : {housing.info()}")
#print( f"housing describe : {housing.describe()}")
#print( f"housing value_count : {housing['CHAS'].value_counts()}")

# to visualize the data
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20, 15))
#plt.show()

# Train-Test Splitting
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42 , shuffle=True)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}")


#getting same percentage of data for each category
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# looking for corelation
corr_matrix = housing.corr()    
sorted_corelation = corr_matrix['MEDV'].sort_values(ascending=False)  # this shows with respect to MEDV which column are directly or indirectly perpotional

# Filter correlations greater than 0.5 or less than -0.5
attributes = sorted_corelation[(sorted_corelation > 0.5) | (sorted_corelation < -0.5)]
print(f"filtered_correlation : {attributes}")

#importing pandas.plotting module
from pandas.plotting import scatter_matrix
scatter_matrix(housing[attributes.index], figsize=(12, 8))
#plt.show()

#training dataset 
housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()

#creating a pipeline 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 

my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
])
