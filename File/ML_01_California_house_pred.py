import pandas as pd                                                           # Importing modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os                                                                    # Fetching data.
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
fetch_housing_data()

# Load the data
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# Loading the data into housing (A dataFrame)
housing = load_housing_data()
housing.head()
# Describes the dataframe's Features.
housing.info()

housing["ocean_proximity"].value_counts()
# Summarize the dataframe.
housing.describe()
# Creating a histogram.
housing.hist(bins=50,figsize=(20,15))
plt.show()


housing["income_cat"] = pd.cut(housing["median_income"],
 bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
 labels=[1, 2, 3, 4, 5])


# In[10]:


from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)


# In[11]:


# Visualizing Geographical data (Scatter-plot)
housing.plot(kind="scatter",x="longitude",y="latitude",figsize=(10,6))


# ### The above plot looks like california.

# In[12]:


housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1,figsize=(10,6))


# In[13]:


housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,
             s=housing["population"]/100,label="population",figsize=(10,6),
            c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True,)
plt.legend()
# Correlation of each Features with each other.
corr_matrix=housing.corr()
# Correlation of "median_house_value" with other Features.
corr_matrix["median_house_value"].sort_values(ascending=False)
# To plot scatter_matrix using pandas.
from pandas.plotting import scatter_matrix
attributes=["median_house_value","median_income","total_rooms","housing_median_age"]
scatter_matrix(housing[attributes],figsize=(12,8))
# From above we can determine that the most promising attribute to predict the median house value is median income
housing.plot(kind="scatter",x="median_income",y="median_house_value",alpha=0.1)
# Data Preparation.
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
# Experimenting with Attribute Combinations
housing["rooms_per_household"]=housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"]=housing["total_bedrooms"]/housing["total_rooms"]
housing["pop_per_households"]=housing["population"]/housing["households"]

corr_matrix=housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# Data Cleaning
# Dropping every rows with empty data from "total_bedrooms".
housing.dropna(subset=["total_bedrooms"])

housing_cat=housing[["ocean_proximity"]]
housing_cat.head(10)
# Converting the text to numbers.
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder=OrdinalEncoder()
housing_cat_encoded=ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[700:710]
ordinal_encoder.categories_
#  One issue with this representation is that
#  ML algorithms will assume that two nearby values are more similar
#  than two distant values. This may be fine in some cases (e.g., for
#  ordered categories such as “bad”, “average”, “good”, “excellent”),
#  but it is obviously not the case for the ocean_proximity column 
# (for example, categories 0 and 4 are clearly more similar than categories 0 and 1)

# Fixing the issue with One Hot Encoding.
from sklearn.preprocessing import OneHotEncoder
cat_encoder=OneHotEncoder()
housing_cat_one=cat_encoder.fit_transform(housing_cat)
housing_cat_one
# Converting the sparse matrix into numpy 2D array
housing_cat_one.toarray()
cat_encoder.categories_

# Custom Transformers
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
        
        
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
housing_extra_attribs.shape


# ## Feature Scaling
# - Min-Max scaling (Normalization)
# - Standardization
 ## Transformation Pipelines
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler  #Standardization

housing_num=housing.drop("ocean_proximity",axis=1)

num_pipeline=Pipeline([('imputer',SimpleImputer(strategy="median")),
                       ('attribs_adder',CombinedAttributesAdder()),
                       ('std_scaler',StandardScaler()),])
housing_num_tr=num_pipeline.fit_transform(housing_num)



# ColumnTransformer can handle both categorial columns and numerical columns at same time.
from sklearn.compose import ColumnTransformer
num_attribs=list(housing_num)
cat_attribs=["ocean_proximity"]
full_pipeline=ColumnTransformer([("num",num_pipeline,num_attribs),
                                 ("cat",OneHotEncoder(),cat_attribs),])
housing_prepared=full_pipeline.fit_transform(housing)
housing_prepared[:5]


# Train a model
# RandomForestRegressor.
from sklearn.ensemble import RandomForestRegressor
fr=RandomForestRegressor()
fr_score=cross_val_score(fr,housing_prepared,housing_labels,scoring="neg_mean_squared_error",cv=10)
fr_rmse_score=np.sqrt(-fr_score)
display_scores(fr_rmse_score)
# Fine-tune Your Model
# Best combination of hyperparameter values for RandomForestRegressor.
from sklearn.model_selection import GridSearchCV
param_grid=[{'n_estimators':[3,10,30],'max_features':[2,4,6,8]},
           {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]},]
fr=RandomForestRegressor()
grid_search=GridSearchCV(fr,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(housing_prepared,housing_labels)
grid_search.best_estimator_