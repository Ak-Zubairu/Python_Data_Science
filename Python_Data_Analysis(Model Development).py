#!/usr/bin/env python
# coding: utf-8

# <h1 align=center><font size = 5>Data Analysis with Python</font></h1>

# ### House Sales in King County, USA
# 

# This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015.
# 

# | Variable      | Description                                                                                                 |
# | ------------- | ----------------------------------------------------------------------------------------------------------- |
# | id            | A notation for a house                                                                                      |
# | date          | Date house was sold                                                                                         |
# | price         | Price is prediction target                                                                                  |
# | bedrooms      | Number of bedrooms                                                                                          |
# | bathrooms     | Number of bathrooms                                                                                         |
# | sqft_living   | Square footage of the home                                                                                  |
# | sqft_lot      | Square footage of the lot                                                                                   |
# | floors        | Total floors (levels) in house                                                                              |
# | waterfront    | House which has a view to a waterfront                                                                      |
# | view          | Has been viewed                                                                                             |
# | condition     | How good the condition is overall                                                                           |
# | grade         | overall grade given to the housing unit, based on King County grading system                                |
# | sqft_above    | Square footage of house apart from basement                                                                 |
# | sqft_basement | Square footage of the basement                                                                              |
# | yr_built      | Built Year                                                                                                  |
# | yr_renovated  | Year when house was renovated                                                                               |
# | zipcode       | Zip code                                                                                                    |
# | lat           | Latitude coordinate                                                                                         |
# | long          | Longitude coordinate                                                                                        |
# | sqft_living15 | Living room area in 2015(implies-- some renovations) This might or might not have affected the lotsize area |
# | sqft_lot15    | LotSize area in 2015(implies-- some renovations)                                                            |
# 

# In[1]:


get_ipython().system('pip3 install scikit-learn --upgrade --user')


# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# # Module 1: Importing Data Sets
# 

# In[20]:


file_name='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
df=pd.read_csv(file_name)


# In[22]:


df.head()


# Displaying the data types of each column using the function dtypes.

# In[24]:


print(df.dtypes)


# Using describe to obtain a statistical summary of the dataframe.

# In[5]:


df.describe()


# # Module 2: Data Wrangling
# 

# Dropping columns <code>"id"</code>  and <code>"Unnamed: 0"</code> from axis 1 using the method <code>drop()</code>.

# In[27]:


df = df.drop(columns=["id", "Unnamed: 0"], axis=1)
df.describe()


# In[60]:


# Checking Null Values
print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# Replacing the missing values of the column <code>'bedrooms'</code> with the mean of the column  <code>'bedrooms' </code> using the method <code>replace()</code>.
# 

# In[43]:


mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)


# Replacing the missing values of the column <code>'bathrooms'</code> with the mean of the column  <code>'bathrooms' </code> using the method <code>replace()</code>.

# In[44]:


mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)


# In[45]:


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# # Module 3: Exploratory Data Analysis
# 

# Using the method <code>value_counts</code> to count the number of houses with unique floor values and the method <code>.to_frame()</code> to convert it to a dataframe.
# 

# In[28]:


floor_counts = df['floors'].value_counts().to_frame()
floor_counts


# Using the function <code>boxplot</code> in the seaborn library  to  determine whether houses with a waterfront view or without a waterfront view have more price outliers.

# In[29]:


sns.boxplot(x="waterfront", y="price", data=df)


# Using the function <code>regplot</code>  in the seaborn library  to  determine if the feature <code>sqft_above</code> is negatively or positively correlated with price.

# In[30]:


sns.regplot(x="sqft_above", y="price", data=df, line_kws={"color": "red"})
plt.ylim(0,)


# # Module 4: Model Development
# 

# Fitting a linear regression model to predict the <code>'price'</code> using the feature <code>'sqft_living'</code> then calculating the R^2.
# 

# In[38]:


Z = df[['sqft_living']]
Y = df['price']
lm = LinearRegression()
lm.fit(Z,Y)
lm.score(Z, Y)


# Fitting a linear regression model to predict the <code>'price'</code> using the list of features

# In[47]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]     

U = df[features]
Y = df['price']
lm = LinearRegression()
lm.fit(U,Y)


# In[48]:


lm.score(U,Y)


# Creating a pipeline object to predict the 'price'.

# In[56]:


Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]


# In[50]:


pipeline = Pipeline(Input)
pipeline.fit(U, Y)
pipeline.score(U, Y)


# # Module 5: Model Evaluation and Refinement
# 

# In[51]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
print("done")


# Split the data into training and testing sets:
# 

# In[52]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# Creating and fitting a Ridge regression object using the training data, regularization parameter as 0.1. Calculating the $R^2$ using the test data.
# 

# In[53]:


from sklearn.linear_model import Ridge


# In[54]:


r_model = Ridge(alpha=0.1)
r_model.fit(x_train, y_train)
r_model.score(x_test, y_test)


# A second order polynomial transform on both the training data and testing data. 
# Creating and fitting a Ridge regression object using the training data.

# In[59]:


scale = StandardScaler()
x_train1 = scale.fit_transform(x_train)
x_test1 = scale.transform(x_test)
        
# Perform second order polynomial transformation
polynomial = PolynomialFeatures(degree=2, include_bias=False)
x_train_poly = polynomial.fit_transform(x_train1)
x_test_poly = polynomial.transform(x_test1)

r_model2 = Ridge(alpha=0.1)
r_model2.fit(x_train_poly, y_train)
r_model2.score(x_test_poly, y_test)

