
# coding: utf-8

# ## Multiple Linear Regression - 50 Startups
# 
# ### Project Description:
# 
# You got 50 companies in total New York and Florida and what they have is they have some extracts from their profit and loss statements from their income report.
# 
# So How much did the company in this given financial year that you're analyzing and for how much in that year did it spend on research and development?
# 
# How much in that year did it spend on Administration?
# 
# How much in that year did it spend on Marketing?and In Which State the most?
# 
# And Finally By spending on which department the company got more profits?

# ### Imports

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## Get the Data

# We'll work on 50_Startups csv file. It has Financial Year Profit or loss information,Expenditure details.It got numercial value columns.
# 
# - R&D Spend : R&D Expenditures
# - Administration : Administration Expenditures
# - Marketing Spend : Marketing expenditure
# - Profit : Profit/ Loss Details

# **Read in the 50_Startups csv file as a DataFrame called startups.**

# In[3]:


startups = pd.read_csv('50_Startups.csv')


# In[4]:


startups.head()


# In[5]:


startups.describe()


# In[6]:


startups.info()


# ## Exploratory Data Analysis
# 
# #### Let's explore the data!
# 
# For the rest of the exercise we will only be using the numerical data of the csv file.
# 
# **Use Seaborn to create a joinplot to compare the R&D Spend and Profit columns. Does the correlation make sense?**

# In[7]:


sns.jointplot(data=startups, x='R&D Spend', y='Profit')


# In[8]:


sns.jointplot(data=startups, x='Administration', y='Profit')


# In[9]:


sns.jointplot(data=startups, x='Marketing Spend', y='Profit')


# In[10]:


sns.jointplot(data=startups, x='R&D Spend', y='Profit', kind='hex')


# In[11]:


sns.jointplot(data=startups, x='Administration', y='Profit', kind='hex')


# In[12]:


sns.jointplot(data=startups, x='Marketing Spend', y='Profit', kind='hex')


# In[13]:


sns.pairplot(startups)


# ### Based off this plot what looks to be the most correlated feautre with profit?
# 
# ### Answer: R&D Spend

# ---

# **Create a linear model plot(using seaborn's implot) of Profit vs R&D Spend**

# In[14]:


sns.lmplot(x='R&D Spend', y='Profit', data=startups)


# ### Training and Testing Data
# 
# Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets.
# 
# ### Set a variable X equal to the numerical features of the startups and a variable Y equal to the "Profit" column.

# In[15]:


startups.columns


# In[40]:


X = startups[['R&D Spend', 'Administration', 'Marketing Spend']]


# In[41]:


Y = startups['Profit']


# **Use Cross_validation, train_test_split from sklearn to split the data into training and testing sets. Set test Size = 0.3 and random state=101**

# In[42]:


from sklearn.model_selection import train_test_split


# In[43]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=0)


# ---

# ## Training the Model
# 
# Now its time to train our model on our training data!
# 
# **Import LinearRegression from sklear.linear_model**

# In[44]:


from sklearn.linear_model import LinearRegression


# In[45]:


regressor = LinearRegression()


# **Train/fit regressor on the training data**

# In[46]:


regressor.fit(X_train, Y_train)


# **Print out the coefficients of the model**

# In[48]:


regressor.coef_


# ## Predicting Test Data

# Now that we fit our model, lets evaluate its performance by predicting off the test values!
# 
# **Use regressor.predict() to predict off the X_test of the data.**

# In[49]:


predictions = regressor.predict(X_test)


# **Create a Scatterplot of the real test values versus the predicted values.**

# In[50]:


plt.scatter(Y_test, predictions)
plt.xlabel('Y_test(True Values)')
plt.ylabel('Predicted Values')


# ## Evaluating the model
# 
# Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score(R^2).
# 
# **Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error. Refer to Wikipedia for the formulas.**

# In[51]:


from sklearn import metrics


# In[52]:


print('MAE', metrics.mean_absolute_error(Y_test, predictions))
print('MSE', metrics.mean_squared_error(Y_test, predictions))
print('RMSE', np.sqrt(metrics.mean_squared_error(Y_test, predictions)))


# In[53]:


metrics.explained_variance_score(Y_test, predictions)


# ## Residuals
# 
# You should have gotten a very good model with a good fit. Lets quickly explore the residuals to make sure everything was okay with our data.
# 
# **Plot a histogram of the residuals and make sure it looks normally distributed. Use either seaborn distplot, or just plt.hist()**

# In[57]:


sns.distplot((Y_test-predictions), bins=7)


# ## Conclusion
# 
# We still want to figure out the answer to the original question, On which departments companies are spending much amount to bring better profits to the company? 
# 
# Let's see if we can interpret the coefficients at all to get an idea.
# 
# ### Recreate the dataframe below.

# In[58]:


cdf = pd.DataFrame(regressor.coef_, X.columns, columns=['Coeff'])
cdf


# In[59]:


cdf.to_csv('Results.csv')


# **What do you think, on which department the companies are spending more and by which they are getting more profits?**
# 
# The coefficients showing that, companies are spending more on R&D department and since they are getting more profits through that department, it is suggested to put more focus on that department for more better results.
