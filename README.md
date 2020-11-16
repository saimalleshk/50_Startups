# 50_Startups

## Project Description:
(Data from kaggle)
You got 50 companies in total New York and Florida and what they have is they have some extracts from their profit and loss statements from their income report.

So How much did the company in this given financial year that you're analyzing and for how much in that year did it spend on research and development?

How much in that year did it spend on Administration?

How much in that year did it spend on Marketing?and In Which State the most?

And Finally By spending on which department the company got more profits?

## Import the Dataset

## Get the Data

We'll work on 50_Startups csv file. It has Financial Year Profit or loss information,Expenditure details.It got numercial value columns.

R&D Spend : R&D Expenditures
Administration : Administration Expenditures
Marketing Spend : Marketing expenditure
Profit : Profit/ Loss Details

## Exploratory Data Analysis
Let's explore the data!
For the rest of the exercise we will only be using the numerical data of the csv file.

### Use Seaborn to create a joinplot to compare the R&D Spend and Profit columns. Does the correlation make sense?

## Based off this plot what looks to be the most correlated feautre with profit?

## Training and Testing Data
Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets.

Set a variable X equal to the numerical features of the startups and a variable Y equal to the "Profit" column.

## Training the Model
Now its time to train our model on our training data!

Import LinearRegression from sklear.linear_model

## Predicting Test Data¶

Now that we fit our model, lets evaluate its performance by predicting off the test values!

Use regressor.predict() to predict off the X_test of the data.

## Evaluating the model

Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score(R^2).

Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error. Refer to Wikipedia for the formulas.

## Residuals

Lets quickly explore the residuals to make sure everything was okay with our data.

## Conclusion

We still want to figure out the answer to the original question, On which departments companies are spending much amount to bring better profits to the company?

Let's see if we can interpret the coefficients at all to get an idea.

## What do you think, on which department the companies are spending more and by which they are getting more profits?
