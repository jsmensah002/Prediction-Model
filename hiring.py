#File named 'hiring.csv' is imported using pandas
import pandas as pd
hiring_data = pd.read_csv('hiring.csv')
print(hiring_data)

#The experience column and the test_score(out of 10) column both have null values
#So we need to fill those null values

#Filling the null values in the test_score columnm using the median
median_testscore = hiring_data['test_score(out of 10)'].median()
print(median_testscore)

#Replacing the null value with the calculated median value for the test_score(out of 10) column
hiring_data['test_score(out of 10)'] = hiring_data['test_score(out of 10)'].fillna(median_testscore)
print(hiring_data)

#For the experience column, we first fill the null value with zero, then change the words to numbers using word2number.w2n as w2n
hiring_data['experience'] = hiring_data['experience'].fillna('zero')
print (hiring_data)

#Converting the experience column to numbers 
import word2number.w2n as w2n
hiring_data['experience'] = hiring_data['experience'].apply(w2n.word_to_num)
print(hiring_data)

#Calculating the median of the experience column, and using it to replace the zero values we initially inserted
median_experience = hiring_data['experience'].median()
print (median_experience)

#Replacing the median experience value with the 0 we initially used
hiring_data['experience'] = hiring_data['experience'].replace(value=median_experience, to_replace=0)
print(hiring_data)

# Linear regression (reg, short form of regression)
import sklearn.linear_model as linear_model
reg = linear_model.LinearRegression()

#Defining x and y
#We have 3 x values here : experience, test_score(out of 10), and interview_score(out of 10)
#And a single y value : salary($)
x = hiring_data[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']].values
y = hiring_data['salary($)'].values
reg.fit(x,y)

#Calculating the slope and intercept (y0 = mx + c, where m is slope and c is the intercept)
slope = reg.coef_
print(slope)
intercept = reg.intercept_
print(intercept)

#Predicting the salary($) using experience, test_score(out of 10), and interview_score(out of 10)
#Question 1: With a 9 year working experience, a test score of 9, and an interview score of 8, what would be the predicted salary?
prediction1 = reg.predict([[9,9,8]])
print(prediction1)

#Question 1: With a 3 year working experience, a test score of 5, and an interview score of 6, what would be the predicted salary?
prediction2 = reg.predict([[3,5,6]])
print(prediction2)

#Saving the prediction model using joblib
import joblib
joblib.dump(reg, 'Prediction Model')

#Loading the prediction model using joblib
load_model = joblib.load('Prediction Model')

#Using the prediction model. Eg predict the salary when the experience is 5, test score is 3, and interview score is 9
prediction3 = load_model.predict([[5,3,9]])
print(prediction3)

#If your prediction is based on multiple criteria, include it in the x definition
# e.g. If the salary depends on 4 columns
#The x becomes: x = hiring_data[['x1','x2','x3','x4']].values
#And y remains the same: y = hiring_data['y1'].values 

