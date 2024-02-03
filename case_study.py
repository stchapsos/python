import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

# read data
data = pd.read_csv("weather_data.csv")
data.head(5)

# Question 1
#replace both one and two spaces with NaN
data['MONTH'].replace('  ', np.nan, inplace=True)
data['MONTH'].replace(' ', np.nan, inplace=True)
# fill NaN with DEC
data['MONTH'].fillna('DEC', inplace=True)
# replace empty HIGH values with NaN
data[(data['HIGH'] == ' ')] = data.loc[data['HIGH'] == ' '].replace(' ', np.nan)
# convert it again to float
data['HIGH'] = data['HIGH'].astype(float)
# replace missing values
data['HIGH'].interpolate(method='cubic', inplace=True)
# replace empty HIGH values with NaN
data[(data['LOW'] == ' ')] = data.loc[data['LOW'] == ' '].replace(' ', np.nan)
# convert it again to float
data['LOW'] = data['LOW'].astype(float)
# replace missing values
data['LOW'].interpolate(method='cubic', inplace=True)

# Question 2
# add new line on dataframe
data.loc[365] = ['', '', data['TEMP'].mean(), data['HIGH'].abs().max(), '', data['LOW'].abs().min(), '', data['HDD'].sum(), data['CDD'].sum(), data['RAIN'].sum(), '', data['WINDHIGH'].abs().max(),'','']

# Question 3
print("Standard deviation is :{}".format(data.loc[0:364, 'TEMP'].std()))
print("The median of the temperatures is: {}".format(data.loc[0:364, 'TEMP'].median()))

# Question 4
# print number of days per direction
wind_per_days = data.loc[0:364, 'DIR'].value_counts()
print(wind_per_days)
# Creating plot
fig = plt.figure(figsize=(10, 7))
plt.pie(wind_per_days.values, labels=wind_per_days.index)
# show plot
plt.show()

# Question 5
# print number of days per direction
times_with_max_temp = data.loc[0:364, 'TIME'].value_counts().iloc[:1]
print("Time with most max temperatures is: {}".format(times_with_max_temp.index[0]))
# print number of days per direction
times_with_lower_temp = data.loc[0:364, 'TIME.1'].value_counts().iloc[:1]
print("Time with most min temperatures is: {}".format(times_with_lower_temp.index[0]))

# Question 6
data_without_last_line = data.loc[0:364]
data_without_last_line['DIFF'] = data_without_last_line['HIGH'] - data_without_last_line['LOW']
day = data_without_last_line[data_without_last_line['DIFF'] == data_without_last_line['DIFF'].max()].index.values
print("On day {} we had the max variance on temperature".format(day-1))

# Question 7
print("The direction which the wind followed most of the days of the year is : {}".format(wind_per_days.index[0]))

# Question 8
data_without_last_line = data.loc[0:364]
direction = data_without_last_line[data_without_last_line['WINDHIGH'] == data_without_last_line['WINDHIGH'].max()]
print("The direction which produced the highest wind speed is : {}".format(direction['DIR'].values))

# Question 9
#take only mean temperatures and directions
temp_direction = data.loc[0:364, ['TEMP','DIR']]
# group by direction to take the temperatures per direction
grouped = temp_direction.groupby('DIR')
# take mean temperature per direction
mean_temp = grouped.mean()
print(mean_temp)
# min and max mean temperatures and the direction
print("Direction with max mean temperature is : {}".format(mean_temp['TEMP'][mean_temp['TEMP'] == mean_temp['TEMP'].max()].index[0]))
print("Direction with min mean temperature is : {}".format(mean_temp['TEMP'][mean_temp['TEMP'] == mean_temp['TEMP'].min()].index[0]))

# Question 10
# take only rain and months
rain_month = data.loc[0:364, ['MONTH','RAIN']]
# group by month
grouped_rain = rain_month.groupby('MONTH')
# take the sum of the total rain per month
rain_sum = grouped_rain.sum()
print(rain_sum)
# do the plot
fig = plt.figure(figsize = (10, 5))
labels = rain_sum.index.values
scores = rain_sum.values
# creating the bar plot
plt.bar(labels, list(map(float, scores)), width=0.4)
plt.xlabel("Month")
plt.ylabel("Rain amount")
plt.title("Amount of rain per month")
plt.show()

# Question 11
# take only december
december = data[data['MONTH'] == 'DEC']
# take X,y to fit our regression model
X = december['DAY'].values.reshape(-1, 1)
y = december['TEMP'].values
# fit the linear regression model
regr = LinearRegression().fit(X,y)
# make the prediction for 25/12. Give as input 25
pred = regr.predict(np.array(25).reshape(-1, 1))
print("For 25/12/2018 the mean temperature will be: {}".format(pred))

# Question 12
# create list with corresponding months
winter = ['DEC', 'JAN', 'FEB']
spring = ['MAR', 'APR', 'MAY']
summer = ['JOU', 'JUL', 'AUG']
autumn = ['SEP', 'OCT', 'NOV']

# define labels for values and colors
labels = ['mean', 'max', 'min']
colors = ['green', 'red', 'blue']

# create a list with mean,max,min values for winter
winter_temp = [data['TEMP'][data['MONTH'].isin(winter)].mean(),
data['TEMP'][data['MONTH'].isin(winter)].max(),
data['TEMP'][data['MONTH'].isin(winter)].min()]

# create a list with mean,max,min values for spring
spring_temp = [data['TEMP'][data['MONTH'].isin(spring)].mean(),
data['TEMP'][data['MONTH'].isin(spring)].max(),
data['TEMP'][data['MONTH'].isin(spring)].min()]

# create a list with mean,max,min values for summer
summer_temp = [data['TEMP'][data['MONTH'].isin(summer)].mean(),
data['TEMP'][data['MONTH'].isin(summer)].max(),
data['TEMP'][data['MONTH'].isin(summer)].min()]

# create a list with mean,max,min values for autumn
autumn_temp = [data['TEMP'][data['MONTH'].isin(autumn)].mean(),
data['TEMP'][data['MONTH'].isin(autumn)].max(),
data['TEMP'][data['MONTH'].isin(autumn)].min()]

# Do the plot
fig = plt.figure(figsize = (14, 7))
# Draw first subplot
plt.subplot(2, 2, 1)
plt.bar(labels, winter_temp, color=colors, width=0.4)
plt.title('Winter')

# Draw second subplot
plt.subplot(2, 2, 2)
plt.bar(labels, spring_temp, color=colors, width=0.4)
plt.title('Spring')

# Draw third subplot
plt.subplot(2, 2, 3)
plt.bar(labels, summer_temp, color=colors, width=0.4)
plt.title('Summer')

# Draw forth subplot
plt.subplot(2, 2, 4)
plt.bar(labels, autumn_temp, color=colors, width=0.4)
plt.title('Autumn')

plt.show()

# Question 13
# define the function


def return_rain_quantity_info(sum_of_rain_amount):
    if sum_of_rain_amount < 400:
        return 'Λειψυδρία'
    elif 400 <= sum_of_rain_amount < 600:
        return 'Ικανοποιητικά ποσά βροχής'
    elif sum_of_rain_amount >= 600:
        return 'Υπερβολική βροχόπτωση'


# test function
print(return_rain_quantity_info(250))
print(return_rain_quantity_info(500))
print(return_rain_quantity_info(900))