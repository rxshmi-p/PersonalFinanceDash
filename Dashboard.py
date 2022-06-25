# %%
from pickle import FALSE
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# %%
### Importing Data
# must update total months and new dataset name 
names = ["Sept 2021.csv", 'Oct 2021.csv', 'Nov 2021.csv', 'Dec 2021.csv', 'Jan 2022.csv', 'Feb 2022.csv', 'March 2022.csv',
         'April 2022.csv']
path = '/Users/rashmipanse/Documents/Projects/Budget files/Monthly Budget  - '

monthly  = []
for i in range(0,len(names)):
    monthly.append(pd.read_csv(path + names[i]))

#%%
### Cleaning Dataframes 

# drop unnecessary columns
cols = [5,6,7,8,9]
for dataset in monthly:
    dataset.drop(dataset.columns[cols], axis=1, inplace = True)
    dataset.dropna(inplace = True)

# %% 
# reformat date column to date datatype
months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6, 'July': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
year=0
for df in monthly:
    year+=1
    for i in range(0,len(df['Date '])):
        day=list(df['Date '].iloc[i].split('-'))
        for month,val in months.items():
            if month == day[1]:
                df['Date '].iloc[i] = day[0]+"/"+str(val)+"/"+names[year-1][-6:-4]

# %%
### Combine monthly datasets, convert to datetime, set date as index 
res = pd.concat(monthly)
res['Date ']= pd.to_datetime(res['Date '])
res = res.set_index('Date ')

# %% 
### Selecting time period for monthly aggregation 
res = res['2021-09-01':'2022-05-29'].resample('M').sum()



# %%
data = res.drop(columns = ['Amount Spent ']) 
target = res['Amount Spent ']

# Visualizing Data 
date_plot = res.plot(x= 'Date ')
plt.show()
ty_plot = res['Type'].value_counts().plot(kind = 'bar')
plt.show()


# %%
# Split data into train and test data (75% train, 25% test)
x_train, x_test, y_train, y_test = train_test_split(data, target, random_state=1)
#print(x_train.shape)
#print(x_test.shape)
#print(y_train.shape)
#print(y_test.shape)

#%%
my_lm = LinearRegression().fit(data,target)
print(my_lm)

# %%
