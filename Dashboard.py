# %%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
import streamlit as st
import statsmodels.api as sm
import plotly.graph_objects as go
from statsmodels.tsa.api import SimpleExpSmoothing 
from statsmodels.tsa.api import ExponentialSmoothing

# %%
# Page Configuration 
st.set_page_config(
    page_title = 'Financial Dashboard',
    layout = 'wide'
)

st.title('Financial Dashboard')

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
cols = [1,5,6,7,8,9]
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
df = res.copy()
res = res.set_index('Date ').sort_index()
res = res.loc['2021-08-01':'2022-06-01']

if st.checkbox('Show model data'):
    st.subheader('Model data')
    st.write(res)

# %% 
# Plots  
st.subheader('Summary Plots')

col1, col2 = st.columns(2)

with col1:
    st.subheader("Total Amount Spent Over Time")
    st.line_chart(res['Amount Spent '], use_container_width=True)

with col2:
    st.header("Total Spending by Type")
    ty_plot = res['Type'].value_counts()
    st.bar_chart(ty_plot)




# %% 
### Selecting time period for monthly aggregation and reducing dataframe to df to prepare for time series model
spent_by_month = res['2021-07-01':'2022-05-30'].resample('M').sum()

# %%

df2 = df.groupby(['Date ','Type'], as_index=False)['Amount Spent '].sum()
df2.groupby([df2['Date '].dt.to_period('M'), 'Type']).sum().reset_index()

clist = df2["Type"].unique().tolist()
types = st.multiselect("Select the types of spending you would like to compare", clist)
st.header("You selected: {}".format(", ".join(types)))      

dfs = {type: df2[df2["Type"] == type] for type in types}

fig = go.Figure()
for type, df2 in dfs.items():
    fig = fig.add_trace(go.Scatter(x=df2["Date "], y=df2["Amount Spent "], name=type))

st.plotly_chart(fig, use_container_width=True)


tab1, tab2 = st.tabs(["📈 Data Exploration", "🗃 Predictions"])

tab1.header("Data Exploration")

tab1.markdown("We will now further explore the data in preparation for time series modelling.")

tab1.subheader("Total Monthly Spending with Monthly Mean Resample")
# This is monthly spending and mean resampling on monthly basis
fig, ax = plt.subplots(figsize=(20, 6))
ax.plot(res['Amount Spent '],marker='.', linestyle='-', linewidth=0.5, label='Monthly')
ax.plot(res['2021-07-01':'2022-05-30'].resample('M').mean(),marker='o', markersize=8, linestyle='-', label='Monthly Mean Resample')
ax.set_ylabel('Total Spent')
ax.legend()
tab1.pyplot(fig)
tab1.markdown("this plot allows us to visualize the spending on a monthly basis in comparison to the average amount spent that month. It is important to check time series data for patterns that may affect the results, and can inform which forecasting model to use. Depending on the stability of the mean line we can judge to volatility of an individuals spending")
# get mean, median, mode based on Type of spending?
# Explain whats happening in this plot and what we're using it for - seasonal decompose & stationarity


# %%
# graphs to show seasonal_decompose
# Examined model for patterns:
# Level (avg value in series) - increases and then drops  after a peak, seen pattern twice
# Trend (increases, decreases, stays same over time)
# Seasonal/Periodic
# Cyclical (increase/decrease non-seasonal related, like business cycles)
# Random/Irregular variations

tab1.subheader("Decomposition")

tab1.markdown("First I start by looking for patterns in the model. I do this by decomposing the data using the seasonal_decompose function, within the 'statsmodel' package, to view more of the complexity behind the linear visualization. This function helps to decompose the data into the four common time series data patterns; Observed, Trended, Seasonal, Residual.")

def seasonal_decompose (y):
    decomposition = sm.tsa.seasonal_decompose(y, model='additive',extrapolate_trend='freq', period=1)
    fig = decomposition.plot()
    fig.set_size_inches(14,7)
    tab1.pyplot(fig)

df = pd.DataFrame(res['Amount Spent '])
seasonal_decompose(df)

# %% 
# Stationarity - must check if data is stationary 
### plot for Rolling Statistic for testing Stationarity

tab1.subheader("Stationarity")
tab1.markdown("Next I check the data for stationarity. Data is statinary when its statistical properties do not change much over time. It is important to have stationary data when building a time series forecasting model to make accurate predictions. I check stationarity using isualization and the Augmented Dickey-Fuller (ADF) Test.")

def test_stationarity(timeseries, title):
    
    #Determing rolling statistics
    rolmean = pd.Series(timeseries).rolling(window=12).mean() 
    rolstd = pd.Series(timeseries).rolling(window=12).std()
    
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(timeseries, label= title)
    ax.plot(rolmean, label='rolling mean')
    ax.plot(rolstd, label='rolling std (x10)')
    ax.legend()
    tab1.pyplot(fig)
pd.options.display.float_format = '{:.8f}'.format

# %%
test_stationarity(df['Amount Spent '],'raw data')

tab1.markdown("Using the test_stationarity function we can see the rolling statistics (mean and variance) at a glance to determine how drastically the standard deviation over time. Since both mean and standard deviation do not change much over time we can assume they are stationary, but can further use the ADF for more confidence")
# %% 
# Augmented Dickey-Fuller Test

from statsmodels.tsa.stattools import adfuller

tab1.subheader("Augmented Dickey-Fuller Test")

def ADF_test(timeseries, dataDesc):
    tab1.text(' > Is the {} stationary ?'.format(dataDesc))
    dftest = adfuller(timeseries.dropna(), autolag='AIC')
    tab1.text('Test statistic = {:.3f}'.format(dftest[0]))
    tab1.text('P-value = {:.3f}'.format(dftest[1]))
    tab1.text('Critical values :')
    for k, v in dftest[4].items():
        tab1.text('\t{}: {} - The data is {} stationary with {}% confidence'.format(k, v, 'not' if v<dftest[0] else '', 100-int(k[:-1])))

# %%
ADF_test(df,'raw data')
# Using ADF we can see the data is stationary 
tab1.markdown("Using the ADF test we can be confident the data is stationary ")


############

# %%
y_to_train = spent_by_month['2021-07-01':'2022-01-01'] # dataset to train
y_to_val = spent_by_month['2022-01-02':'2022-05-30'] # last X months for test  
predict_date = len(spent_by_month) - len(spent_by_month['2022-01-02':'2022-05-30']) # the number of data points for the test set

# %%

tab2.header("Predicting Future Spending")

tab2.subheader("Choosing a Model: Simple Exponential Smoothing (SES)")

ses_fit = SimpleExpSmoothing(spent_by_month, initialization_method="estimated").fit(
    smoothing_level=0.8, optimized=False
)
ses_forecast = ses_fit.forecast(4).rename(r"$\alpha=0.8$")
fig, ax = plt.subplots(figsize=(8, 3))
(line3,) = plt.plot(ses_forecast, marker="o", color="red")
ax.plot(spent_by_month, marker="o", color="black")
ax.plot(ses_fit.fittedvalues, marker="o", color="red")
ax.legend([line3], [ses_forecast.name])
tab2.pyplot(fig)

tab2.text(ses_forecast)

# Not a good rep bc forecast is linear so this is not a good fit for our model 
# %%
tab2.subheader("Choosing a Model: Holt - Winters")


hw_add = ExponentialSmoothing(
    spent_by_month,
    seasonal_periods=4,
    trend="add",
    seasonal="add",
    use_boxcox=True,
    initialization_method="estimated",
).fit()
hw_mul = ExponentialSmoothing(
    spent_by_month,
    seasonal_periods=4,
    trend="mul",
    seasonal="mul",
    use_boxcox=True,
    initialization_method="estimated",
).fit()
ax = spent_by_month.plot(
    figsize=(8, 3),
    marker="o",
    color="black",
    title="Forecasts from Holt-Winters' multiplicative method",
)
ax.set_ylabel("Amount Spent ($)")
ax.set_xlabel("Month")
hw_add.fittedvalues.plot(ax=ax, style="--", color="red")
hw_mul.fittedvalues.plot(ax=ax, style="--", color="green")
hw_add.forecast(12).rename("Holt-Winters (add-add-seasonal)").plot(
    ax=ax, style="--", marker="o", color="red", legend=True
)
hw_mul.forecast(12).rename("Holt-Winters (add-mul-seasonal)").plot(
    ax=ax, style="--", marker="o", color="green", legend=True
)

st.set_option('deprecation.showPyplotGlobalUse', False)
tab2.pyplot()


# %%

# taking a look at the model metrics, comparing the HW Additive method vs the HW Multiplicative Method

results = pd.DataFrame(
    index=[r"$\alpha$", r"$\beta$", r"$\phi$", r"$\gamma$", r"$l_0$", "$b_0$", "SSE"]
)
params = [
    "smoothing_level",
    "smoothing_trend",
    "damping_trend",
    "smoothing_seasonal",
    "initial_level",
    "initial_trend",
]
results["Additive"] = [hw_add.params[p] for p in params] + [hw_add.sse]
results["Multiplicative"] = [hw_mul.params[p] for p in params] + [hw_mul.sse]
tab2.dataframe(results) 

# since multiplicative has the lowest sum of squared estimate of error we will using this method.

# %%


with st.container():
    m = st.slider('How many months would you like to forecast?', 5, 10, 5)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Forecast - Additive Method")
        add_forecast = hw_add.forecast(m)
        st.dataframe(add_forecast)

    with col2:
        st.subheader("Forecast - Multiplicative Method")
        mul_forecast = hw_mul.forecast(m)
        st.dataframe(mul_forecast)

