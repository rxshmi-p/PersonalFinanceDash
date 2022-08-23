# Personal Finance Dashboard 

The goal of this Personal Finance Dashboard is to have a better understanding of spending habits and predict future spending on a monthly basis. As a result, the aim of this better understanding of spending is to have more realistic budgets and reduce spending where possible.

## Background

I have created a Python based web application dashboard on Streamlit to gain better insight of my spending habits. This web app is using personal spending data that I have collected for 9 months to analyze and visualize spending patterns, as well as predict future spending. 

## How it Works 
Page 1: Dashboard

Upon setup the user has the option to view mode data being used by selecting "Show model data". The first part of this app will show two summmary plots, aggregating total spending data on a monthly basis as well as total amounts spent by category. Then we can choose the types of spending to compare on a monthly basis. This page is a great overview of spending patterns and would point out any obvious anomolies.

![Screen Shot 2022-08-21 at 10 49 17 AM](https://user-images.githubusercontent.com/86248667/185796914-a3559571-1e81-4968-9552-dad71b2ad473.png)
![Screen Shot 2022-08-21 at 10 51 44 AM](https://user-images.githubusercontent.com/86248667/185796995-fbbebb85-24f4-4884-a1a9-3f1736c2768f.png)

Page 2: Data Exploration 

This page is the experimental stage where the data is checked and explored before prediction. The time series data is modelled in comparison to monthly mean, then decomposed to find patterns or trends, and finally checked for stationarity as time series data must be stationary for accurate predictions. Stationarity is checked with visualization and the Augmented Dickey-Fuller Test. 
![Screen Shot 2022-08-21 at 10 53 03 AM](https://user-images.githubusercontent.com/86248667/185797301-627b99f1-5bff-4be0-bb49-d05d4723fa00.png)
![Screen Shot 2022-08-21 at 10 53 38 AM](https://user-images.githubusercontent.com/86248667/185797308-152e21ea-85d4-4eff-92a2-be74df8382ec.png)

Page 3: Prediction

After exploring the data I can now predict future spending. After testing a Simple Exponential Smooting (SES) model, Holt-Winters Additive Model and Holt-Winters Multiplicative Model, I was able to conclude that the Holt-Winter Multiplicative Model is the most accurate prediction. Using this model the user has the option to have the model predict spending between five to ten months. This is meant to represent what spending can be expected, what a realistic budget could look like and an indicator of possibly needing to save more. 

![Screen Shot 2022-08-21 at 11 00 02 AM](https://user-images.githubusercontent.com/86248667/185797346-6024060c-3789-4e1d-938e-9ab718a69132.png)
![Screen Shot 2022-08-21 at 11 00 21 AM](https://user-images.githubusercontent.com/86248667/185797358-f4a24272-4702-4bc5-b64b-4f97ed5b95f8.png)


## How to Run 
1. Clone the repositiory 
```
$ git clone git@github.com:rxshmi-p/PersonalFinanceDash
$ cd Dashboard
```
2. Install dependencies:
```
$ pip install -r requirements.txt
```
3. Start the application:
```
$ streamlit run Dashboard.py
```

## References 

Some Python codes and project structure was inspired by the resources below: 
https://www.statsmodels.org/dev/examples/notebooks/generated/exponential_smoothing.html
https://www.bounteous.com/insights/2020/09/15/forecasting-time-series-model-using-python-part-one/

