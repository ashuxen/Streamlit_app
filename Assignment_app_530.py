#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 09:59:33 2022

@author: ashutoshkumar
"""

import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd
import numpy as np
st.set_page_config(page_title='Model Hyperparameter Tuning App',layout='wide') 
#Setting Page layout, Page expands to full width
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.model_selection import RandomizedSearchCV
from IPython.display import Image
from sklearn import metrics
from sklearn.metrics import r2_score,mean_squared_error
# we will use Gradient Boosting model as this gives us best result 
# from sklearn.ensemble import GradientBoostingRegressor
import xgboost
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
import plotly.graph_objects as go #to plot visualization
import plotly.express as px
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns
st.title("Model Hyperparameter Tuning App")
st.write("By:-Ashutosh Kumar")

st.write("[Source of the IoT project data link from kaggle>](https://www.kaggle.com/atulanandjha/temperature-readings-iot-devices/version/1?select=IOT-temp.csv)")
# Load Source datafile
DATA_URL= 'https://drive.google.com/file/d/1xccnxhugxzX_f07ww5ZhdDq1A9sFscu1/view?usp=sharing'
DATA_URL= 'https://drive.google.com/uc?id=' + DATA_URL.split('/')[-2]
data = pd.read_csv(DATA_URL)
st.subheader("Sample of Raw Data header display")
st.markdown('Temperature readings from IoT devices installed outside and inside of an anonymous room.')
st.write(data.head(5)) #displays the first five-row of dataset.
st.sidebar.header('Set HyperParameters For Random SearchCV') #to create header in sidebar
st.sidebar.write('---')
parameter_cross_validation=st.sidebar.slider('Number of Cross validation split', 2, 10)
split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 50, 90, 80, 5) 
st.sidebar.subheader('Learning Parameters')
parameter_n_estimators = st.sidebar.slider('Number of estimators for Regressor (n_estimators)', 0, 500, (10,50), 50)
parameter_n_estimators_step = st.sidebar.number_input('Step size for n_estimators', 10) 
# Using the slider, we can get the range for several estimator we can use in randomsearchcv and (10,50) is the default range. 
st.sidebar.write('---')
parameter_max_features =st.sidebar.multiselect('Max Features (You can select multiple options)',['auto', 'sqrt', 'log2'],['log2'])
parameter_max_depth =st.sidebar.multiselect('Max Depth (You can select multiple options)',[4,6,8,10,'none'],[6])
st.sidebar.write('---')
parameter_criterion = st.sidebar.selectbox('criterion',('gini', 'entropy'))
st.sidebar.subheader('Other Parameters')
parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
# function to convert month variable into seasons
def month2seasons(x):
    if x in [12, 1, 2]:
        season = 'Winter'
    elif x in [3, 4, 5]:
        season = 'Summer'
    elif x in [6, 7, 8, 9]:
        season = 'Monsoon'
    elif x in [10, 11]:
        season = 'Post_Monsoon'
    return season
#Function to convert hour to timing of day
def hours2timing(x):
    if x in [22,23,0,1,2,3]:
        timing = 'Night'
    elif x in range(4, 12):
        timing = 'Morning'
    elif x in range(12, 17):
        timing = 'Afternoon'
    elif x in range(17, 22):
        timing = 'Evening'
    else:
        timing = 'X'
    return timing
##  Function to calculate r2_score and RMSE on train and test data
def get_model_score(model, flag=True):
    '''
    model : classfier to predict values of X

    '''
    # defining an empty list to store train and test results
    score_list=[] 
    
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    
    train_r2=metrics.r2_score(Y_train,pred_train)
    test_r2=metrics.r2_score(Y_test,pred_test)
    train_rmse=np.sqrt(metrics.mean_squared_error(Y_train,pred_train))
    test_rmse=np.sqrt(metrics.mean_squared_error(Y_test,pred_test))
    
    #Adding all scores in the list
    score_list.extend((train_r2,test_r2,train_rmse,test_rmse))
    
    # If the flag is set to True then only the following print statements will be dispayed, the default value is True
    if flag==True: 
        st.write("R-sqaure on training set : ",metrics.r2_score(Y_train,pred_train))
        st.write("R-square on test set : ",metrics.r2_score(Y_test,pred_test))
        st.write("RMSE on training set : ",np.sqrt(metrics.mean_squared_error(Y_train,pred_train)))
        st.write("RMSE on test set : ",np.sqrt(metrics.mean_squared_error(Y_test,pred_test)))
    
    # returning the list with train and test scores
    return score_list
data.drop('room_id/id', axis=1, inplace=True)
# drop duplicate record
data = data.drop_duplicates()
data.rename(columns={'noted_date':'date', 'out/in':'place'}, inplace=True)
data['date'] = pd.to_datetime(data['date'],format = '%d-%m-%Y %H:%M')
data['year'] = data['date'].apply(lambda x : x.year)
data['month'] = data['date'].apply(lambda x :x.month)
data['day'] = data['date'].apply(lambda x :x.day)
data['weekday'] = data['date'].apply(lambda x : x.day_name())
data['weekofyear'] = data['date'].apply(lambda x :x.weekofyear)
data['hour'] = data['date'].apply(lambda x :x.hour)
data['minute'] = data['date'].apply(lambda x :x.minute)
data['season'] = data['month'].apply(month2seasons)
data['timing'] = data['hour'].apply(hours2timing)
data['id_number']=data['id'].apply(lambda x: int(x.split('_')[6]))
data1=data.copy(deep=True)
# We can drop column id as this is alphanumeric  and we have id_number as unique key that can be used for sorting
data.drop(['id'],axis=1,inplace=True)
data.drop(['id_number'],axis=1,inplace=True)
# Split the data to train and test
X = data.drop(['temp'],axis=1)
Y = data['temp']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size, random_state=7,stratify=Y)
X_train=pd.get_dummies(X_train,drop_first=True)
X_test=pd.get_dummies(X_test,drop_first=True)
X_train = X_train.apply(pd.to_numeric)
Y_train = Y_train.apply(pd.to_numeric)
X_test  = X_test.apply(pd.to_numeric)
Y_test = Y_test.apply(pd.to_numeric)
#Model Building
st.subheader("Perform EDA")
st.write("Summary of Temprature data from IoT devices")
alt_chart = alt.Chart(data1).mark_circle().encode(
     x='temp', y='season', size='weekday', color='weekday', tooltip=['temp', 'season', 'weekday'])
st.altair_chart(alt_chart, use_container_width=True)
if st.button('Distribution charts for analysis'):
        st.write("Distribution of temprature inside and outside of anonymus room for installed IoT devices" ) 
        fig1= sns.FacetGrid(data1,hue="place",size=10).map(sns.distplot,"temp").add_legend()
        st.pyplot(fig1)
        sns.set(rc={'figure.figsize':(16,10)})
        st.write("Value count detail of features" ) 
        month_count=sns.set({'figure.figsize': (10,8)})
        sns.countplot(data1['month']) # Plot for rating
        plt.title('value count for Month') #Lable the Title on graph
        st.pyplot(month_count)
        season_count=sns.set({'figure.figsize': (10,8)})
        sns.countplot(data1['season']) # Plot for rating
        plt.title('Value count for season') #Lable the Title on graph
        st.pyplot(season_count)
        timing_count=sns.set({'figure.figsize': (10,8)})
        sns.countplot(data1['timing']) # Plot for rating
        plt.title('Value count for timing of record') #Lable the Title on graph
        st.pyplot(timing_count)
        fig2=sns.pairplot(
        data,
        x_vars=["season", "timing", "weekofyear","weekday"],
        y_vars=["temp"],
        height=4,
        aspect=1
        );
        st.pyplot(fig2)
        
if st.button('Distribution Tables for Analysis'):
        st.write("Details of the dataset described as below")
        st.write(data.describe().T)   
        st.write("Average temprature recorded in every season with count of temp" ) 
        season_temp = data1[["season","temp"]]
        season_temp1=season_temp.copy(deep=True)
        season_temp_count=pd.DataFrame(season_temp1.groupby('season').temp.count())
        season_temp_count['Average season temp']=season_temp1.groupby('season').temp.mean()
        st.table(season_temp_count)
        st.write("Average temprature recorded during specific time daily with count of temp" ) 
        timing_temp = data1[["timing","temp"]]
        timing_temp1=timing_temp.copy(deep=True)
        timing_temp_count=pd.DataFrame(timing_temp1.groupby('timing').temp.count())
        timing_temp_count['Average temp']=timing_temp1.groupby('timing').temp.mean()
        st.table(timing_temp_count)
        st.write("Average temprature recorded per weekday with count of temp" ) 
        weekday_temp = data1[["weekday","temp"]]
        weekday_temp1=weekday_temp.copy(deep=True)
        weekday_temp_count=pd.DataFrame(weekday_temp1.groupby('weekday').temp.count())
        weekday_temp_count['Average temp on weekday']=weekday_temp1.groupby('weekday').temp.mean()
        st.table(weekday_temp_count)
        # Making a list of all catrgorical variables
        cat_col = [
            "place",
            "weekday",
            "season",
            "timing",
            ]
        
        # Printing number of count of each unique value in each column
        st.write("Below is the number of count of each unique value in each column" )
        for column in cat_col:
            st.table(data1[column].value_counts())
            st.write("-" * 40)
st.subheader("Model building")
if st.button('Click to Build Various Standard Models'):
    st.write("R2 score for various Models without hyperparameter tuning is as below:" )    
    models = []  # Empty list to store all the models
    
    # Appending pipelines for each model into the list
    models.append(
        (
            "Decission Tree",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("decision_tree", DecisionTreeRegressor(random_state=1)),
                ]
            ),
        )
    )
    models.append(
        (
            "Random Forest",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("random_forest", RandomForestRegressor(random_state=1)),
                ]
            ),
        )
    )
    models.append(
        (
            "Gradient Boost",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("gradient_boosting", GradientBoostingRegressor(random_state=1)),
                ]
            ),
        )
    )
    models.append(
        (
            "AdaBoost",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("adaboost", AdaBoostRegressor(random_state=1)),
                ]
            ),
        )
    )
    models.append(
        (
            "XG Boost",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("xgboost", XGBRegressor(random_state=1,eval_metric='logloss')),
                ]
            ),
        )
    )
    
    
    results = []  # Empty list to store all model's CV scores
    names = []  # Empty list to store name of the models
    
    # loop through all models to get the mean cross validated score
    for name, model in models:
        scoring = "r2"
        kfold = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=1
        )  # Setting number of splits equal to 5
        cv_result = cross_val_score(
            estimator=model, X=X_train, y=Y_train, scoring=scoring, cv=kfold
        )
        results.append(cv_result)
        names.append(name)
        st.write("{}: {}".format(name, cv_result.mean() * 100))
    # comparison of model performance 
     # Plotting boxplots for CV scores of all models defined above
    fig = plt.figure(figsize=(10, 7))
    
    fig.suptitle("Algorithm Comparison")
    ax = fig.add_subplot(111)
    
    plt.boxplot(results)
    ax.set_xticklabels(names)
    st.pyplot(fig)    
      
if st.button('Click to Build Random Forest Model with Hyperparameter'):
    n_estimators_range = np.arange(parameter_n_estimators[0] & parameter_n_estimators[1]+parameter_n_estimators_step, parameter_n_estimators_step)
    # Choose the type of classifier. 
    rf_tuned = RandomForestRegressor(random_state=parameter_random_state)
    
    # Grid of parameters to choose from
    parameters = {  
                    'max_depth': parameter_max_depth,
                    'max_features': parameter_max_features,
                    'n_estimators': n_estimators_range
    } 
    
    # Type of scoring used to compare parameter combinations
    scorer = metrics.make_scorer(metrics.r2_score)
    
    # Run the grid search
    RF_r_obj = RandomizedSearchCV(rf_tuned, parameters, scoring=scorer,cv=parameter_cross_validation)
    RF_r_obj = RF_r_obj.fit(X_train, Y_train)
    
    # Set the model to the best combination of parameters
    rf_tuned = RF_r_obj.best_estimator_
    
    # Fit the best algorithm to the data. 
    rf_tuned.fit(X_train, Y_train)
    st.write("model score in terms of r2 score and RMSE:")
    rf_tuned_score=get_model_score(rf_tuned)
    # So plot observed and predicted values of the test data
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pred=rf_tuned.predict(X_test)
    ax.scatter(Y_test, y_pred, edgecolors=(0, 0, 1))
    ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=3)
    ax.set_xlabel('Observed')
    ax.set_ylabel('Predicted')
    ax.set_title("Observed vs Predicted")
    plt.grid()
    st.pyplot(fig)
    # importance of features in the tree building ( The importance of a feature is computed as the 
    #(normalized) total reduction of the criterion brought by that feature. It is also known as the Gini importance )
    st.write("Importance of feature considered by model")
    feature_names = X_train.columns
    importances = rf_tuned[1].feature_importances_
    indices = np.argsort(importances)
    fig = plt.figure(figsize=(12, 12))
    plt.title("Feature Importances")
    plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Relative Importance")
    st.pyplot(fig)
    