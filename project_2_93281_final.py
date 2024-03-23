# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:49:07 2024

@author: josoa
"""
#First import the main libraries
import webbrowser

import dash
from dash import Dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
from sklearn import  metrics
import numpy as np
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

import os
from dash.exceptions import PreventUpdate

###Load data of 2019###

## Raw Data ##
df_raw_2019 = pd.read_csv("testData_2019_Civil.csv")
df_raw_2019['Date']=pd.to_datetime(df_raw_2019['Date'])
df_raw_2019=df_raw_2019.set_index(['Date'], drop=True)
df_raw_2019.rename(columns = {'Civil (kWh)': 'Power (kW)','solarRad_W/m2': 'Solar Radiation (W/m2)', 'windSpeed_m/s': 'Wind Speed (m/s)', 'windGust_m/s': 'Wind Gust (m/s)', 'rain_mm/h': 'Rain (mm/h)','rain_day': 'Rain Day', 'temp_C': 'Temperature (C)', 'HR': 'Relative Humidity (%)', 'pres_mbar': 'Pressure (mbar)' }, inplace = True)
print(df_raw_2019)

#Clean Data with additional features#
df_2019 = df_raw_2019.copy()
df_2019.rename(columns = {'Civil (kWh)': 'Power (kW)','solarRad_W/m2': 'Solar Radiation (W/m2)', 'windSpeed_m/s': 'Wind Speed (m/s)', 'windGust_m/s': 'Wind Gust (m/s)', 'rain_mm/h': 'Rain (mm/h)','rain_day': 'Rain Day', 'temp_C': 'Temperature (C)', 'HR': 'Relative Humidity (%)', 'pres_mbar': 'Pressure (mbar)' }, inplace = True)
df_2019.loc[:, 'Power-1'] = df_2019['Power (kW)'].shift(1) # Previous hour consumption
df_2019.loc[:, 'Power-2'] = df_2019['Power-1'].shift(1) # Second previous hour consumption
df_2019=df_2019.dropna() #removes first two lines
df_2019['Month'] = df_2019.index.month
df_2019['Hour'] = df_2019.index.hour
df_2019['Week Day'] = df_2019.index.weekday + 1
df_2019['Weekend'] = df_2019.index.weekday.isin([5, 6]).astype(int)
df_2019=df_2019.iloc[:, [0,9,10,1,2,3,4,5,6,7,8,11,12,13,14]] #arranges final order
print(df_2019)

#Dataframes used for regression models#
df_2019_fs = df_2019.copy()
df_2019_fs = df_2019_fs.drop(['Wind Speed (m/s)', 'Wind Gust (m/s)', 'Week Day', 'Rain (mm/h)', 'Rain Day'], axis=1)  #removes features not used in project 1 for the regression models
print(df_2019_fs)
df_2019_fs2 = df_2019.copy()
df_2019_fs2 = df_2019_fs2.drop(['Wind Speed (m/s)', 'Wind Gust (m/s)', 'Rain (mm/h)','Rain Day', 'Temperature (C)', 'Relative Humidity (%)', 'Pressure (mbar)', 'Month', 'Weekend', 'Week Day'], axis=1) #removes additional features not used in the random forest - Less Features Model
print(df_2019_fs2)

fig_raw=px.line(df_raw_2019, x=df_raw_2019.index, y=df_raw_2019.columns[1:9])

## Regression Models ##

#define vectors#
Z=df_2019_fs.values
Z2=df_2019_fs2.values
y2=Z[:,0] #real power data of 2019
X2=Z[:,1:] #features for regression models
X3=Z2[:,1:] #features for RF - Less Features model

#import .pkl saved models#
with open('RF_model.pkl','rb') as file:
    RF_model=pickle.load(file)
    y2_pred_RF = RF_model.predict(X2)
    
MAE_RF=metrics.mean_absolute_error(y2,y2_pred_RF)
MSE_RF=metrics.mean_squared_error(y2,y2_pred_RF)
RMSE_RF=np.sqrt(metrics.mean_squared_error(y2,y2_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y2)

with open('BT_model.pkl','rb') as file:
    BT_model=pickle.load(file)
    y2_pred_BT = BT_model.predict(X2)
    
MAE_BT=metrics.mean_absolute_error(y2,y2_pred_BT)
MSE_BT=metrics.mean_squared_error(y2,y2_pred_BT)
RMSE_BT=np.sqrt(metrics.mean_squared_error(y2,y2_pred_BT))
cvRMSE_BT=RMSE_BT/np.mean(y2)

with open('RF_model2.pkl','rb') as file:
    RF_model2=pickle.load(file)
    y2_pred_RF2 = RF_model2.predict(X3)
    
MAE_RF2=metrics.mean_absolute_error(y2,y2_pred_RF2)
MSE_RF2=metrics.mean_squared_error(y2,y2_pred_RF2)
RMSE_RF2=np.sqrt(metrics.mean_squared_error(y2,y2_pred_RF2))
cvRMSE_RF2=RMSE_RF2/np.mean(y2)

with open('GB_model.pkl','rb') as file:
    GB_model=pickle.load(file)
    y2_pred_GB = GB_model.predict(X2)
    
MAE_GB=metrics.mean_absolute_error(y2,y2_pred_GB)
MSE_GB=metrics.mean_squared_error(y2,y2_pred_GB)
RMSE_GB=np.sqrt(metrics.mean_squared_error(y2,y2_pred_GB))
cvRMSE_GB=RMSE_GB/np.mean(y2)

with open('DT_regr_model.pkl','rb') as file:
    DT_regr_model=pickle.load(file)
    y2_pred_DT = DT_regr_model.predict(X2)
    
MAE_DT=metrics.mean_absolute_error(y2,y2_pred_DT)
MSE_DT=metrics.mean_squared_error(y2,y2_pred_DT)
RMSE_DT=np.sqrt(metrics.mean_squared_error(y2,y2_pred_DT))
cvRMSE_DT=RMSE_DT/np.mean(y2)

with open('regr.pkl','rb') as file:
    LR_model=pickle.load(file)
    y2_pred_LR = LR_model.predict(X2)
    
MAE_LR=metrics.mean_absolute_error(y2,y2_pred_LR)
MSE_LR=metrics.mean_squared_error(y2,y2_pred_LR)
RMSE_LR=np.sqrt(metrics.mean_squared_error(y2,y2_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y2)

#Dataframes with predictions and errors#
d_models_metrics = {'Methods':['Linear Regression', 'Random Forest', 'RF - Less Features', 'Bootstrapping', 'Gradient Boosting', 'Decision Tree'],'MAE':[MAE_LR, MAE_RF, MAE_RF2, MAE_BT, MAE_GB, MAE_DT], 'MSE':[MSE_LR, MSE_RF, MSE_RF2, MSE_BT, MSE_GB, MSE_DT], 'RMSE':[RMSE_LR, RMSE_RF,RMSE_RF2, RMSE_BT, RMSE_GB, RMSE_DT], 'cvRMSE':[cvRMSE_LR, cvRMSE_RF,cvRMSE_RF2, cvRMSE_BT, cvRMSE_GB, cvRMSE_DT]}
df_metrics = pd.DataFrame(data=d_models_metrics)
df_metrics=df_metrics.set_index(['Methods'], drop=True)
d_models_results = {'Date':df_2019.index,'Linear Regression':y2_pred_LR, 'Random Forest':y2_pred_RF, 'RF - Less Features':y2_pred_RF2, 'Bootstrapping':y2_pred_BT, 'Gradient Boosting': y2_pred_GB, 'Decision Tree': y2_pred_DT}
df_forecast=pd.DataFrame(data=d_models_results)
df_forecast['Date']=pd.to_datetime(df_forecast['Date'])
df_forecast=df_forecast.set_index(['Date'], drop=True)

print(df_metrics)
print(df_forecast)
df_results = pd.merge(df_2019,df_forecast, on='Date')
df_results = df_results.iloc[:,[0,15,16,17,18,19,20]]
print(df_results)

fig_results = px.line(df_results, x=df_results.index, y=df_results.columns[0:6])


### Dashboard ###

#" Create Dash app ##
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
white_text_style = {'color': 'white'}

# Define auxiliary functions
def generate_table(dataframe, max_rows=10, max_height=None):
    # Apply some CSS styles to the table
    table_style = {
        'borderCollapse': 'collapse',
        'borderSpacing': '0',
        'width': '100%',
        'border': '1px solid #ddd',
        'fontFamily': 'Arial, sans-serif',
        'fontSize': '14px'
    }
    
    th_style = {
        'border': '1px solid #ddd',
        'padding': '8px',
        'textAlign': 'left',
        'backgroundColor': '#f2f2f2',
        'fontWeight': 'bold',
        'color': '#333'
    }
    
    td_style = {
        'border': '1px solid #ddd',
        'padding': '8px',
        'textAlign': 'left'
    }
    
    # Define a style for the table container to enable scrolling
    table_container_style = {
        'overflowY': 'scroll',
        'maxHeight': f'{max_height}px' if max_height else 'none'
    }
    
    return html.Div(
        style=table_container_style,
        children=[
            html.Table(
                # Apply the table style
                style=table_style,
                children=[
                    # Add the table header
                    html.Thead(
                        html.Tr([
                            html.Th('Index', style=th_style),
                            *[html.Th(col, style=th_style) for col in dataframe.columns]
                        ])
                    ),
                    # Add the table body
                    html.Tbody([
                        html.Tr([
                            html.Td(dataframe.index[i], style=td_style),
                            *[html.Td(dataframe.iloc[i][col], style=td_style) for col in dataframe.columns]
                        ])
                        for i in range(min(len(dataframe), max_rows))
                    ])
                ]
            )
        ]
    )





def generate_graph(df, columns, start_date, end_date):
    filtered_df = df.loc[start_date:end_date, columns]
    
    # Define a list to hold the y-axis configurations
    y_axis_config = []
    
    # Loop through each column and define a new y-axis configuration
    for i, column in enumerate(columns):
        y_axis_config.append({'title': column, 'overlaying': 'y', 'side': 'right', 'position': i * 0.1})
    
    # Define the data and layout of the figure
    data = [go.Scatter(x=filtered_df.index, y=filtered_df[column], name=column) for column in filtered_df.columns]
    layout = go.Layout(title=', '.join(columns), xaxis_title='Date')
    
    # Update the layout to include the y-axis configurations
    layout.update({'yaxis{}'.format(i + 1): y_axis_config[i] for i in range(len(y_axis_config))})
    
    # Create the figure with the data and layout
    fig = go.Figure(data=data, layout=layout)
    
    return fig

def generate_stats_table(dataframe):
    stats_table = dataframe.describe().reset_index().rename(columns={'index': 'Statistic'})
    stats_table_html = html.Table(
        # Header
        [html.Tr([html.Th(col) for col in stats_table.columns])] +
        # Body
        [html.Tr([html.Td(stats_table.iloc[i][col]) for col in stats_table.columns]) for i in range(len(stats_table))]
    )
    return stats_table_html


## App layout ##

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(style={'backgroundColor': 'white'},children = [
    html.H1(html.B('IST Civil Pavilion Energy Analysis')),
    html.Div(id='df_raw_2019', children=df_raw_2019.to_json(orient='split'), style={'display': 'none'}),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='2019 Raw Data', children=[
            html.Div([
                html.H2("Registered parameters"),
                dcc.Dropdown(
                    id='column-dropdown',
                    options=[{'label': i, 'value': i} for i in df_raw_2019.columns],
                    value=[df_raw_2019.columns[0]],
                    multi=True
                ),
                dcc.DatePickerRange(
                    id='date-picker',
                    min_date_allowed=df_raw_2019.index.min(),
                    max_date_allowed=df_raw_2019.index.max(),
                    start_date=df_raw_2019.index.min(),
                    end_date=df_raw_2019.index.max()
                ),
                dcc.Graph(id='graph'),
            ])
        ]),
        
        dcc.Tab(label='Exploratory Data Analysis', value='tab-2', children=[
            html.Div([
                html.H2("EDA of 2019 Raw Data"),
                dcc.Dropdown(
                    id='feature1',
                    options=[{'label': col, 'value': col} for col in df_raw_2019.columns],
                    value=df_raw_2019.columns[1]
                    ),
                dcc.Dropdown(
                    id='feature2',
                    options=[{'label': col, 'value': col} for col in df_raw_2019.columns],
                    value=df_raw_2019.columns[0]
                    ),
                dcc.Graph(id='scatter-plot'),
                dcc.Dropdown(
                    id='feature-boxplot',
                    options=[{'label': col, 'value': col} for col in df_raw_2019.columns],
                    value=df_raw_2019.columns[0]
                    ),
                dcc.Graph(id='box-plot'),
                html.H2("Statistics"),
                html.Div(id='stats-table-div')
                ])
            ], style={'backgroundColor': 'blue', 'color': 'white'}),
        
        dcc.Tab(label='Selected Features', value='tab-3', children=[
    html.Div([
        html.H2("Select the DataFrame for visualization"),
        dcc.Dropdown(
            id='dataframe-dropdown',
            options=[
                {'label': 'All Features Data', 'value': 'df_2019_fs'},
                {'label': 'Less Features Data', 'value': 'df_2019_fs2'}
            ],
            value='df_2019_fs'
        ),
        html.Div(id='dataframe-table'),  # This div will contain the table
        dcc.Dropdown(
            id='feature-dropdown',
            options=[{'label': col, 'value': col} for col in df_2019_fs.columns],
            value=[df_2019_fs.columns[0]],
            multi=True
        ),
        dcc.DatePickerRange(
            id='date-selector',
            min_date_allowed=df_2019_fs.index.min(),
            max_date_allowed=df_2019_fs.index.max(),
            start_date=df_2019_fs.index.min(),
            end_date=df_2019_fs.index.max()
        ),
        dcc.Graph(id='selected-features-graph')
    ])
], style={'backgroundColor': 'Green', 'color': 'white'}),
        
        dcc.Tab(label='Forecast', value='tab-4', children=[
    html.Div([
        html.H2("Forecast Results"),
        dcc.Dropdown(
            id='model-selector',
            options=[
                {'label': 'Linear Regression', 'value': 'Linear Regression'},
                {'label': 'Random Forest', 'value': 'RF_model'},
                {'label': 'RF - Less Features', 'value': 'RF_model2'},
                {'label': 'Bootstrapping', 'value': 'Bootstrapping'},
                {'label': 'Gradient Boosting', 'value': 'Gradient Boosting'},
                {'label': 'Decision Tree', 'value': 'Decision Tree'}
            ],
            value=['RF_model'],  # Default model selected
            multi=True  # Allow multiple selections
        ),
        dcc.DatePickerRange(
            id='forecast-date-selector',
            min_date_allowed=df_results.index.min(),
            max_date_allowed=df_results.index.max(),
            start_date=df_results.index.min(),
            end_date=df_results.index.max()
        ),
        dcc.Graph(id='forecast-graph'),
        html.Div([
            html.H3("Regression Models Metrics"),
            html.Div(id='metrics-table-div', children=[
                generate_table(df_metrics)
            ])
        ]),
        dcc.DatePickerRange(
            id='difference-date-selector',
            min_date_allowed=df_results.index.min(),
            max_date_allowed=df_results.index.max(),
            start_date=df_results.index.min(),
            end_date=df_results.index.max()
        ),
        dcc.Graph(id='difference-graph'),  # Add a new graph for showing differences    
    ]),
], style={'backgroundColor': 'Orange', 'color': 'white'}),
        
])
])

## Define the callbacks ##

# callback for tab '2019 Raw Data'#

@app.callback(Output('graph', 'figure'),
              Input('column-dropdown', 'value'),
              Input('date-picker', 'start_date'),
              Input('date-picker', 'end_date')
)
def update_figure(columns, start_date, end_date):
    
    filtered_df = df_raw_2019.loc[start_date:end_date, columns]
    
    # Define a list to hold the y-axis configurations
    y_axis_config = []
    
    # Loop through each column and define a new y-axis configuration
    for i, column in enumerate(columns):
        y_axis_config.append({'overlaying': 'y', 'side': 'right', 'position': 1 - i * 0.1})
    
    # Define the data and layout of the figure
    data = [{'x': filtered_df.index, 'y': filtered_df[column], 'type': 'line', 'name': column} for column in filtered_df.columns]
    layout = {'title': {'text': ', '.join(columns)}, 'xaxis': {'title': 'Date'}}
    
    # Update the layout to include the y-axis configurations
    layout.update({'yaxis{}'.format(i + 1): y_axis_config[i] for i in range(len(y_axis_config))})
    
    # Create the figure with the data and layout
    fig = {'data': data, 'layout': layout}
    
    return fig

# callback for tab 'Exploratory Data Analysis'#

@app.callback(Output('scatter-plot', 'figure'),
              Input('feature1', 'value'),
              Input('feature2', 'value'))


def update_scatter_plot(feature1, feature2):
    fig = {
        'data': [{
            'x': df_raw_2019[feature1],
            'y': df_raw_2019[feature2],
            'mode': 'markers'
        }],
        'layout': {
            'title': f'{feature1} vs {feature2}',
            'xaxis': {'title': feature1},
            'yaxis': {'title': feature2},
        }
    }
    return fig

@app.callback(
    Output('box-plot', 'figure'),
    Input('feature-boxplot', 'value')
)
def update_box_plot(feature_boxplot):
    fig = go.Figure()
    fig.add_trace(go.Box(y=df_raw_2019[feature_boxplot], name=feature_boxplot))
    fig.update_layout(title=f"Box Plot for {feature_boxplot}")
    return fig

@app.callback(
    Output('stats-table-div', 'children'),
    [Input('tabs', 'value')]
)
def update_stats_table(tab):
    if tab == 'tab-2':
        return generate_stats_table(df_raw_2019)
    else:
        return None

# callback for tab 'Selected Features'#

@app.callback(
    Output('dataframe-table', 'children'),
    Input('dataframe-dropdown', 'value')
)
def display_selected_dataframe_table(selected_dataframe):
    df_selected = globals()[selected_dataframe]
    return generate_table(df_selected)



@app.callback(
    Output('feature-dropdown', 'options'),
    Output('feature-dropdown', 'value'),  # resets the selected features (dataframe)
    Input('dataframe-dropdown', 'value')
)
def update_feature_dropdown_options(selected_dataframe):
    df_selected = globals()[selected_dataframe]
    options = [{'label': col, 'value': col} for col in df_selected.columns]
    return options, []  # Reset the selected features to an empty list



@app.callback(
    Output('selected-features-graph', 'figure'),
    [Input('dataframe-dropdown', 'value'),
     Input('feature-dropdown', 'value'),
     Input('date-selector', 'start_date'),
     Input('date-selector', 'end_date')]
)
def update_selected_features_graph(selected_dataframe, selected_features, start_date, end_date):
    df_selected = globals()[selected_dataframe]
    filtered_df = df_selected.loc[start_date:end_date, selected_features]

    # Create the line plot using Plotly Express
    fig = px.line(filtered_df, x=filtered_df.index, y=selected_features)

    return fig

# callback for tab 'Forecast'#

@app.callback(
    Output('forecast-graph', 'figure'),
    [Input('model-selector', 'value'),
     Input('forecast-date-selector', 'start_date'),
     Input('forecast-date-selector', 'end_date')]
)
def update_forecast(selected_models, start_date, end_date):
    selected_columns = ['Power (kW)']  # Always include 'Power (kW)'
    
    # Add selected models to the list of columns to display
    if selected_models:
        for model_name in selected_models:
            if model_name == 'RF_model':
                selected_columns.append('Random Forest')
            elif model_name == 'RF_model2':
                selected_columns.append('RF - Less Features')
            elif model_name == 'Bootstrapping':
                selected_columns.append('Bootstrapping')
            elif model_name == 'Gradient Boosting':
                selected_columns.append('Gradient Boosting')
            elif model_name == 'Decision Tree':
                selected_columns.append('Decision Tree')
            elif model_name == 'Linear Regression':
                selected_columns.append('Linear Regression')

    # Filter data based on selected date range
    df_filtered = df_results.loc[start_date:end_date]

    # Reshape the data for plotting
    df_plot = df_filtered.reset_index().melt(id_vars='Date', var_name='Model', value_name='Power')
    df_plot = df_plot[df_plot['Model'].isin(selected_columns)]

    # Generate the line plot using Plotly Express
    fig = px.line(df_plot, x='Date', y='Power', color='Model')

    return fig


@app.callback(
    Output('difference-graph', 'figure'),
    [Input('model-selector', 'value'),
     Input('difference-date-selector', 'start_date'),
     Input('difference-date-selector', 'end_date')]
)
def update_difference_graph(selected_models, start_date, end_date):
    data = []

    if selected_models:
        for model_name in selected_models:
            if model_name == 'RF_model':
                predicted_values = df_results['Random Forest']
            elif model_name == 'RF_model2':
                predicted_values = df_results['RF - Less Features']
            elif model_name == 'Bootstrapping':
                predicted_values = df_results['Bootstrapping']
            elif model_name == 'Gradient Boosting':
                predicted_values = df_results['Gradient Boosting']
            elif model_name == 'Decision Tree':
                predicted_values = df_results['Decision Tree']
            elif model_name == 'Linear Regression':
                predicted_values = df_results['Linear Regression']

            # Filter data based on selected date range
            filtered_predicted_values = predicted_values.loc[start_date:end_date]

            # Calculate the difference between the predicted values and actual power values
            difference = filtered_predicted_values - df_results.loc[start_date:end_date, 'Power (kW)']

            # Add a trace for the difference
            trace = go.Scatter(x=filtered_predicted_values.index, y=difference, mode='lines', name=f'{model_name} - Difference')
            data.append(trace)

    # Define the layout for the difference graph
    layout = go.Layout(title='Difference between Predicted and Actual Power', xaxis_title='Date', yaxis_title='Difference')

    # Create the figure with the data and layout
    fig = go.Figure(data=data, layout=layout)

    return fig  
 
if __name__ == '__main__':
    #webbrowser.open('http://127.0.0.1:8050/') #uncomment this line in the first time running the code, then it can be commented again
    app.run_server()