from dash import Dash, dcc,html, Output, Input, State
import dash_bootstrap_components as dbc
import plotly.express as px
import scripts.data as data
from datetime import datetime, timedelta
import dash_mantine_components as dmc
import pickle
import pandas as pd
import plotly.graph_objects as go
from io import StringIO  


def graph(prev_data,steps,dataframe):


    fig = go.Figure()

    hours = 24*prev_data
    prediction_df = prediction(dataframe,12*steps)

    fig.add_trace(go.Scatter(x=dataframe.index[-hours:],y=dataframe["Actual Load"].iloc[-hours:],mode="lines",name="actual power load",line = dict(color="dodgerblue")))
    fig.add_trace(go.Scatter(x = prediction_df.index, y = prediction_df.values,name="predicted power load",line = dict(color="orange",dash="dash")))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    
    fig.update_layout(title_x = 0.5,xaxis_title = "date",yaxis_title="Total Load",plot_bgcolor= 'rgba(0, 0, 0, 0)',paper_bgcolor= 'rgba(0, 0, 0, 0)',xaxis_rangeslider_visible=True,title = f"Daily {12*steps}H Ahead Prediction")
    return fig, dataframe
def prediction(dataframe,steps):

    now = datetime.now()

    with open('models/xgboost_v1_no_temp.joblib', 'rb') as pickle_file:
        model1 = pickle.load(pickle_file)


    lags = dataframe
    if lags.index.tzinfo != None:
        lags.index.tz_convert(tz = "utc")
    lags.index = lags.index.tz_localize(None)
    lags = lags.asfreq("1H")

    start = now.strftime("%Y-%m-%d %H:00:00")
    end = now + timedelta(hours=steps)
    end = end.strftime("%Y-%m-%d %H:00:00")
    i = pd.date_range(start,end,freq = "1H")

    exo_df = pd.DataFrame(index = i)

    poly_cols =     ['sin_month_1', 
            'cos_month_1',
            'sin_week_of_year_1',
            'cos_week_of_year_1',
            'sin_week_day_1',
            'cos_week_day_1',
            'sin_hour_day_1',
            'cos_hour_day_1',
            'daylight_hours',
            'is_daylight']

    exog = data.create_features(exo_df,poly_cols)

    features = []
    # Columns that ends with _sin or _cos are selected
    features.extend(exog.filter(regex='^sin_|^cos_').columns.tolist())
    # columns that start with temp_ are selected
    features.extend(exog.filter(regex='^temp_.*').columns.tolist())
    # Columns that start with holiday_ are selected
    features.extend(['temp'])
    exog = exog.filter(features, axis=1)
    #removes temp features for testing
    features = [x for x in features if "temp" not in x]

    return model1.predict(steps = steps,
    exog = exog[features],
    last_window = lags["Actual Load"])


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

header = dcc.Markdown(children="# Forecasting Power Consumption",style={'textAlign':'center'})
#version = dcc.Markdown(children="#### ",style={'textAlign':'center'})
fig = px.line(title = "Daily 12H Ahead Prediction")
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
#'#f9f9f9'
fig.update_layout(title_x = 0.5,xaxis_title = "date",yaxis_title="K",plot_bgcolor= 'rgba(0, 0, 0, 0)',paper_bgcolor= 'rgba(0, 0, 0, 0)',xaxis_rangeslider_visible=True)

cards = html.Div()

app.layout = dmc.Grid(
    children = [
    dmc.Col([header],style={'margin-bottom':50,'padding-top':30},span = 12),
    #html.Br(),
    #dmc.Col([cards],span = 12),
    dmc.Col(
        dmc.Card(children = [
        
                html.H6("Daily High",style={"color":"grey"}),
                html.H3("0 kw",id = "high-value")

            ],
                shadow= "0px 1px 3px rgba(0,0,0,0.12), 0px 1px 2px rgba(0,0,0,0.24)",
                    radius="30",
                    withBorder=False,w= 150,
                    
                    ),span = 2,offset=4),
        
    dmc.Col(
            dmc.Card(children = [
                html.H6("Daily Low",style={"color":"grey"}),
                html.H3("0 kw",id = "low-value")

            ],
                shadow= "0px 1px 3px rgba(0,0,0,0.12), 0px 1px 2px rgba(0,0,0,0.24)",
                    radius="30",
                    withBorder=False,w= 150,
                    ),span = 2),
            
    dmc.Col(
        dmc.Card(children = [
            html.H6("Daily Average",style={"color":"grey"}),
            html.H3("0 kw",id = "avg-value")

        ],
            shadow= "0px 1px 3px rgba(0,0,0,0.12), 0px 1px 2px rgba(0,0,0,0.24)",
                radius="30",
                withBorder=False,w= 150),span = 2),
    
    
    

    dmc.Col(dmc.Paper(
        children = [
            dcc.Markdown('Filtering Options:',style={"font-size":'22px'}),

            html.Br(),
            dcc.Markdown('Length of Previous Data',style={"font-size":'16px'}),
            pred_length := dmc.Slider(
                id = "data-length",
                min = 1,max = 4,value = 1,
                step = 1,
                
                marks = [
                {"value": 1, "label" : "24H"},
                {"value": 2, "label" : "48H"},
                {"value": 3, "label" : "72H"},
                {"value": 4, "label" : "96H"}]

            ),
            html.Br(),
            dcc.Markdown('Prediction Length',style={"font-size":'16px'}),
            pred_length := dmc.Slider(
                id = "prediction-length",
                min = 1,max = 4,value = 1,
                step = 1,
                
                marks = [
                {"value": 1, "label" : "12H"},
                {"value": 2, "label" : "24H"},
                {"value": 3, "label" : "36H"},
                {"value": 4, "label" : "48H"}]

            ),
            html.Br(),
            dmc.Button("Confirm",variant="filled",id = "confirm-button",n_clicks=0)
        ],
        #xs, sm, md, lg, xl
        shadow = "0px 1px 3px rgba(0,0,0,0.12), 0px 1px 2px rgba(0,0,0,0.24)",
        withBorder=False,
        radius = "30",
        style={"width":"90%",
                "margin-left":10,
                "height":300,
                "backgroundColor":"#f9f9f9",
                "padding":"1rem"
        },
        
        
    
    
    
    ),span = 3),
            #dbc.Col(dbc.Card(gr := dcc.Graph(figure=fig)),width=9
                    
            
            #)
    dmc.Col(dmc.Paper(
            children=[gr := dcc.Graph(figure=fig)],
            shadow= "0px 1px 3px rgba(0,0,0,0.12), 0px 1px 2px rgba(0,0,0,0.24)",
            radius="30",
            withBorder=False,
        ),span =7,
        style={"margin-right":10})
        ,
        #dbc.Col(dbc.Card(gf := dcc.Graph(figure=fig))),
    dcc.Interval(
        id='interval-component',
        interval=60*1000000, # in milliseconds
        #interval = 60000,
        n_intervals=0
    ),
    dcc.Store(
        id='session', storage_type='session'
    )
    
          
],style={'backgroundColor':'#f2f2f2','height':"100vh"})

@app.callback(
    Output(component_id=gr,component_property = 'figure'),
    Output("high-value", "children"),
    Output("low-value", "children"),
    Output("avg-value", "children"),
    Output(component_id="session",component_property="data"),
    Input(component_id='interval-component',component_property ='n_intervals') ,
    State(component_id="session",component_property="data"),

)

def load_graph(n,storedata):
    
    now = datetime.now()    
    prevhour = now - timedelta(hours=169)
    starttime = prevhour.strftime("%Y-%m-%d %H:00:00")
    endtime = now.strftime("%Y-%m-%d %H:00:00")
    df = data.energy_api(starttime,endtime=endtime)
    print(now)
    print(now.strftime("%M"))

    fig,df = graph(1,1,df)
    
    high = df["Actual Load"].iloc[-24:].max()
    low = df["Actual Load"].iloc[-24:].min()
    avg = df["Actual Load"].iloc[-24:].mean()

    storedata = df.to_json(orient="split")
    

    return fig, f"{high} kw",f"{low} kw",f"{avg} kw",storedata
   

@app.callback(
    Output(component_id=gr,component_property = 'figure',allow_duplicate=True),
    Input(component_id="confirm-button",component_property="n_clicks"),
    State(component_id="prediction-length",component_property="value"),
    State(component_id="data-length",component_property="value"),
    State(component_id="session",component_property="data"),
    prevent_initial_call = True
)
def update_graph(btn,pred_length,prev_length,storedata):

    df = pd.read_json(StringIO(storedata),orient="split")

    fig, _ = graph(prev_length,pred_length,df)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True,port=8050)