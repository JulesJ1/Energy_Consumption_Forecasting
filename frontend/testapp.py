from dash import html, Dash, dcc, Input, Output, callback, State
import dash_mantine_components as dmc
import pandas as pd
import plotly.express as px

import plotly.graph_objects as go
from datetime import datetime
import dash_bootstrap_components as dbc
import scripts.data as data
from datetime import datetime, timedelta
import pickle

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


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

def graph(prev_data,steps,df):
    
    fig = go.Figure()


    hours = 24*prev_data
    prediction_df = prediction(df,12*steps)

    fig.add_trace(go.Scatter(x=df.index[-hours:],y=df["Actual Load"].iloc[-hours:],mode="lines",name="actual power load",line = dict(color="dodgerblue")))
    fig.add_trace(go.Scatter(x = prediction_df.index, y = prediction_df.values,name="predicted power load",line = dict(color="orange",dash="dash")))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    
    fig.update_layout(title_x = 0.5,xaxis_title = "date",yaxis_title="Total Load",plot_bgcolor= 'rgba(0, 0, 0, 0)',paper_bgcolor= 'rgba(0, 0, 0, 0)',xaxis_rangeslider_visible=True,title = f"Daily {12*steps}H Ahead Prediction")
    return fig, df

header = dcc.Markdown(children="# Forecasting Power Consumption",style={'textAlign':'center'})
fig = px.line(title = "Daily 12H Ahead Prediction")
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
#'#f9f9f9'
fig.update_layout(title_x = 0.5,xaxis_title = "date",yaxis_title="K",plot_bgcolor= 'rgba(0, 0, 0, 0)',paper_bgcolor= 'rgba(0, 0, 0, 0)',xaxis_rangeslider_visible=True)

app.layout = dmc.MantineProvider(

    #dmc.Container( children = [
    dmc.Grid(
        children = [
            dmc.Col(dcc.Markdown(children="# Forecasting Power Consumption",style={'textAlign':'center',"margin-bottom":100,'padding-top':30}),span = 12),
            dmc.Space(h="xl"),
            
            dmc.Col(children = [
            dmc.SimpleGrid(cols = 3,children = [
                dmc.Card(children = [
                
                        html.H6("Daily High",style={"color":"grey"}),
                        html.H3("1234 kw",id = "high-value",style = {"color":"green"})

                    ],
                        shadow= "0px 1px 3px rgba(0,0,0,0.12), 0px 1px 2px rgba(0,0,0,0.24)",
                            radius="30",
                            
                           
                            
                            ),
                
            
                    dmc.Card(children = [
                        html.H6("Daily Low",style={"color":"grey"}),
                        html.H3("234 kw",style = {"color":"red"})

                    ],
                        shadow= "0px 1px 3px rgba(0,0,0,0.12), 0px 1px 2px rgba(0,0,0,0.24)",
                            radius="30",
                          
                            ),
                    
            
                dmc.Card(children = [
                    html.H6("Daily Average",style={"color":"grey"}),
                    html.H3("5343 kw",style = {"color":"blue"})

                ],
                    shadow= "0px 1px 3px rgba(0,0,0,0.12), 0px 1px 2px rgba(0,0,0,0.24)",
                        radius="30"
                      
                        )
            
        ])],span = 6,offset = 3),

        
        #,style={"margin-left":"20rem"}
            dmc.Col(
                dmc.Paper(
                    children = [
                        dmc.Button("Confirm",variant="filled",id = "confirm-button",n_clicks=0),
                        dcc.Markdown('Select Prediction Length',style={"font-size":'16px'}),
                        html.Br(),
                        pred_length := dmc.Slider(1,4,
                            
                            marks = [{
                            1: "12H",
                            2: "24H",
                            3: "36H",
                            4: "48H"}]
                        )
                    ],
                    #xs, sm, md, lg, xl
                    shadow = "0px 1px 3px rgba(0,0,0,0.12), 0px 1px 2px rgba(0,0,0,0.24)",
                    withBorder=False,
                    radius = "30",
                    style={"width":"90%",
                            "margin-left":"1.5rem",
                            "height":350,
                            #"backgroundColor":"#f9f9f9",
                            "padding":"1rem"
                    },
                
                ),span = 3),

            dmc.Col(
                dmc.Paper(
                            children=[gr := dcc.Graph(figure=fig)],
                            shadow= "0px 1px 3px rgba(0,0,0,0.12), 0px 1px 2px rgba(0,0,0,0.24)",
                            radius="30",
                            withBorder=False,
                            ),
                #style={"margin-left":30},
                span = 7)
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
            
            

    ],gutter = "xs")
)
#,'height':"100vh"
@callback(
    Output('high-value', 'children'),
    Input(component_id='interval-component',component_property ='n_intervals')
)
def card(n):
    return "453"


@app.callback(
    Output(component_id=gr,component_property = 'figure'),
    Output(component_id="session",component_property="data"),
    
    Input(component_id='interval-component',component_property ='n_intervals'),
    State(component_id="session",component_property="data"), 

)

def load_graph(n,storedata):

    now = datetime.now()   
    prevhour = now - timedelta(hours=169)
    starttime = prevhour.strftime("%Y-%m-%d %H:00:00")
    endtime = now.strftime("%Y-%m-%d %H:00:00")
    
    df = data.energy_api(starttime,endtime=endtime)

    fig,df = graph(1,1,df)
    print(df.head)
    storedata = df.to_json(orient="split")

    return fig,storedata


@app.callback(
    Output(component_id=gr,component_property = 'figure',allow_duplicate=True),
    Input(component_id="confirm-button",component_property="n_clicks"),
    State(component_id="session",component_property="data"),
    prevent_initial_call = True
)
def update_graph(btn,storedata):
    #print(storedata)
    #df = pd.json_normalize(storedata)
    df = pd.read_json(storedata,orient="split")
    
    print(df.head)

    fig, _ = graph(2,2,df)
    return fig

if __name__ == '__main__':
    app.run_server(
  
        debug=True,
        dev_tools_props_check=True
    )
