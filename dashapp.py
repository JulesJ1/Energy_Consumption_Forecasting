from dash import Dash, dcc,html, Output, Input
import dash_bootstrap_components as dbc
import plotly.express as px
import scripts.data as data
from datetime import datetime, timedelta
import dash_mantine_components as dmc



#dbc.themes.SLATE
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

header = dcc.Markdown(children="# Forecasting Power Consumption",style={'textAlign':'center'})
fig = px.line(title = "Daily 12H Ahead Prediction")
fig.update_layout(title_x = 0.5,xaxis_title = "date",yaxis_title="K",plot_bgcolor= 'rgba(0, 0, 0, 0)',paper_bgcolor= '#f9f9f9',xaxis_rangeslider_visible=True)

cards = html.Div()

app.layout = dbc.Container([
    dbc.Row([header],style={'margin-bottom':100,'padding-top':30}),
    html.Br(),
    dbc.Row([
            dbc.Col(dmc.Paper(
                children = [
                    dcc.Markdown('Select data frequency',style={"font-size":'16px'}),
                    freq := dcc.Dropdown(['hourly','daily'],'daily',id = 'freq-dropdown'),
                    html.Br(),
                    dcc.Markdown('Select Model',style={"font-size":'16px'}),
                    model := dcc.RadioItems(['XGBoost','LTSM'],'XGBoost')
                ],
                #xs, sm, md, lg, xl
                shadow = "0px 1px 3px rgba(0,0,0,0.12), 0px 1px 2px rgba(0,0,0,0.24)",
                withBorder=False,
                radius = "30",
                style={"width":"90%",
                        "margin-left":10,
                        "height":450,
                        "backgroundColor":"#f9f9f9",
                        "padding":"1rem"
                },
                
                
            
            
            
            )),
            #dbc.Col(dbc.Card(gr := dcc.Graph(figure=fig)),width=9
                    
            
            #)
            dbc.Col(dmc.Paper(
                    children=[gr := dcc.Graph(figure=fig)],
                    shadow= "0px 1px 3px rgba(0,0,0,0.12), 0px 1px 2px rgba(0,0,0,0.24)",
                    radius="30",
                    withBorder=False,
            ),width=7,
            style={"margin-right":10})
            ,
            #dbc.Col(dbc.Card(gf := dcc.Graph(figure=fig))),
            dcc.Interval(
            id='interval-component',
            interval=60*1000, # in milliseconds
            n_intervals=0
        ),
        
        dbc.Col([cards]),
    
    
        
    ])
],fluid = True,style={'backgroundColor':'#f2f2f2','height':"100vh"})
"""
@app.callback(
    Output(component_id=gr,component_property = 'figure'),
    Input(component_id='interval-component',component_property ='n_intervals') 

)

def update_graph(n):
    now = datetime.now()
    prevhour = now - timedelta(hours=24)
    starttime = prevhour.strftime("%Y-%m-%d %H:00:00")
    endtime = now.strftime("%Y-%m-%d %H:00:00")
    df = data.energy_api(starttime,endtime=endtime)

    fig = px.line(x=df.index,y=df["Actual Load"]).update_layout(
                                template='plotly_dark',
                                plot_bgcolor='rgba(0, 0, 0, 0)',
                                paper_bgcolor='#f9f9f9',
                                title_x = 0.5,
                                xaxis_rangeslider_visible=True,
                                showgrid=False

                            )
    return fig
"""
@app.callback(
    Output(cards, "children"),
    Input(component_id='interval-component', component_property ='n_intervals')

)
def update_cards(n):
    now = datetime.now()
    endtime = now.strftime("%Y-%m-%d %H:00:00")
    prevhour = now - timedelta(hours=24)
    starttime = prevhour.strftime("%Y-%m-%d %H:00:00")
    df = data.energy_api(starttime,endtime=endtime)

    high = df["Actual Load"].max()
    low = df["Actual Load"].min()
    avg = df["Actual Load"].mean()

    card_layout = [
            dmc.Paper(children = [
                html.H4("Daily High",style={'textAlign':'center'}),
                html.H4(f"{high} kw",style={'textAlign':'center'})
                

            ],
                shadow= "0px 1px 3px rgba(0,0,0,0.12), 0px 1px 2px rgba(0,0,0,0.24)",
                    radius="30",
                    withBorder=False),
            html.Br(),
            dmc.Paper(children = [
                html.H4("Daily Low",style={'textAlign':'center'}),
                html.H4(f"{low} kw",style={'textAlign':'center'})

            ],
                shadow= "0px 1px 3px rgba(0,0,0,0.12), 0px 1px 2px rgba(0,0,0,0.24)",
                    radius="30",
                    withBorder=False),
            html.Br(),
            dmc.Paper(children = [
                html.H4("Daily Average",style={'textAlign':'center'}),
                html.H4(f"{avg} kw",style={'textAlign':'center'})

            ],
                shadow= "0px 1px 3px rgba(0,0,0,0.12), 0px 1px 2px rgba(0,0,0,0.24)",
                    radius="30",
                    withBorder=False)]
    
    return card_layout

if __name__ == '__main__':
    app.run_server(debug=True )