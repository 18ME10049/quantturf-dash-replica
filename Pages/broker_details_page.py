
import os
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import json


Paper = False
Live = False

USER_ALPACA_KEY = ""
USER_ALPACA_SECRET = ""


trade_type = dbc.Row([html.Div([
    dcc.Dropdown(['Paper', 'Live'], 'Paper', id='trade-type-dropdown'),
    html.Div(id='dd-output-container')
    ], className="mb-3")
])


key_input = dbc.Row([
        dbc.Label("ALPACA API KEY"
                , html_for="example-key-row"
                , width=2),
        dbc.Col(dbc.Input(
                type="text"
                , id="example-key-row"
                , placeholder="ENTER YOUR ALPACA API KEY VALUE"
            ),width=10,
        )],className="mb-3"
)

secret_input = dbc.Row([
        dbc.Label("ALPACA SECRET KEY", html_for="example-secret-row", width=2),
        dbc.Col(
            dbc.Input(
                type="text"
                , id="example-secret-row"
                , placeholder="ENTER YOUR ALPACA API SECRET VALUE"
                , maxLength = 80
            ),width=10
        )], className="mb-3"
)

# message = dbc.Row([
#         dbc.Label("Message", html_for="example-message-row", width=2)
#         ,dbc.Col(
#             dbc.Textarea(id = "example-message-row"
#                 , className="mb-3"
#                 , placeholder="Enter message"
#                 , required = True)
#             , width=10)
#         ], className="mb-3")

def make_layout():
    broker_form = html.Div([ dbc.Container([
            dcc.Markdown("Fill Your Broker Details")
            , html.Br()
            , dbc.Card(
                dbc.CardBody([
                     dbc.Form([trade_type
                        , key_input
                        , secret_input])
                ,html.Div(id = 'div-button', children = [
                    dbc.Button('Submit'
                    , style={'background-color': '#242324', 'color': '#FAF18F', "border-color":'#242324'}
                    , id='button-submit'
                    , n_clicks=0)
                ]) #end div
                ])#end cardbody
            )#end card
            , html.Br()
            , html.Br()
        ])
        ])
    return broker_form

def register_callbacks(app):

    # @app.callback(Output('dd-output-container', 'children'), [Input('trade-type-dropdown', 'value')])
    # def updateTradeType(input_value):
    #     if input_value=="Paper":
    #         Paper = True
    #     elif input_value == "Live":
    #         Live = True
    #     return f'You have Selected: {input_value}'
    
    # @app.callback(Output('key-output-container', 'children'), [Input('example-key-row', 'value')])
    # def updateAlpacaKey(input_key):
    #     USER_ALPACA_KEY = input_key
    
    # @app.callback(Output('secret-output-container', 'children'), [Input('example-secret-row', 'value')])
    # def updateSecretKey(input_secret):
    #     USER_ALPACA_SECRET = input_secret
    
    @app.callback(Output('dd-output-container', 'children'),[Input('button-submit', 'n_clicks')], [State('example-key-row', 'value'), State('example-secret-row', 'value')],prevent_initial_call=True)
    def save_inputs(n_clicks, value1, value2):
        if n_clicks > 0:
            if value1 is None or value2 is None:
                return f"Both KEY and Secret are required"
            else: 
                file_name = 'alpaca_input_values.json'
                data = {'ALPACA_KEY': value1, 'ALPACA_SECRET': value2}

                file_path = os.path.abspath(file_name)
                # Save the dictionary to a JSON file
                with open(file_path, 'w') as f:
                    json.dump(data, f)

                return f"Data Saved{file_path}"


