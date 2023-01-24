import importlib
from alpaca_trade_api.rest import TimeFrame
import alpaca_trade_api as tradeapi
import configuration as oc
import backend as ob
import flask
import redis
import re
from re import S
import plotly.graph_objs as go
import yahoo_fin.stock_info as si
import yfinance as yf
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
from pandas_datareader import data as web
import matplotlib.pyplot as plt
import pyfolio as pf
import dash.dependencies
from plotly.tools import mpl_to_plotly
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
import plotly.express as px
from datetime import datetime
import uuid
import zipfile
import tempfile
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Customized Bullet chart
# import pandas_datareader.data as web

plt.switch_backend('Agg')
#import empyrical
#import quantstats as qs
#from quantstats import stats


# Raw Package
# from pandas_datareader import data as pdr

# Market Data

# Graphing/Visualization

# from turtle import onclick

with open('G:\Quanturf\quantturf-dash-replica\Alpaca_input_values.json') as infile:
    data = json.load(infile)

APCA_API_KEY_ID = data['ALPACA_KEY']  # "PKWW7CAGNXC9BD8C1UEW"
APCA_API_SECRET_ID = data['ALPACA_SECRET']
BASE_URL = "https://paper-api.alpaca.markets"


# Use ALPACA Client
api = tradeapi.REST(key_id=APCA_API_KEY_ID,
                    secret_key=APCA_API_SECRET_ID, base_url=BASE_URL)
# Get list of all the Symbols Available in Alpaca
result = api.list_assets(status='active')
result_df = pd.DataFrame(columns=['class', 'exchange', 'symbol'])
class_list = []
exchange_list = []
symbol_list = []

for res in result:
    # print(res.class)
    # class_list.append(res.class)
    exchange_list.append(res.exchange)
    symbol_list.append(res.symbol)

result_df = pd.DataFrame({'exchange': exchange_list, 'Symbol': symbol_list})
symbolList = result_df.Symbol.unique().tolist()
exchangeList = result_df.exchange.unique().tolist()

company_list = symbolList

PRIMARY = '#FFFFFF'
SECONDARY = '#FFFFFF'
ACCENT = '#98C1D9'
DARK_ACCENT = '#474747'
SIDEBAR = '#F7F7F7'

# global yf_data
# yf_data = pd.DataFrame()
df_dict = {}

debug_mode = False  # set False to deploy

root_directory = os.getcwd()
stylesheets = ['tabs.css']
jss = ['script.js']
static_route = '/Static/'

# level_marks = ['Debug', 'Info', 'Warning', 'Error']
level_marks = {0: 'Debug', 1: 'Info', 2: 'Warning', 3: 'Error'}
frequencyList = ['Days', 'Ticks', 'MicroSeconds', 'Seconds',
                 'Minutes', 'Weeks', 'Months', 'Years', 'NoTimeFrame']
num_marks = 4

all_files = os.listdir("MyBacktestStrategies")
algo_files = list(filter(lambda f: f.endswith('.py'), all_files))
algo_avlb = [s.rsplit(".", 1)[0] for s in algo_files]


page = html.Div([
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        'Backtesting- ', style={'color': DARK_ACCENT}),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.CardBody([
                                                html.Div([
                                                    'Select Symbol: ',
                                                    dcc.Dropdown(
                                                        value='AMZN',
                                                        id='symbols',
                                                        options=[
                                                            {'label': name, 'value': name} for name in company_list],
                                                        # options=['AAPL', 'TSLA', 'MSFT', 'AMZN'], #Replace this with list
                                                        multi=True, clearable=False)
                                                ], className='row mb-10'),
                                                html.Div([
                                                    'Choose Algorithm',
                                                    dcc.Dropdown(
                                                        id='module-gc1', options=algo_avlb, className='eight columns u-pull-right')
                                                ], className='row mb-10'),

                                                html.Div([
                                                    # Create a date picker using the dcc.DatePicker component
                                                    'Select Start Date ',
                                                    dcc.DatePickerSingle(
                                                        id='start-date-picker',
                                                        placeholder='Start Date',
                                                        min_date_allowed=datetime(
                                                            1995, 8, 5),
                                                        max_date_allowed=datetime.today(),
                                                        initial_visible_month=datetime.today(),
                                                        date=str(
                                                            datetime.today())
                                                    ),
                                                    html.Div(id='output-date')
                                                ],  className='row mb-10'),

                                                html.Div([
                                                    # Create a date picker using the dcc.DatePicker component
                                                    'Select End Date ',
                                                    dcc.DatePickerSingle(
                                                        id='end-date-picker',
                                                        placeholder='End Date',
                                                        min_date_allowed=datetime(
                                                            1995, 8, 5),
                                                        max_date_allowed=datetime.today(),
                                                        initial_visible_month=datetime.today(),
                                                        date=str(
                                                            datetime.today())
                                                    ),
                                                    html.Div(id='output-date')
                                                ],  className='row mb-10'),

                                                html.Div([
                                                    'Enter Capital Value:',
                                                    dcc.Input(id='cash', className='eight columns u-pull-right', value=10000, style={
                                                              'margin-left': '10px', 'width': '170px', 'font-size': '15px', 'font-weight': '5', 'border-radius': 5})
                                                ], className='row mb-10'),

                                                html.Br(),
                                                html.Button('Run Backtest', id='backtest-btn', className='eight columns u-pull-right', n_clicks=0, style={
                                                            'font-size': '15px', 'font-weight': '5', 'color': '#FAF18F', 'background-color': '#242324', "border-color": '#242324', 'border-radius': 5}),
                                                # html.Button('Generate Backtest Code', id='Generate-backtest-Code', n_clicks=0, className='eight columns u-pull-right', style={
                                                #     'font-size': '15px', 'font-weight': '5', 'color': PRIMARY, 'background-color': ACCENT, "border-color": ACCENT, 'border-radius': 5}),
                                            ])]), dbc.Col([
                                            ])])
                                ], color=PRIMARY, style={'border-radius': 10, "width": "12rem"}),
                            html.Br(),
                            html.Div(
                                id='intermediate-value', style={'display': 'none'}),
                            html.Div(
                                id='intermediate-params', style={'display': 'none'}),
                            html.Div(
                                id='code-generated', style={'display': 'none'}),
                            html.Div(
                                id='code-generated-backtest-2', style={'display': 'none'}),
                            # dcc.Download(id="download-data-csv"),
                            html.Div(
                                id='intermediate-status', style={'display': 'none'}),
                            html.Div(
                                id='level-log', contentEditable='True', style={'display': 'none'}),
                            dcc.Input(
                                id='log-uid', type='text', style={'display': 'none'})
                        ], width=2),
                        dbc.Col([

                            html.Div([
                                dbc.Card(
                                    dbc.CardBody([
                                        dbc.Tabs(
                                            [
                                                dbc.Tab(dcc.Graph(id='charts', config={
                                                    'displayModeBar': False}), label='Backtest', className='nav-pills'),
                                            ],
                                            id='tabs',
                                            # active_tab='tab-1',
                                        ),

                                    ]), color=SECONDARY, style={'border-radius': 10}
                                ),
                            ]),
                        ], width=7),

                        dbc.Col([
                            html.Div([
                                dbc.Card(
                                    dbc.CardBody([
                                        html.Div(id='stat-block')
                                    ]), color=SECONDARY, style={'border-radius': 10}
                                )
                            ])
                        ], width=3)
                    ]),
                    dbc.Row([
                        html.Div([
        dash_table.DataTable(
    id='logs-table',)
    ])
                    ]),
                ])
                )
], id='graph-container', style={'margin-bottom': '30rem'})


def make_layout():
    return page
#     return html.Div([page, html.Div([
#         dash_table.DataTable(
#     id='logs-table',
#     columns=[{'name': col, 'id': col} for col in logs_df.columns],
#     data=logs_df.to_dict('records'),
# )
#     ])])



PRIMARY = '#FFFFFF'
SECONDARY = '#FFFFFF'
ACCENT = '#EF5700'
DARK_ACCENT = '#474747'
SIDEBAR = '#F7F7F7'

# PRIMARY = '#15202b'
# SECONDARY = '#192734'
# ACCENT = '#FFFFFF'
# SIDEBAR = '#F4511E'
# F4511E

DATATABLE_STYLE = {
    'color': 'white',
    'backgroundColor': PRIMARY,
}

DATATABLE_HEADER = {
    'backgroundColor': SIDEBAR,
    'color': 'white',
    'fontWeight': 'bold',
}

TABS_STYLES = {
    'height': '44px'
}
TAB_STYLE = {
    'padding': '15px',
    'fontWeight': 'bold',
    'color': DARK_ACCENT,
    'backgroundColor': SECONDARY,
    'borderRadius': '10px',
    'margin-left': '6px',
}

TAB_SELECTED_STYLE = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': ACCENT,
    'color': PRIMARY,
    'padding': '15px',
    'borderRadius': '10px',
    'margin-left': '6px',
}

# helper function for closing temporary files


def close_tmp_file(tf):
    try:
        os.unlink(tf.name)
        tf.close()
    except:
        pass

# Text field


def drawText(title, text):
    return html.Div([
        dbc.Card([
                    dbc.CardHeader(title, style={'color': DARK_ACCENT}),
                    dbc.CardBody([
                        html.Div([
                            # html.Header(title, style={'color': 'white', 'fontSize': 15, 'text-decoration': 'underline', 'textAlign': 'left'}),
                            # html.Br(),
                            html.Div(str(round(text, 2)), style={
                                'color': DARK_ACCENT, 'textAlign': 'center'}),
                            # str(round(text, 2))
                        ], style={'color': DARK_ACCENT})
                    ])
                    ], color=PRIMARY, style={'height': 100, 'border-radius': 10}),  # , 'backgroundColor':'#FFFFFF', 'border':'1px solid'
    ])


def beautify_plotly(fig):
    return html.Div([
                    dbc.Card(
                        dbc.CardBody([
                            dcc.Graph(
                                figure=fig,
                                config={
                                    'displayModeBar': False
                                }
                            )
                        ]), color=SECONDARY, style={'border-radius': 10}
                    ),
                    ])


def register_callbacks(app):

    @app.callback(Output('intermediate-value', 'children'), Output('logs-table', 'data'), [Input('backtest-btn', 'n_clicks')])
    def on_click_backtest_to_intermediate(n_clicks):
        if n_clicks != 0:
            try:
                strategy = "MyStrategy1"
                result, logs = ob.create_ts2(strategy)
                logs_df = pd.DataFrame(logs)
                return result, logs_df.to_dict('records')
            except json.decoder.JSONDecodeError:
                # Ignoring this error (this is happening when inputting values in Module/Strategy boxes)
                return []

    @app.callback(Output('charts', 'figure'),
                  [Input('intermediate-value', 'children')], prevent_initial_call=True)
    def on_intermediate_to_chart(children):
        # r = redis.StrictRedis(oc.cfg['default']['redis'], 6379, db=0)
        # size = r.get(uid + 'size')
        # w, h = size.decode('utf8').split(',')
        # return ob.extract_figure(children, w, h)
        if children == None or len(children) == 0:
            return dash.no_update
        return ob.extract_figure(children)

    @app.callback(Output('stat-block', 'children'), [Input('intermediate-value', 'children')])
    def on_intermediate_to_stat(children):
        statistic = ob.extract_statistic(children)
        ht = []
        for section in statistic:
            ht.append(html.Div(html.B(section, style={
                      'font-size': '1.1em', 'line-height': '1.5m'}), className='row'))
            for stat in statistic[section]:
                ht.append(
                    html.Div([
                        html.Div(
                            children=[html.H6(stat + " = " + str(statistic[section].get(stat)))])
                        # html.Div(stat, className='u-pull-left'),
                        # html.Div(html.B(statistic[section].get(
                        #     stat)), className='u-pull-right')
                    ], className='row'))
            ht.append(
                html.Div(style={'border': '2px solid #999', 'margin': '10px 10px 5px'}))
        return html.Div(ht[:-1])

    # @app.callback(Output('strategy', 'options'), [Input('symbols', 'value')])
    # def update_strategy_list(symbols):
    #     all_files = os.listdir("MyStrategies")
    #     backtest_files = list(filter(lambda f: f.endswith('.py'), all_files))
    #     backtest_avlb = [s.rsplit(".", 1)[0] for s in backtest_files]
    #     # print(backtest_avlb)
    #     return backtest_avlb

    # @app.server.route('{}<file>'.format(static_route))
    # def serve_file(file):
    #     if file not in stylesheets and file not in jss:
    #         raise Exception(
    #             '"{}" is excluded from the allowed static css files'.format(file))
    #     static_directory = os.path.join(root_directory, 'Static')
    #     return flask.send_from_directory(static_directory, file)

    ####  Run Backtest button #####

    # @app.callback(Output('status-area', 'children'),
    #               [
    #     Input('backtest-btn', 'n_clicks'),
    #     Input('strategy', 'value'),
    #     Input('intermediate-value', 'children')
    # ])
    # def update_status_area(n_clicks, strategy, result):
    #     if result:
    #         return 'Done!'
    #     if n_clicks == 0:
    #         return ''
    #     #strategy = None

    #     if strategy is None:
    #         return 'Please provide a value for: {}!'.format(', '.join(strategy))

    #     return "Backtesting.."

    # @app.callback(Output('log-uid', 'value'), [Input('symbols', 'options')])
    # def create_uid(m):
    #     return uuid.uuid4().hex

    # @app.callback(Output('backtest-btn', 'n_clicks'),
    #               [
    #     #Input('module', 'value'),
    #     Input('strategy', 'value')
    #     #Input('symbols', 'value'),
    #     #Input('params-table', 'columns')
    # ])
    # def reset_button(*args):
    #     return 0

#     @app.callback(Output('code-generated-backtest-2', 'children'),
#                   [
#         Input('Generate-backtest-Code', 'n_clicks'),
#         Input('symbols', 'value'),
#         Input('module-gc1', 'value'),
#         Input('backtest-freqeuncy-selected-property', 'value'),
#         Input('filename', 'value'),
#     ])

    #####  Download Button #####

    # @app.callback(Output('code-generated2', 'children'),
    #               [
    #     Input('download-btn', 'n_clicks'),
    #     Input('symbols', 'value')
    # ])
    # def download_data(n_clicks, symbols):
    #     if n_clicks == 0:
    #         return ''
    #     #symbols = ['TSLA', 'GE']
    #     print("testing Datas ")
    #     print(symbols)
    #     for s in symbols:
    #         df = yf.download(s, start="2018-01-01")
    #         data_dir = "Data/"
    #         filename = s + ".csv"
    #         df.to_csv(os.path.join(data_dir, filename))
    #     return 0

    # Commenting it out for now as there is no level-slider exist.

    # @app.callback(
    #     dash.dependencies.Output('level-log', 'children'),
    #     [dash.dependencies.Input('level-slider', 'value')])
    # def level_output(value):
    #     begin, end = value
    #     res = []
    #     for i in range(begin, end+1):
    #         res.append(level_marks[i].upper())
    #     return ','.join(res)

    # if not debug_mode:
    #     auth = dash_auth.BasicAuth(
    #         app,
    #         ob.get_users()
    #     )


key_metrics_df = pd.DataFrame()


def update_code(symbols, cash, strategy):
    backTestCode = f"""import alpaca_backtrader_api
import backtrader as bt
from datetime import datetime

# Your credentials here
ALPACA_API_KEY = "{APCA_API_KEY_ID}"
ALPACA_SECRET_KEY = "{APCA_API_SECRET_ID}"

IS_BACKTEST = True
IS_LIVE = False
symbol = "{symbols}"


class SmaCross1(bt.Strategy):
    def notify_fund(self, cash, value, fundvalue, shares):
        super().notify_fund(cash, value, fundvalue, shares)

    def notify_store(self, msg, *args, **kwargs):
        super().notify_store(msg, *args, **kwargs)
        self.log(msg)

    def notify_data(self, data, status, *args, **kwargs):
        super().notify_data(data, status, *args, **kwargs)
        print('*' * 5, 'DATA NOTIF:', data._getstatusname(status), *args)
        if data._getstatusname(status) == "LIVE":
            self.live_bars = True

    # list of parameters which are configurable for the strategy
    params = dict(
        pfast=10,  # period for the fast moving average
        pslow=30   # period for the slow moving average
    )

    def log(self, txt, dt=None):
        dt = dt or self.data.datetime[0]
        dt = bt.num2date(dt)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_trade(self, trade):
        self.log("placing trade for {{}}. target size: {{}}".format(
            trade.getdataname(),
            trade.size))

    def notify_order(self, order):
        print(order)
        print(f"Order notification. status {{order.getstatusname()}}.")
        print(f"Order info. status {{order.info}}.")
        #print(f'Order - {{order.getordername()}} {{order.ordtypename()}} {{order.getstatusname()}} for {{order.size}} shares @ ${{order.price:.2f}}')

    def stop(self):
        print('==================================================')
        print('Starting Value - %.2f' % self.broker.startingcash)
        print('Ending   Value - %.2f' % self.broker.getvalue())
        print('==================================================')

    def __init__(self):
        self.live_bars = False
        sma1 = bt.ind.SMA(self.data0, period=self.p.pfast)
        sma2 = bt.ind.SMA(self.data0, period=self.p.pslow)
        self.crossover0 = bt.ind.CrossOver(sma1, sma2)

    def next(self):
        #self.buy(data=data0, size=2)
        if not self.live_bars and not IS_BACKTEST:
            # only run code if we have live bars (today's bars).
            # ignore if we are backtesting
            return
        # if fast crosses slow to the upside
        if not self.positionsbyname[symbol].size and self.crossover0 > 0:
            self.buy(data=self.data0, size=5)  # enter long

        # in the market & cross to the downside
        if self.positionsbyname[symbol].size and self.crossover0 <= 0:
            self.close(data=self.data0)  # close long position


def runStrategy():
    import logging
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    cerebro = bt.Cerebro()

    store = alpaca_backtrader_api.AlpacaStore(
        key_id=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
        paper=not IS_LIVE,
    )

    DataFactory = store.getdata  # or use alpaca_backtrader_api.AlpacaData
    if IS_BACKTEST:
        data0 = DataFactory(dataname=symbol,
                            historical=True,
                            fromdate=datetime(2021, 7, 1),
                            todate=datetime(2022, 7, 11),
                            timeframe=bt.TimeFrame.Days,
                            data_feed='iex')
    else:
        data0 = DataFactory(dataname=symbol,
                            historical=False,
                            timeframe=bt.TimeFrame.Ticks,
                            backfill_start=False,
                            data_feed='iex'
                            )
        # or just alpaca_backtrader_api.AlpacaBroker()
        broker = store.getbroker()
        cerebro.setbroker(broker)
    #cerebro.broker.setcash(100000)
    cerebro.adddata(data0)
    cerebro.addstrategy(SmaCross1)

    #add Analyzers
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='SQN')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    if IS_BACKTEST:
        # backtrader broker set initial simulated cash
        cerebro.broker.setcash({cash})

    print('Starting Portfolio Value: {{}}'.format(cerebro.broker.getvalue()))
    results = cerebro.run()
    pnl = cerebro.broker.getvalue() - {cash}
    print('Final Portfolio Value: {{}}'.format(cerebro.broker.getvalue()))
    return pnl, results[0]
    #cerebro.plot()"""
    # .format(APCA_API_KEY_ID = APCA_API_KEY_ID, APCA_API_SECRET_ID = APCA_API_SECRET_ID, live = live, symbol=symbol)

    # strategy_file=strategy+".py"
    # strategy_file = "SampleStrategies/"+strategy_file

    # with open(strategy_file) as fp:
    #     data = fp.read()
    # data += "\n"

    data = backTestCode
    path_dir = "MyBacktestStrategies/"
    filename_save = strategy+".py"

    with open(os.path.join(path_dir, filename_save), 'w') as fp:
        fp.write(data)


# def balance_sheet(symbol):

#     ticker = symbol
#     data = yf.Ticker(ticker)

#     df = pd.DataFrame(data.balance_sheet).T

#     return html.Div([
#                     dbc.Card(
#                         dbc.CardBody([dash_table.DataTable(df.to_dict('records'), [{'name': i, 'id': i} for i in df.columns],
#                                                            style_data=DATATABLE_STYLE, style_header=DATATABLE_HEADER, style_table={'overflowX': 'auto'})
#                                       ]), color=SECONDARY
#                     ),
#                     ])


# def eps_trend(symbol):

#     ticker = symbol
#     df = si.get_analysts_info(ticker)['EPS Trend'].assign(
#         hack='').set_index('hack')
#     return html.Div([
#         dbc.Card(
#                     dbc.CardBody([dash_table.DataTable(df.to_dict('records'), [{'name': i, 'id': i} for i in df.columns],
#                                                        style_data=DATATABLE_STYLE, style_header=DATATABLE_HEADER, style_table={'overflowX': 'auto'})
#                                   ]), color=SECONDARY
#                     ),
#     ])


# def growth_estimates(symbol):
#     ticker = symbol
#     df = si.get_analysts_info(ticker)['Growth Estimates'].assign(
#         hack='').set_index('hack')
#     return html.Div([
#         dbc.Card(
#                     dbc.CardBody([dash_table.DataTable(df.to_dict('records'), [{'name': i, 'id': i} for i in df.columns],
#                                                        style_data=DATATABLE_STYLE, style_header=DATATABLE_HEADER, style_table={'overflowX': 'auto'})
#                                   ]), color=SECONDARY
#                     ),
#     ])


# def earnings_estimate(symbol):
#     ticker = symbol
#     df = si.get_analysts_info(ticker)['Earnings Estimate'].assign(
#         hack='').set_index('hack')
#     return html.Div([
#         dbc.Card(
#                     dbc.CardBody([dash_table.DataTable(df.to_dict('records'), [{'name': i, 'id': i} for i in df.columns],
#                                                        style_data=DATATABLE_STYLE, style_header=DATATABLE_HEADER, style_table={'overflowX': 'auto'})
#                                   ]), color=SECONDARY
#                     ),
#     ])


# def revenue_estimate(symbol):
#     ticker = symbol
#     df = si.get_analysts_info(ticker)['Revenue Estimate'].assign(
#         hack='').set_index('hack')
#     return html.Div([
#         dbc.Card(
#                     dbc.CardBody([dash_table.DataTable(df.to_dict('records'), [{'name': i, 'id': i} for i in df.columns],
#                                                        style_data=DATATABLE_STYLE, style_header=DATATABLE_HEADER, style_table={'overflowX': 'auto'})
#                                   ]), color=SECONDARY
#                     ),
#     ])


# def earnings_history(symbol):
#     ticker = symbol
#     df = si.get_analysts_info(ticker)['Earnings History'].assign(
#         hack='').set_index('hack')
#     return html.Div([
#         dbc.Card(
#                     dbc.CardBody([dash_table.DataTable(df.to_dict('records'), [{'name': i, 'id': i} for i in df.columns],
#                                                        style_data=DATATABLE_STYLE, style_header=DATATABLE_HEADER, style_table={'overflowX': 'auto'})
#                                   ]), color=SECONDARY
#                     ),
#     ])


# def eps_revisions(symbol):
#     ticker = symbol
#     df = si.get_analysts_info(ticker)['EPS Revisions'].assign(
#         hack='').set_index('hack')
#     return html.Div([
#         dbc.Card(
#                     dbc.CardBody([dash_table.DataTable(df.to_dict('records'), [{'name': i, 'id': i} for i in df.columns],
#                                                        style_data=DATATABLE_STYLE, style_header=DATATABLE_HEADER, style_table={'overflowX': 'auto'})
#                                   ]), color=SECONDARY
#                     ),
#     ])


# def income_statement(symbol):

#     ticker = symbol
#     data = yf.Ticker(ticker)

#     df = pd.DataFrame(data.financials).T
#     return html.Div([
#                     dbc.Card(
#                         dbc.CardBody([dash_table.DataTable(df.to_dict('records'), [{'name': i, 'id': i} for i in df.columns],
#                                                            style_data=DATATABLE_STYLE, style_header=DATATABLE_HEADER, style_table={'overflowX': 'auto'})
#                                       ]), color=SECONDARY
#                     ),
#                     ])


# def cash_flows(symbol):

#     ticker = symbol
#     data = yf.Ticker(ticker)

#     df = pd.DataFrame(data.cashflow).T
#     return html.Div([
#                     dbc.Card(
#                         dbc.CardBody([dash_table.DataTable(df.to_dict('records'), [{'name': i, 'id': i} for i in df.columns],
#                                                            style_data=DATATABLE_STYLE, style_header=DATATABLE_HEADER, style_table={'overflowX': 'auto'})
#                                       ]), color=SECONDARY
#                     ),
#                     ])
