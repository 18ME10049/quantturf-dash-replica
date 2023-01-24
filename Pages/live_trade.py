
import datetime
from collections import deque
import plotly.graph_objs as go
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
from pandas_datareader import data as web
import empyrical
import matplotlib.pyplot as plt
import pyfolio as pf
import dash.dependencies
from plotly.tools import mpl_to_plotly
import plotly.express as px
import datetime as dt
import os
import json
from time import sleep
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from threading import Thread
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import warnings

import requests
warnings.filterwarnings('ignore')
# import tempfile
# import zipfile

# Customized Bullet chart

plt.switch_backend('Agg')
# import quantstats as qs
# from quantstats import stats


# Raw Package
# from pandas_datareader import data as pdr

# Alpaca Api

# Graphing/Visualization


# global yf_data
# yf_data = pd.DataFrame()
df_dict = {}

all_paper_strategy_files = os.listdir("MyLiveStrategies")
stategy_list = list(
    filter(lambda f: f.endswith('.py'), all_paper_strategy_files))
list_select_strategy = [s.rsplit(".", 1)[0] for s in stategy_list]

# Styles

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

# Dollar_Pnl, Return, Realized_PL, Unrealized_PL, Total_Val, Max_DD, Aval_Cash, Win_Rate = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


def make_layout():

    # genrate_datasheet_for_unrealised_realised_profits, top_stats, performance_pnl_vector, close_Orders_Sheets, open_Orders_Sheets= key_metrics()
    # global Dollar_Pnl, Return, Realized_PL, Unrealized_PL, Total_Val, Max_DD, Aval_Cash, Win_Rate
    # Dollar_Pnl, Return, Realized_PL, Unrealized_PL, Total_Val, Max_DD, Aval_Cash, Win_Rate = top_stats()
    # status = genrate_datasheet_for_unrealised_realised_profits
    # print(status)
    return html.Div([
        dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Select Strategy', style={
                                   'color': DARK_ACCENT}),
                    dbc.CardBody([
                        # Run backtest
                        html.Div([
                            dcc.Dropdown(
                                id='run-paper-strategy',
                                options=list_select_strategy,
                            ),
                            html.Div(id='output_container')
                        ]),

                        html.Br(),
                        dbc.Row([
                            # html.Br(),
                            dbc.Col([
                                html.Button('Live Trade', id='run-paper-btn', className='eight columns u-pull-right', n_clicks=0, style={
                                    'font-size': '15px', 'font-weight': '5', 'color': '#FAF18F', 'background-color': '#242324', "border-color": '#242324', 'border-radius': 5}),
                            ]),
                            html.Br(),

                            dbc.Col([
                                html.Button('Cloud Deploy', id='deploy-cloud-btn', className='eight columns u-pull-right', n_clicks=0, style={
                                    'font-size': '15px', 'font-weight': '5', 'color': '#FAF18F', 'background-color': '#242324', "border-color": '#242324', 'border-radius': 5}),
                                html.Div(id='output-message'),
                            ]),
                        ]),
                        html.Div(
                            id='intermediate-value', style={'display': 'none'}),
                        html.Div(
                            id='intermediate-params', style={'display': 'none'}),
                        html.Div(
                            id='code-generated', style={'display': 'none'}),
                        html.Div(
                            id='code-generated2', style={'display': 'none'}),
                        # dcc.Download(id="download-data-csv"),
                        html.Div(
                            id='intermediate-status', style={'display': 'none'}),
                        html.Div(
                            id='level-log', contentEditable='True', style={'display': 'none'}),
                        dcc.Input(
                            id='log-uid', type='text', style={'display': 'none'})
                    ])
                ], color=PRIMARY, style={'border-radius': 10}),
                ], width=3),
        dbc.Card(
            dbc.CardBody([
                dbc.Row(children=createMetric(), id='metric_row'),
                html.Br(),
                dbc.Row([
                    # dbc.Col([
                    # 	eps_trend(symbol),
                    # 	eps_revisions(symbol)
                    # ], width=3),
                    dbc.Col([
                        dbc.Tabs([
                            dbc.Tab(children=[
                                html.Div([
                                    dcc.Graph(
                                        id='Graph_live', animate=True, figure=performance_pnl_vector()),
                                    dcc.Interval(
                                        id='interval-component',
                                        interval=10*1000,  # in milliseconds
                                        n_intervals=0
                                    )
                                ])
                            ], id='performance_chart', label='Performance'),
                            dbc.Tab(children=[
                                html.Div([
                                    dash_table.DataTable(

                                        id='close-live-update-table',
                                        data=close_Orders_Sheets(),
                                        style_data_conditional=[
                                            {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(220, 220, 220)'}],
                                        style_header={
                                            'backgroundColor': 'rgb(210, 210, 210)', 'color': 'black', 'fontWeight': 'bold'}
                                    )
                                ])
                            ], id='close_positions_sheet', label='Closed Position'),
                            dbc.Tab(children=[
                                html.Div([
                                    dash_table.DataTable(

                                        id='open-live-update-table',
                                        data=open_Orders_Sheets(),
                                        style_data_conditional=[
                                            {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(220, 220, 220)'}],
                                        style_header={
                                            'backgroundColor': 'rgb(210, 210, 210)', 'color': 'black', 'fontWeight': 'bold'}
                                    )
                                ])
                            ], id='open_positions_sheet', label='Open Positions')
                        ], id='tabs',),
                    ], width=9)
                ], align='center'),
                html.Br(),
            ]), color=PRIMARY, style={'border-radius': 10}  # all cell border
        )
    ], style={'margin-bottom': '30rem'})


# helper function for closing temporary files
def close_tmp_file(tf):
    try:
        os.unlink(tf.name)
        tf.close()
    except:
        pass


class Alpaca_order:
  def __init__(self, client_order_id, filled_avg_price, filled_qty, side, symbol, qty, updated_at,status):
    self.client_order_id = client_order_id
    self.filled_avg_price = filled_avg_price
    self.filled_qty = filled_qty
    self.side = side
    self.symbol = symbol
    self.qty = qty
    self.updated_at = updated_at
    self.status = status

# def key_metrics():


def getAlpcaClient():
    with open('alpaca_input_values.json') as infile:
        data = json.load(infile)
    APCA_API_KEY_ID = data['ALPACA_KEY']  # "PKWW7CAGNXC9BD8C1UEW"
    APCA_API_SECRET_ID = data['ALPACA_SECRET']
    BASE_URL = "https://paper-api.alpaca.markets"
    client = tradeapi.REST(
        key_id=APCA_API_KEY_ID, secret_key=APCA_API_SECRET_ID, base_url=BASE_URL, api_version="v2")
    return client


def get_orderids():
    datasheet_dir = "Datasheets/"
    Orderid_df = pd.read_csv(os.path.join(
        datasheet_dir, "orderids.csv"), parse_dates=True)
    orderid_list = Orderid_df['OrderID'].tolist()
    return orderid_list


def get_transactions_details_of_given_orderids_for_a_strategy(api, order_ids):
    result = api.list_orders(status='closed', limit=500)
    result_df = pd.DataFrame()

    #print(result[0]) # cum_qty, price, symbol, transaction_time

    result_df = pd.DataFrame(columns = ['Qty','price','symbol','transaction_time','order_id'])

    order_id_list = []
    qty_list = []
    price_list = []
    symbol_list = []
    transaction_time_list = []
    type_list = []

    for res in result:
        if (res.client_order_id in order_ids):
            order_id_list.append(res.client_order_id)
            qty_list.append(res.filled_qty)
            price_list.append(res.filled_avg_price)
            symbol_list.append(res.symbol)
            transaction_time_list.append(res.updated_at)
            type_list.append(res.side)



    result_df = pd.DataFrame({
        'order_id':order_id_list,
        'Qty': qty_list, 
        'Price': price_list, 
        'Symbol': symbol_list, 
        'Transaction_time': transaction_time_list, 
        'Type': type_list})

    return result_df


def realized_profit_df_strategy(api, order_ids):

    result = api.list_orders(status='closed', limit=500)#api.get_activities()

   # result_df = pd.DataFrame(columns = ['Order_Id','Symbol', 'Qty','Price', 'RealisedPL_Per_Unit', 'Total_RealisedPL','Transaction_Time'])

    buy_and_sell_order_list = []
    # buy_order_list = []

    for res in result:
        if (res.client_order_id in order_ids):
            # buy_order_list.append(Alpaca_order(res.client_order_id, float(res.filled_avg_price), int(res.filled_qty), res.side, res.symbol, res.qty, res.updated_at, res.status))
            buy_and_sell_order_list.append(Alpaca_order(res.client_order_id, float(res.filled_avg_price), int(res.filled_qty), res.side, res.symbol, res.qty, res.updated_at, res.status))
    
    #Lists to create a dataframe for sell_orders_profits_and_loss
    order_id_list = []
    qty_list = []
    price_list = []
    cost_price_list = []
    symbol_list = []
    transaction_time_list = []
    type_list = []
    realised_profit_per_unit = []
    total_realised_profit = []

    last_unsold_buy_order_index = len(buy_and_sell_order_list)-1

    for current_order in reversed(buy_and_sell_order_list):
        if current_order.side == 'sell':
            current_sold_qty = current_order.filled_qty
            buy_order_qty = 0
            buy_order_average_cost_price = 0.0
            while last_unsold_buy_order_index >= 0:
                current_buy_order_qty = buy_order_qty + buy_and_sell_order_list[last_unsold_buy_order_index].filled_qty
                if current_buy_order_qty >= current_sold_qty:
                    buy_order_average_cost_price += (current_sold_qty - buy_order_qty) * buy_and_sell_order_list[last_unsold_buy_order_index].filled_avg_price
                    if current_buy_order_qty != current_sold_qty:
                        buy_and_sell_order_list[last_unsold_buy_order_index].filled_qty = current_buy_order_qty - current_sold_qty
                    break
                else:
                    buy_order_average_cost_price += (buy_and_sell_order_list[last_unsold_buy_order_index].filled_qty) * buy_and_sell_order_list[last_unsold_buy_order_index].filled_avg_price
                    buy_order_qty = current_buy_order_qty
                    last_unsold_buy_order_index -= 1
            
            buy_order_average_cost_price = buy_order_average_cost_price / current_sold_qty

            order_id_list.append(current_order.client_order_id)
            qty_list.append(current_order.filled_qty)
            price_list.append(current_order.filled_avg_price)
            symbol_list.append(current_order.symbol)
            transaction_time_list.append(current_order.updated_at)
            cost_price_list.append(round(buy_order_average_cost_price,2))
            realised_profit_per_unit.append(round(current_order.filled_avg_price-buy_order_average_cost_price, 2))
            total_realised_profit.append(round(qty_list[-1]*realised_profit_per_unit[-1],2))


    WinningBetList = [item > 0 for item in realised_profit_per_unit]

    result_df = pd.DataFrame({
        'Order_Id':order_id_list,
        'Symbol': symbol_list,
        'selling_qty': qty_list, 
        'Avg_selling_Price': price_list,
        'Avg_buying_cost': cost_price_list,
        'Profit_per_unit': realised_profit_per_unit,
        'Total Profit': total_realised_profit, 
        'Sell_time': transaction_time_list,
        'Winning_bet?': WinningBetList})

    return result_df


def get_all_open_transactions_unrealised_profit(api, order_ids):
    result = api.list_orders(status='closed', limit=500)#api.get_activities()

    result_df = pd.DataFrame()

    result_df = pd.DataFrame(columns = ['Order_Id','Symbol', 'Qty','Price', 'Unrealised_Profit_Per_Unit', 'Total_Unrealised_Profit','Transaction_Time'])

    sell_order_list = []
    buy_order_list = []

    for res in result:
        if (res.client_order_id in order_ids):
            if res.side == "buy":
                buy_order_list.append(Alpaca_order(res.client_order_id, float(res.filled_avg_price), int(res.filled_qty), res.side, res.symbol, res.qty, res.updated_at, res.status))
            elif res.side == "sell":
                sell_order_list.append(Alpaca_order(res.client_order_id, float(res.filled_avg_price), int(res.filled_qty), res.side, res.symbol, res.qty, res.updated_at, res.status))

    for sell_order in reversed(sell_order_list):
        curr_sell_order_symbol = sell_order.symbol
        curr_sell_order_qty = sell_order.filled_qty
        curr_sell_order_transaction_time = sell_order.updated_at

        buy_order_index_that_are_closed = []
        for index, buy_order in reversed(list(enumerate(buy_order_list))):
            curr_buy_order_qty = buy_order.filled_qty
            if curr_sell_order_qty == 0:
                break
            if buy_order.symbol == curr_sell_order_symbol and buy_order.updated_at < curr_sell_order_transaction_time:
                if curr_buy_order_qty <= curr_sell_order_qty:
                    curr_sell_order_qty = curr_sell_order_qty - curr_buy_order_qty
                    buy_order_index_that_are_closed.append(index)
                elif curr_buy_order_qty > curr_sell_order_qty:
                    buy_order.filled_qty = buy_order.filled_qty - curr_sell_order_qty
                    break
        #update the buy order containing only open position orders
        buy_order_list = [item for idx, item in enumerate(buy_order_list) if idx not in buy_order_index_that_are_closed]
    
    # #update the buy order containing only open position orders
    # updated_buy_order_list = [item for idx, item in enumerate(buy_order_list) if idx not in buy_order_index_that_are_closed]

    #create lists to create dataframe
    order_id_list = []
    qty_list = []
    price_list = []
    symbol_list = []
    transaction_time_list = []
    type_list = []
    unrealised_profit_per_unit = []
    total_unrealised_profit = []
    current_price = 100.00 #Getting exact price is not possible as alpaca does not show data for last 15minutes.

    account_positions = api.list_positions()
    current_price_dict = {}
    for position in account_positions:
        current_price_dict[position.symbol] = float(position.current_price)

    # current_price_dict["META"] = 0.0
    for res in buy_order_list:
        current_price = current_price_dict[(res.symbol).replace("/", "")]
        order_id_list.append(res.client_order_id)
        qty_list.append(res.filled_qty)
        price_list.append(res.filled_avg_price)
        symbol_list.append(res.symbol)
        transaction_time_list.append(res.updated_at)
        unrealised_profit_per_unit.append(round(current_price-res.filled_avg_price, 2))
        total_unrealised_profit.append(round(qty_list[-1]*unrealised_profit_per_unit[-1],2))


    result_df = pd.DataFrame({
        'Order_Id':order_id_list,
        'Symbol': symbol_list,
        'Qty': qty_list, 
        'Price': price_list,
        'Unrealised_Profit_Per_Unit': unrealised_profit_per_unit,
        'Total_Unrealised_Profit': total_unrealised_profit, 
        'Transaction_Time': transaction_time_list})
    result_df['Transaction_Time'] = pd.to_datetime(result_df['Transaction_Time'])
    return result_df


def get_pnl_df_strategy(open_positions, close_positions):
    # Open
    df_copy_open = open_positions.copy()
    df_copy_open['date'] = df_copy_open['Transaction_Time'].dt.date
    unrealized_pnl_df = df_copy_open.groupby(
        'date')['Total_Unrealised_Profit'].sum().reset_index()

    # Close
    df_copy_close = close_positions.copy()
    df_copy_close['Sell_time'] = pd.to_datetime(df_copy_close['Sell_time'])
    df_copy_close['date'] = df_copy_close['Sell_time'].dt.date
    realized_pnl_df = df_copy_close.groupby(
        'date')['Total Profit'].sum().reset_index()

    df = pd.merge(unrealized_pnl_df, realized_pnl_df, on='date', how='outer')
    df = df.rename(columns={'Total Profit': 'realized_pnl',
                   'Total_Unrealised_Profit': 'unrealized_pnl'})
    df = df.fillna(0)
    df['total_pnl'] = df['realized_pnl'] + df['unrealized_pnl']
    return df


def generate_datasheet_for_unrealised_realised_profits():
    client = getAlpcaClient()
    order_id_list = get_orderids()
    close_df = realized_profit_df_strategy(client, order_id_list)
    # close_df = pd.read_csv(os.path.join(
    #     datasheet_dir, "close.csv"), parse_dates=True)
    # print("done!-Close Get")
    # close_df['Sell_time'] = pd.to_datetime(close_df['Sell_time'])
    # print("done!-SellTime")
    open_df = get_all_open_transactions_unrealised_profit(
        client, order_id_list)
    pnl_df = get_pnl_df_strategy(open_df, close_df)
    close_df.to_csv("Datasheets/close.csv", index=False)
    open_df.to_csv("Datasheets/open.csv", index=False)
    pnl_df.to_csv("Datasheets/pnl_df.csv", index=False)
    return "Yes"


def top_stats():
    client = getAlpcaClient()
    accountInfo = client.get_account()
    datasheet_dir = "Datasheets/"
    pnldf = pd.read_csv(os.path.join(
        datasheet_dir, "pnl_df.csv"), parse_dates=True)
    closedf = pd.read_csv(os.path.join(
        datasheet_dir, "close.csv"), parse_dates=True)
    opendf = pd.read_csv(os.path.join(
        datasheet_dir, "open.csv"), parse_dates=True)
    Realized_PL = pnldf['realized_pnl'].sum(skipna=True)
    Unrealized_PL = pnldf['unrealized_pnl'].sum(skipna=True)
    Dollar_Pnl = Realized_PL+Unrealized_PL
    InitialCash = 100000
    Total_Val = float(accountInfo.equity)
    Max_DD = pnldf['total_pnl'].min(skipna=True)
    Aval_Cash = float(accountInfo.cash)
    Return = (Total_Val - InitialCash)*100/InitialCash
    try:
        yes_count = closedf['Winning_bet?'].value_counts()[True]
    except Exception:
        yes_count = 0
    total_count = len(closedf['Winning_bet?'])
    if total_count == 0:
        Win_Rate = 0
    else:    
        Win_Rate = (yes_count / total_count) * 100
    return [
        Dollar_Pnl,
        Return,
        Realized_PL,
        Unrealized_PL,
        Total_Val,
        Max_DD,
        Aval_Cash,
        Win_Rate
    ]


dataCount = 1

now = datetime.datetime.now()
X = deque([now], maxlen=30)

Y = deque(maxlen=30)
datasheet_dir = "Datasheets/"
pnldf = pd.read_csv(os.path.join(
    datasheet_dir, "pnl_df.csv"), parse_dates=True)
total_sum_pnldf = pnldf['total_pnl'].sum()
# print(" Data Point Number . {} : {}".format(dataCount, total_sum_pnldf))
# dataCount+=1
Y.append(round(total_sum_pnldf, 2))


def performance_pnl_vector():
    datasheet_dir = "Datasheets/"
    pnldf = pd.read_csv(os.path.join(
        datasheet_dir, "pnl_df.csv"), parse_dates=True)
    total_sum_pnldf = pnldf['total_pnl'].sum()

    X.append(datetime.datetime.now())
    # global dataCount
    # print(" Data Point Number . {} : {}".format(dataCount, total_sum_pnldf))
    # dataCount+=1
    Y.append(round(total_sum_pnldf, 2))

    data = go.Scatter(
        x=list(X),
        y=list(Y),
        name='Scatter',
        mode='lines+markers'
    )
    return {'data': [data],
            'layout': go.Layout(xaxis=dict(range=[min(X), max(X)]), yaxis=dict(range=[min(Y), max(Y)]),)}


def close_Orders_Sheets():
    datasheet_dir = "Datasheets/"
    closedf = pd.read_csv(os.path.join(
        datasheet_dir, "close.csv"), parse_dates=True)
    return closedf.to_dict("rows")


def open_Orders_Sheets():
    datasheet_dir = "Datasheets/"
    opendf = pd.read_csv(os.path.join(
        datasheet_dir, "open.csv"), parse_dates=True)
    return opendf.to_dict("rows")

# return  generate_datasheet_for_unrealised_realised_profits(), top_stats(), performance_pnl_vector(), close_Orders_Sheets(), open_Orders_Sheets()


def drawText(title, text):
    return html.Div([
        dbc.Card([
                    dbc.CardHeader(title, style={'color': DARK_ACCENT}),
                    dbc.CardBody([
                        html.Div([
                            # html.Header(title, style={'color': 'white', 'fontSize': 15, 'text-decoration': 'underline', 'textAlign': 'left'}),
                            # html.Br(),
                            html.Div(str(round(text, 2)), style={
                                'color': DARK_ACCENT, 'textAlign': 'left'}),
                            # str(round(text, 2))
                        ], style={'color': DARK_ACCENT})
                    ])
                    ], color=PRIMARY, style={'height': 100, 'border-radius': 10}),  # , 'backgroundColor':'#FFFFFF', 'border':'1px solid'
    ])

# Text field


def createMetric():
    value_list = top_stats()
    title_list = ["Dollar Pnl", "Return%", "RealizedPL",
                  "UnrealizedPL", "Total_Val", "Max_DD", "Aval_Cash", "Win_Rate%"]
    metric_card_list = []
    for i in range(len(title_list)):
        item_card = dbc.Col([
            drawText(title_list[i], value_list[i])
        ])
        metric_card_list.append(item_card)
    return metric_card_list


def beautify_plotly(fig):
    return dcc.Graph(
        id='Graph_live',
        figure=fig,
        animate=True
    )


def show_dataplots(df):
    return html.Div([
                    dash_table.DataTable(

                        id='live-update-table',
                        # columns=[{"name": i, "id": i} for i in df.columns],
                        data=df.to_dict("rows"),
                        # dataframe.to_dict('records'), [{"name": i, "id": i} for i in dataframe.columns],

                        # style_cell_conditional=[{'if': {'column_id': c},'textAlign': 'left'} for c in ['Date', 'Region']],
                        # style_data=	{ 'color': 'black','backgroundColor': 'white'},
                        style_data_conditional=[
                            {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(220, 220, 220)'}],
                        style_header={
                            'backgroundColor': 'rgb(210, 210, 210)', 'color': 'black', 'fontWeight': 'bold'}
                    )
                    ])


def run_Paper_Strategy(strategy):
    print("Papar Strategy Running + {}".format(strategy))
    pass


def calculateMetricsAndCharts():
    status = generate_datasheet_for_unrealised_realised_profits()
    print("Performing Calculatons")
    pass


def cronJob():
    scheduler = BackgroundScheduler()
    scheduler.start()

    trigger = CronTrigger(
        year="*", month="*", day="*", hour="*", minute="*", second="*/5"
    )
    scheduler.add_job(
        calculateMetricsAndCharts,
        trigger=trigger,
        args=[],
        name="calculateMetricsAndCharts",
    )
    while True:
        sleep(60)


def register_callbacks(app):

    @app.callback(Output('output_container', 'children'),
                  [Input('run-paper-btn', 'n_clicks')],
                  [State('run-paper-strategy', 'value')],
                  loading_state={'is_loading': False})
    def runPaperStrategy(n_clicks, strategy):
        if n_clicks > 0:
            paper_trade_thread = Thread(
                target=run_Paper_Strategy, args=["MyStrategy1"])
            charts_and_sheet_calculation_thread = Thread(
                target=cronJob, args=[])
            paper_trade_thread.start()
            charts_and_sheet_calculation_thread.start()

    @app.callback(Output('Graph_live', 'figure'),
                  Output('close-live-update-table', 'data'),
                  Output('open-live-update-table', 'data'),
                  Output('metric_row', 'children'),
                  [Input('interval-component', 'n_intervals')],
                  loading_state={'is_loading': False})
    def update_graph(n):
        # load the latest data and update the graph
        # (code to load data and create the figure goes here) Output('close-live-update-table', 'data'), Output('open_positions_sheet', 'children'),
        status = generate_datasheet_for_unrealised_realised_profits()
        # global Dollar_Pnl, Return, Realized_PL, Unrealized_PL, Total_Val, Max_DD, Aval_Cash, Win_Rate
        # Dollar_Pnl, Return, Realized_PL, Unrealized_PL, Total_Val, Max_DD, Aval_Cash, Win_Rate = top_stats()
        updatedClose = close_Orders_Sheets()
        updatedOpen = open_Orders_Sheets()
        pnlCharts = performance_pnl_vector()
        updatedMetric = createMetric()
        return pnlCharts, updatedClose, updatedOpen, updatedMetric

    @app.callback(Output('output-message', 'children'),
                  [Input('deploy-cloud-btn', 'n_clicks')],
                  [State('run-paper-strategy', 'value')])
    def deploy_strategy_to_cloud(n_clicks, strategy):
        if n_clicks != 0:
            AUTH_URL = "https://quanturf.com/api/auth/"  # API for authentication
              # API for uploading strategy and related files
            FILES_URL = "https://quanturf.com/api/files/"
            strategy_file = strategy+".py"
            path = "MyLiveStrategies/"+strategy_file
            files = []
            files.append(('file', open(path, "rb")))
            password = "YXpLBqNzp2B79EG"
            username = "skumwt"
            user_auth = {'username': username, 'password': password}
            user_auth = json.dumps(user_auth)
            headers = {'Content-type': 'application/json'}
            message1 = ""
            try:
                    auth_request = requests.post(
						url=AUTH_URL, data=user_auth, headers=headers)
                    auth_response = auth_request.json()
                    message1 = auth_response['message']
            except Exception:
                    message1 = "Not Authorized"
            message2 = ""
            try:
                    file_upload_request = requests.post(
                        url=FILES_URL, files=files)
                    # print(file_upload_request.status_code)
                    file_upload_response = file_upload_request.json()
                    message2 = file_upload_response['message']
            except Exception:
                    message2 = "Deployment Failed"
            return message1 + message2
        return ""

