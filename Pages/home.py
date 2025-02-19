import warnings
warnings.filterwarnings('ignore')
import os
import tempfile
import zipfile
from datetime import timedelta, datetime, date
import json


# Customized Bullet chart
import datetime as dt
# import pandas_datareader.data as web
import plotly.express as px
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import plotly.express as px
from plotly.tools import mpl_to_plotly
import dash.dependencies
import pyfolio as pf
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
#import empyrical
#import quantstats as qs
#from quantstats import stats
from pandas_datareader import data as web
from plotly.subplots import make_subplots

# Raw Package
import numpy as np
import pandas as pd
# from pandas_datareader import data as pdr

# Market Data 
import yfinance as yf
import yahoo_fin.stock_info as si

#Alpaca package
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame

with open('alpaca_input_values.json') as infile:
    data = json.load(infile)

APCA_API_KEY_ID = data['ALPACA_KEY']#"PKWW7CAGNXC9BD8C1UEW"
APCA_API_SECRET_ID = data['ALPACA_SECRET']
BASE_URL = "https://paper-api.alpaca.markets"

#Graphing/Visualization
import plotly.graph_objs as go

from Pages import fx

#Use ALPACA Client
api = tradeapi.REST(key_id = APCA_API_KEY_ID, secret_key = APCA_API_SECRET_ID, base_url = BASE_URL)
# Get list of all the Symbols Available in Alpaca
result = api.list_assets(status='active')
result_df = pd.DataFrame(columns = ['class','exchange','symbol'])
class_list = []
exchange_list = []
symbol_list = []

for res in result:
    #print(res.class)
    #class_list.append(res.class)
    exchange_list.append(res.exchange)
    symbol_list.append(res.symbol)

result_df = pd.DataFrame({ 'exchange': exchange_list, 'Symbol': symbol_list})
symbolList = result_df.Symbol.unique().tolist()
exchangeList = result_df.exchange.unique().tolist()

grouped = result_df.groupby('exchange')

# create a dictionary with the categories as keys and the lists of values as values
df_dict = grouped['Symbol'].apply(list).to_dict()

asset_class = ''

# # get tickers
# sp_tickers = pd.read_csv('Static/Data/sp500_companies.csv', usecols=['Symbol'])
# sp_tickers = sp_tickers['Symbol'].values.tolist()


# crypto_tickers = pd.read_csv('Static/Data/crypto_tickers.csv', names=['Symbol'])
# crypto_tickers = crypto_tickers['Symbol'].values.tolist()

# fx_countries = pd.read_csv('Static/Data/Foreign_Exchange_Rates.csv')
# fx_countries = fx_countries.replace('ND', np.nan) 
# fx_countries = fx_countries.dropna()

# country_lst = list(fx_countries.columns[2:])

# equity_df = pd.DataFrame()

# get all companies from json file
# with open('Static/Dropdown Data/companies.json', 'r') as read_file:
# 	company_list = json.load(read_file)
# company_options_list = []
# for company in company_list:
#     company_options_list.append(company)
company_options_list = symbolList

# set asset specific drowdown values
tickers_dict = df_dict#{'Equities': company_options_list, 'Crypto': crypto_tickers, 'FX': country_lst, 'Fixed Income': [], 'Commodities': [], 'Sentiment': []}
asset_classes = list(tickers_dict.keys())
nested_options = tickers_dict[asset_classes[0]]

#asset_classes = ['Equities', 'Crypto', 'FX']
# properties = ['All', 'Open', 'High', 'Low', 'Close', 'Volume']
properties = ['Day', 'Minute', 'Hour']



today = date.today()

offset = max(1, (today.weekday() + 6) % 7 - 3)
timedlt1 = timedelta(offset)

most_recent = today - timedlt1

offset = max(1, (today.weekday() + 7) % 7 - 3)
timedlt1 = timedelta(offset)

yesterday = today - timedelta(1)

second_most_recent = today - timedlt1
previous_2 = today-timedelta(days=3)
previous_1 = today-timedelta(days=2)

weekday = today.weekday()

if weekday == 0:
	previous_2 = today-timedelta(days=4)
	previous_1 = today-timedelta(days=3)
if weekday == 1:
	previous_2 = today-timedelta(days=4)
	previous_1 = today-timedelta(days=1)
if weekday == 5:
	previous_2 = today-timedelta(days=2)
	previous_1 = today-timedelta(days=1)
if weekday == 6:
	previous_2 = today-timedelta(days=3)
	previous_1 = today-timedelta(days=2)


def make_layout():

	# if symbol in fx_countries:
	# 	return fx.make_layout(symbol)

	# if symbol is None:
	# 	symbol = 'AAPL'
		# app.equity_df.append(yf.download(tickers='AAPL',period='1d',interval='1m', group_by='ticker', auto_adjust = False, prepost = False, threads = True, proxy = None))

	return html.Div([
		dbc.Card(
			dbc.CardBody([
				html.Br(),
				dbc.Row([
					dbc.Col([
						html.Div([
							dbc.Card(
								dbc.CardBody([
									dcc.Graph(
										id='center-stock',
									config={
										'displayModeBar': False
									}
									)
								]), color = SECONDARY, style ={'border-radius': 10}
							),  
						])
					], width=10),
					dbc.Col([
						html.Div([
							dbc.Card([
								dbc.CardHeader('Customize', style={'color': DARK_ACCENT}),
								dbc.CardBody([
									dcc.DatePickerRange(
										id='my-date-picker-range',
										min_date_allowed=date(2000, 8, 5),
										# max_date_allowed=date(2017, 9, 19),
										start_date=previous_2,
										# initial_visible_month=date(2022, 1, 1),
										end_date=previous_1,
										style={
											'background-color': PRIMARY,
											'color': 'black',
											'zIndex': 100000
										},
										calendar_orientation='vertical',
									),
									html.Br(),
									html.Br(),
									dcc.Dropdown(asset_classes, id='selected-asset-class', style=SEARCH_STYLE, clearable=False, placeholder='Select Asset Class'),
									html.Br(),
									dcc.Dropdown(id='selected-symbol', style=SEARCH_STYLE, clearable=False, placeholder='Select Ticker'),
									html.Br(),
									dcc.Dropdown(properties, id='selected-property', style=SEARCH_STYLE, clearable=False, placeholder='Select Frequency'),
									html.Br(),
									dcc.Download(id="download-center-stock-csv"),
									dbc.Button('Download Data', id="center_stock", n_clicks=0, style={'background-color': '#242324', 'color': '#FAF18F', "border-color":'#242324'}),
								]),], color=PRIMARY, style={'border-radius': 10}
							),
						], style={'border-radius': 10})
					], width=2),
				]), 
				html.Br(),
	
			]), color = PRIMARY, style ={'border-radius': 10} # all cell border
		)
	], style={'margin-bottom':'30rem'})
		


PRIMARY = '#FFFFFF' 
SECONDARY = '#FFFFFF'
ACCENT = '#98C1D9'
DARK_ACCENT = '#474747'
SIDEBAR = '#F7F7F7'

# PRIMARY = '#15202b'
# SECONDARY = '#192734'
# ACCENT = '#FFFFFF'
# SIDEBAR = '#F4511E'
#F4511E

DATATABLE_STYLE = {
    'color': 'white',
    'backgroundColor': PRIMARY,
}

SEARCH_STYLE  = {
    'background-color': PRIMARY,
    'color': 'black',
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

# # add csv to download folder
# def add_csv_to_folder(df, name):
# 	filepath = Path('/finailab_dash/Static/download_folder/' + name + '.csv')
# 	filepath.parent.mkdir(parents=True, exist_ok=True)
# 	df.to_csv(filepath)

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
				]), color = SECONDARY, style ={'border-radius': 10}
			),  
		])


# create main equity plot
def centerStock(symbol, start, end, metric):

	from plotly.subplots import make_subplots
 
	# Override Yahoo Finance 
	yf.pdr_override()

	delta = dt.datetime.strptime(end, '%Y-%m-%d') - dt.datetime.strptime(start, '%Y-%m-%d')
	# if delta.days < 30:
	# 	# Retrieve stock data frame (df) from yfinance API at an interval of 1m 
	# 	df = yf.download(tickers=symbol,period='1d',interval='1m', start=start,end=end)
	# else:
	# 	df = yf.download(tickers=symbol,period='1d',start=start,end=end)
	timeFrame = TimeFrame.Day
	if metric == 'Minute':
		timeFrame = TimeFrame.Minute
	elif metric == 'Hour':
		timeFrame = TimeFrame.Hour
	df = api.get_bars( symbol=symbol , #any symbol is acceptable if it can be found in Alpaca API
    								timeframe=timeFrame, 
    								start=start,end=end).df
	df.drop(df.tail(1).index,inplace=True)
	# add_csv_to_folder(df, "center_stock")
	df_dict['Download Data'] = df
	# print(yf_data)
	# df = pd.DataFrame(yf_data[symbol])

	Close = 'close'
	Open = 'open'
	High = 'high'
	Low = 'low'

	# add Moving Averages (5day and 20day) to df 
	df['MA5'] = df[Close].rolling(window=5).mean()
	df['MA20'] = df[Close].rolling(window=20).mean()

	# print(df)

	# Declare plotly figure (go)
	fig=go.Figure()

	# Creating figure with second y-axis
	fig = make_subplots(specs=[[{'secondary_y': True}]])

	# Adding line plot with close prices and bar plot with trading volume
	# fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name=symbol+' Close'), secondary_y=False)

	# if metric == 'All': 
	fig.add_trace(go.Candlestick(x=df.index,
					open=df[Open],
					high=df[High],
					low=df[Low],
					close=df[Close], name = 'market data'))
	# elif metric == 'Open':
	# 	fig.add_trace(go.Scatter(x=df.index, 
	# 						y=df['Open'], 
	# 						opacity=0.7, 
	# 						line=dict(color='#98C1D9', width=2), 
	# 						name='Open'))
	# elif metric == 'High':
	# 	fig.add_trace(go.Scatter(x=df.index, 
	# 						y=df['High'], 
	# 						opacity=0.7, 
	# 						line=dict(color='#98C1D9', width=2), 
	# 						name='High'))
	# elif metric == 'Low':
	# 	fig.add_trace(go.Scatter(x=df.index, 
	# 						y=df['Low'], 
	# 						opacity=0.7, 
	# 						line=dict(color='#98C1D9', width=2), 
	# 						name='Low'))
	# elif metric == 'Close':
	# 	fig.add_trace(go.Scatter(x=df.index, 
	# 						y=df['Close'], 
	# 						opacity=0.7, 
	# 						line=dict(color='#98C1D9', width=2), 
	# 						name='Close'))
	# elif metric == 'Volume':
	# 	fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', opacity=0.5, marker_color=['black'], marker_colorscale='Rainbow',), secondary_y=True)

	# Add 5-day Moving Average Trace
	fig.add_trace(go.Scatter(x=df.index, 
							y=df['MA5'], 
							opacity=0.7, 
							line=dict(color='brown', width=2), 
							name='MA 5'))
	# Add 20-day Moving Average Trace
	fig.add_trace(go.Scatter(x=df.index, 
							y=df['MA20'], 
							opacity=0.7, 
							line=dict(color='orange', width=2), 
							name='MA 20'))

	fig.update_xaxes(
		# rangeslider_visible=True,
		rangeselector=dict(
			buttons=list([
				dict(count=15, label='15m', step='minute', stepmode='backward'),
				dict(count=45, label='45m', step='minute', stepmode='backward'),
				dict(count=1, label='HTD', step='hour', stepmode='todate'),
				dict(count=3, label='3h', step='hour', stepmode='backward'),
				dict(step='all')
			]), bgcolor = SECONDARY
		),
		nticks=delta.days * 4
	)
	fig.layout.xaxis.type = 'category'
	

	# Updating layout
	fig.update_layout(
		xaxis_rangeslider_visible=True,
		hovermode='x'
	)

	fig.update_layout(
		title= str(symbol)+' Live Share Price:',
		yaxis_title='Stock Price (USD per Shares)',
		# template='plotly_dark',
		plot_bgcolor= SECONDARY,
		paper_bgcolor= SECONDARY,   
		font=dict(color=DARK_ACCENT),
	)



	return fig

def register_callbacks(app):
	@app.callback(
		Output("download-center-stock-csv", "data"),
		[Input("center_stock", "n_clicks"), Input("center_stock", "children")]
	)
	def func(n_clicks, name):
		df = df_dict[name]
		return dcc.send_data_frame(df.to_csv, "finailab_data.csv")

	@app.callback(Output('center-stock', 'figure'), [Input('selected-property', 'value')], [State('selected-symbol', 'value'),State('my-date-picker-range', 'start_date'),
		State('my-date-picker-range', 'end_date'), ])
	def send_to_graph(metric, symbol, start, end):
		return centerStock(symbol, start, end, metric)

	# adjust dropdown tickers for a given tab
	@app.callback(Output('selected-symbol', 'options'),
				[Input('selected-asset-class', 'value')], initial_callbacks=True)
	def update_dropdown(asset_class):
		if asset_class is not None:
			return list(df_dict[asset_class])
		# if asset_class == 'Equities':
		# 	return tickers_dict[asset_class]
		# else:
		# 	return [{'label': i, 'value': i} for i in tickers_dict[asset_class]]




	# @app.callback(Output('financials', 'children'), Input('financials-tabs', 'value'), Input('selected-symbol', 'value')
	# )
	# def render_financials(tab, symbol):

	# 	if symbol is None:
	# 		symbol = 'AAPL'

	# 	if tab == 'balance-sheet':
	# 		return balance_sheet(symbol)
	# 	elif tab == 'income-statement':
	# 		return income_statement(symbol)
	# 	elif tab == 'cash-flows':
	# 		return cash_flows(symbol)

	