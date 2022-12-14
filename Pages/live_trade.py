
import os
import json
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

import warnings
warnings.filterwarnings('ignore')
# import tempfile
# import zipfile

# Customized Bullet chart
import datetime as dt

import plotly.express as px
from plotly.tools import mpl_to_plotly
import dash.dependencies
import pyfolio as pf
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import empyrical
#import quantstats as qs
#from quantstats import stats
from pandas_datareader import data as web
from plotly.subplots import make_subplots


# Raw Package
import numpy as np
import pandas as pd
# from pandas_datareader import data as pdr

# Alpaca Api
import alpaca_trade_api as tradeapi

#Graphing/Visualization
import plotly.graph_objs as go

# global yf_data
# yf_data = pd.DataFrame()
df_dict = {}

all_paper_strategy_files = os.listdir("MyLiveStrategies") 
stategy_list = list(filter(lambda f: f.endswith('.py'), all_paper_strategy_files))
list_select_strategy = [s.rsplit( ".", 1 )[ 0 ] for s in stategy_list]

#Styles

PRIMARY = '#FFFFFF' 
SECONDARY = '#FFFFFF'
ACCENT = '#EF5700'
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

def make_layout():

	genrate_datasheet_for_unrealised_realised_profits, top_stats, cumulative_returns_plot, annual_monthly_returns_plot, rolling_sharpe_plot= key_metrics()
	kurtosis,Return,Realized_PL,Unrealized_PL,Total_Val,Max_DD,Aval_Cash,Win_Rate = top_stats
	status = genrate_datasheet_for_unrealised_realised_profits
	#print(status)
	return html.Div([
		dbc.Col([					
					dbc.Card([
						dbc.CardHeader('Select Strategy', style={'color': DARK_ACCENT}),
						dbc.CardBody([
								# Run backtest
								html.Div([
									dcc.Dropdown(
										id='backtest-strategy', 
										options=list_select_strategy,
									)
								]),

								html.Br(),
								dbc.Row([
									#html.Br(),
										dbc.Col([
											html.Button('Live Trade', id='backtest-btn', className='eight columns u-pull-right', n_clicks=0, style={'font-size': '15px', 'font-weight': '5', 'color': '#FAF18F', 'background-color': '#242324', "border-color":'#242324', 'border-radius': 5}),
										]),
									html.Br(),

									dbc.Col([
										html.Button('Cloud Deploy', id='backtest-btn', className='eight columns u-pull-right', n_clicks=0, style={'font-size': '15px', 'font-weight': '5', 'color': '#FAF18F', 'background-color': '#242324', "border-color":'#242324', 'border-radius': 5}),
									]),
								]),
								html.Div(id='intermediate-value', style={'display': 'none'}),
								html.Div(id='intermediate-params', style={'display': 'none'}),
								html.Div(id='code-generated', style={'display': 'none'}),
								html.Div(id='code-generated2', style={'display': 'none'}),
								# dcc.Download(id="download-data-csv"),
								html.Div(id='intermediate-status', style={'display': 'none'}),
								html.Div(id='level-log', contentEditable='True', style={'display': 'none'}),
								dcc.Input(id='log-uid', type='text', style={'display': 'none'})
							])
					], color = PRIMARY, style ={'border-radius': 10}),
						], width=3),
		dbc.Card(
			dbc.CardBody([
				dbc.Row([
					dbc.Col([
						drawText('Dollar PnL', kurtosis)
					]),
					dbc.Col([
						drawText('Return%', Return)
					]),

					dbc.Col([
						drawText('Realized PL', Realized_PL)
					]),
					dbc.Col([
						drawText('Unrealized PL', Unrealized_PL)
					]),
					dbc.Col([
						drawText('Total Val.', Total_Val)
					]),
					dbc.Col([
						drawText('Max DD', Max_DD)
					]),
					dbc.Col([
						drawText('Aval. Cash', Aval_Cash)
					]),
					dbc.Col([
						drawText('Win Rate%', Win_Rate)
					]),
					# dbc.Col([
					# 	drawText('Outlier loss Ratio', outlier_loss_ratio)
					# ]),
					
				]), 
				html.Br(),
				dbc.Row([
					# dbc.Col([
					# 	eps_trend(symbol),
					# 	eps_revisions(symbol)
					# ], width=3),
					dbc.Col([
						dbc.Tabs([
						    dbc.Tab(cumulative_returns_plot, label='Performance'),
						    dbc.Tab(annual_monthly_returns_plot, label='Closed Position'),
						    dbc.Tab(rolling_sharpe_plot, label='Open Positions')
					        ],id='tabs',),
					], width=9)
				], align='center'), 
				html.Br(),
			]), color = PRIMARY, style ={'border-radius': 10} # all cell border
		)
	], style={'margin-bottom':'30rem'})


# helper function for closing temporary files
def close_tmp_file(tf):
    try:
        os.unlink(tf.name)
        tf.close()
    except:
        pass

class Alpaca_order:
  def __init__(self, order_id, price, cum_qty, side, symbol, qty, transaction_time,order_status):
    self.order_id = order_id
    self.price = price
    self.cum_qty = cum_qty
    self.side = side
    self.symbol = symbol
    self.qty = qty
    self.transaction_time = transaction_time
    self.order_status = order_status

def key_metrics():

	def getAlpcaClient():
		with open('alpaca_input_values.json') as infile:
			data = json.load(infile)
		APCA_API_KEY_ID = data['ALPACA_KEY']#"PKWW7CAGNXC9BD8C1UEW"
		APCA_API_SECRET_ID = data['ALPACA_SECRET']
		BASE_URL = "https://paper-api.alpaca.markets"
		client = tradeapi.REST(key_id = APCA_API_KEY_ID, secret_key = APCA_API_SECRET_ID, base_url = BASE_URL,api_version="v2")
		return client
	
	def get_orderids():
		datasheet_dir = "Datasheets/"
		Orderid_df =  pd.read_csv(os.path.join(datasheet_dir, "orderids.csv"), parse_dates=True)
		orderid_list = Orderid_df['OrderID'].tolist()
		return orderid_list
	
	def get_transactions_details_of_given_orderids_for_a_strategy(api, order_ids):
		result = api.get_activities()

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
			if (res.order_id in order_ids):
				order_id_list.append(res.order_id)
				qty_list.append(res.cum_qty)
				price_list.append(res.price)
				symbol_list.append(res.symbol)
				transaction_time_list.append(res.transaction_time)
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

		result_df = get_transactions_details_of_given_orderids_for_a_strategy(api, order_ids)

		df_buy = result_df[result_df.Type == 'buy']

		df_sell = result_df[result_df.Type == 'sell']
		
		
		df = pd.merge(df_buy, df_sell, on = 'Symbol', how = 'right', suffixes = ['_buy','_sell'])
		#All the symbols which have only buy go to the unrealized list --- Those who have sell may belong to realised and un_realised profit. 
		df_unrealized = result_df[~result_df['Symbol'].isin(df_sell.Symbol.unique())]
		#print(df_unrealized)

		testing = df_sell.sort_values(by = 'Transaction_time') #All sell order sorted by transection time.

		#testing['Price'] = testing.Price.astype('float') #Convert to Float Type. 
		#df_buy['Price'] = df_buy.Price.astype('float')   #Convert to Float Type.
		convert_dict = {'Price': float}

		testing = testing.astype(convert_dict)
		df_buy = df_buy.astype(convert_dict)

		# output_frame = pd.DataFrame(columns = ['sell_order_id','Symbol', 'selling_qty','Avg_selling_Price','Avg_buying_cost','Avg_holding_period',
		#                              'Earliest_buy_time','Latest_buy_time','Sell_time','Profit_per_unit','Total Profit', 'Winning_bet?'])
		
		output_frame = pd.DataFrame(columns = ['Symbol', 'selling_qty','Avg_selling_Price','Avg_buying_cost','Avg_holding_period_days','Sell_time',
									'Profit_per_unit','Total Profit', 'Winning_bet?'])

		for sym in testing.Symbol.unique():
			#print(sym)
			buy = df_buy.loc[df_buy.Symbol == sym] #all buy order with symbol sym
			#print("This is buy")
			#print(buy)
			#All rows with sell
			sell = testing.loc[testing.Symbol == sym] #all sell order with symbol sym sorted by transection time.
			#print("This is sell")
			#print(sell)
			
			obs = [] # completed sell\'s index
			for i, row in sell.iterrows(): #iterating for every i = 0,1,2... and row as pandas series.
				output_dic = {}
				if i not in obs:
					out = buy.loc[(buy.Transaction_time < row.Transaction_time)] #get all buy order that are made before a sell order (row) is made.
					#print("This is out")
					#print(out)
					idx = [j for j in out.index if j not in obs]
					#print("index operation list",idx)
					out = out.loc[idx] #These orders are not yet used to calculate realised profit
					#print("this is out after indexing removal ")
					#print(out)
					#print("Printing shape of out.shape[0] ")
					#print(out.shape[0])
					#assert out.shape[0] == int(row.Qty) #Match the quantity => this is not a good assert here we need to check if the total quantity bought is >= total quantity sold. 
					
					#[buy, buy, buy, buy, sell, buy, buy, buy, sell, sell, buy]
					#[2,   2,    2,   2,   4,    2,   2,  2,    5,     3,   2]

					# Avg_cost
					#print(row.Price - out.groupby('Symbol').Price.mean())
					#print(row.Transaction_time - out.groupby('Symbol').Transaction_time.mean())
					output_dict = {
									# 'sell_order_id':row.order_id,
								'Symbol': sym, 
								'selling_qty': int(row.Qty), #Quantity of the sold stocks,
								'Avg_selling_Price': row.Price,
								'Avg_buying_cost': round(out.groupby('Symbol').Price.mean()[0],2), #Average Price before this sell is done.  
								'Avg_holding_period_days': (row.Transaction_time - out.groupby('Symbol').Transaction_time.mean()[0]),
								#    'Earliest_buy_time': out.groupby('Symbol').Transaction_time.min()[0],
								#    'Latest_buy_time': out.groupby('Symbol').Transaction_time.max()[0],
								'Sell_time': row.Transaction_time,
								'Profit_per_unit': round(row.Price - out.groupby('Symbol').Price.mean()[0],2),
								'Total Profit': round((row.Price - out.groupby('Symbol').Price.mean())[0] * int(row.Qty),2),
								'Winning_bet?': True if round(row.Price - out.groupby('Symbol').Price.mean()[0],2) > 0 else False}
					output_frame = output_frame.append(output_dict, ignore_index = True)
					
					if len(idx) > 1:
						for ix in idx:
							obs.append(ix)
					else:
						obs.append(idx[0])
				

		# output_frame = output_frame.sort_values('Sell_time', ascending = False)
		output_frame['Avg_holding_period_days'] = output_frame['Avg_holding_period_days'].apply(lambda x: x.days)

		return output_frame

		
	def get_all_open_transactions_unrealised_profit(api, order_ids):
		result = api.get_activities()

		result_df = pd.DataFrame()

		#print(result[0]) # cum_qty, price, symbol, transaction_time
		# List = [latest transection  ...................................... oldest transectiomn]
		#Sample Activity
		# AccountActivity({   
		#     'activity_type': 'FILL',
		#     'cum_qty': '5',
		#     'id': '20221121093007186::d49b832e-5d1b-400a-98d4-05b5cf4da0fa',
		#     'leaves_qty': '0',
		#     'order_id': '16c45099-b77c-449b-9119-2301e1686eef',
		#     'order_status': 'filled',
		#     'price': '94.11',
		#     'qty': '5',
		#     'side': 'buy',
		#     'symbol': 'AMZN',
		#     'transaction_time': '2022-11-21T14:30:07.186233Z',
		#     'type': 'fill'
		#     })

		result_df = pd.DataFrame(columns = ['Order_Id','Symbol', 'Qty','Price', 'Unrealised_Profit_Per_Unit', 'Total_Unrealised_Profit','Transaction_Time'])

		sell_order_list = []
		buy_order_list = []

		for res in result:
			if (res.order_id in order_ids):
				if res.side == "buy":
					buy_order_list.append(Alpaca_order(res.order_id, float(res.price), int(res.cum_qty), res.side, res.symbol, res.qty, res.transaction_time, res.order_status))
				elif res.side == "sell":
					sell_order_list.append(Alpaca_order(res.order_id, float(res.price), int(res.cum_qty), res.side, res.symbol, res.qty, res.transaction_time, res.order_status))

		for sell_order in reversed(sell_order_list):
			curr_sell_order_symbol = sell_order.symbol
			curr_sell_order_qty = sell_order.cum_qty
			curr_sell_order_transaction_time = sell_order.transaction_time

			buy_order_index_that_are_closed = []
			for index, buy_order in reversed(list(enumerate(buy_order_list))):
				curr_buy_order_qty = buy_order.cum_qty
				if curr_sell_order_qty == 0:
					break
				if buy_order.symbol == curr_sell_order_symbol and buy_order.transaction_time < curr_sell_order_transaction_time:
					if curr_buy_order_qty <= curr_sell_order_qty:
						curr_sell_order_qty = curr_sell_order_qty - curr_buy_order_qty
						buy_order_index_that_are_closed.append(index)
					elif curr_buy_order_qty > curr_sell_order_qty:
						buy_order.cum_qty = buy_order.cum_qty - curr_sell_order_qty
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
		current_price = 98.24 #Getting exact price is not possible as alpaca does not show data for last 15minutes.

		account_positions = api.list_positions()
		current_price_dict = {}
		for position in account_positions:
			current_price_dict[position.symbol] = float(position.current_price)

		for res in buy_order_list:
			current_price = current_price_dict[(res.symbol).replace("/", "")]
			order_id_list.append(res.order_id)
			qty_list.append(res.cum_qty)
			price_list.append(res.price)
			symbol_list.append(res.symbol)
			transaction_time_list.append(res.transaction_time)
			unrealised_profit_per_unit.append(round(current_price-res.price, 2))
			total_unrealised_profit.append(round(qty_list[-1]*unrealised_profit_per_unit[-1],2))


		result_df = pd.DataFrame({
			'Order_Id':order_id_list,
			'Symbol': symbol_list,
			'Qty': qty_list, 
			'Price': price_list,
			'Unrealised_Profit_Per_Unit': unrealised_profit_per_unit,
			'Total_Unrealised_Profit': total_unrealised_profit, 
			'Transaction_Time': transaction_time_list})

		return result_df
	
	def get_pnl_df_strategy(open_positions, close_positions):
		#Open
		df_copy_open = open_positions.copy()
		df_copy_open['date'] = df_copy_open['Transaction_Time'].dt.date
		unrealized_pnl_df = df_copy_open.groupby('date')['Total_Unrealised_Profit'].sum().reset_index()

		#Close
		df_copy_close = close_positions.copy()
		df_copy_close['date'] = df_copy_close['Sell_time'].dt.date
		realized_pnl_df = df_copy_close.groupby('date')['Total Profit'].sum().reset_index()

		df = pd.merge(unrealized_pnl_df, realized_pnl_df, on='date', how='outer')
		df = df.rename(columns={'Total Profit': 'realized_pnl', 'Total_Unrealised_Profit': 'unrealized_pnl'})
		df = df.fillna(0)
		df['total_pnl'] = df['realized_pnl'] + df['unrealized_pnl']
		return df

	def genrate_datasheet_for_unrealised_realised_profits():
		client = getAlpcaClient()
		order_id_list = get_orderids()
		close_df = realized_profit_df_strategy(client, order_id_list)
		open_df = get_all_open_transactions_unrealised_profit(client, order_id_list)
		pnl_df = get_pnl_df_strategy(open_df, close_df)
		close_df.to_csv("Datasheets/close.csv", index=False)
		open_df.to_csv("Datasheets/open.csv", index=False)
		pnl_df.to_csv("Datasheets/pnl_df.csv", index=False)
		return "Yes"

	def top_stats():
		client = getAlpcaClient()
		accountInfo = client.get_account()
		datasheet_dir = "Datasheets/"
		pnldf =  pd.read_csv(os.path.join(datasheet_dir, "pnl_df.csv"), parse_dates=True)
		closedf = pd.read_csv(os.path.join(datasheet_dir, "close.csv"), parse_dates=True)
		opendf = pd.read_csv(os.path.join(datasheet_dir, "open.csv"), parse_dates=True)
		Realized_PL = pnldf['realized_pnl'].sum(skipna=True)
		Unrealized_PL = pnldf['unrealized_pnl'].sum(skipna=True)
		kurtosis = Realized_PL+Unrealized_PL
		InitialCash = 100000
		Total_Val = float(accountInfo.equity)
		Max_DD = pnldf['total_pnl'].min(skipna=True)
		Aval_Cash = float(accountInfo.cash)
		Return = (Total_Val - InitialCash)*100/InitialCash
		yes_count = closedf['Winning_bet?'].value_counts()[True]
		total_count = len(closedf['Winning_bet?'])
		Win_Rate = (yes_count / total_count) * 100
		return [
			kurtosis,
			Return,
			Realized_PL,
			Unrealized_PL,
			Total_Val,
			Max_DD,
			Aval_Cash,
			Win_Rate
		]

	def cumulative_returns_plot():

		datasheet_dir = "Datasheets/"
		pnldf =  pd.read_csv(os.path.join(datasheet_dir, "pnl_df.csv"), parse_dates=True)
		fig = go.Figure([go.Scatter(x=pnldf['date'], y=pnldf['total_pnl'])])
		return beautify_plotly(fig)
		
	
	def annual_monthly_returns_plot():
		datasheet_dir = "Datasheets/"
		closedf =  pd.read_csv(os.path.join(datasheet_dir, "close.csv"), parse_dates=True)
		return show_dataplots(closedf)
	
	def rolling_sharpe_plot():
		datasheet_dir = "Datasheets/"
		opendf =  pd.read_csv(os.path.join(datasheet_dir, "open.csv"), parse_dates=True)
		return show_dataplots(opendf)
	
	return  genrate_datasheet_for_unrealised_realised_profits(), top_stats(), cumulative_returns_plot(), annual_monthly_returns_plot(), rolling_sharpe_plot()	



# Text field
def drawText(title, text):
	return html.Div([
		dbc.Card([
			dbc.CardHeader(title, style={'color': DARK_ACCENT}), 
			dbc.CardBody([
				html.Div([
					# html.Header(title, style={'color': 'white', 'fontSize': 15, 'text-decoration': 'underline', 'textAlign': 'left'}),
					# html.Br(),
					html.Div(str(round(text, 2)), style={'color': DARK_ACCENT, 'textAlign': 'left'}),
					# str(round(text, 2))
				], style={'color': DARK_ACCENT}) 
			])
		], color=PRIMARY, style={'height': 100, 'border-radius': 10}), # , 'backgroundColor':'#FFFFFF', 'border':'1px solid'
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
				]), color = SECONDARY, style ={'border-radius': 10}
			),  
		])



def show_dataplots(dataframe):
	return html.Div([
			dash_table.DataTable(dataframe.to_dict('records'), [{"name": i, "id": i} for i in dataframe.columns],
			style_cell_conditional=[
        								{
            								'if': {'column_id': c},
           									'textAlign': 'left'
        								} for c in ['Date', 'Region']
   								     ],
			style_data=					{
        									'color': 'black',
        									'backgroundColor': 'white'
    									},
			style_data_conditional=[
        								{
            								'if': {'row_index': 'odd'},
            								'backgroundColor': 'rgb(220, 220, 220)',
        								}
    								],
			style_header=				{
        									'backgroundColor': 'rgb(210, 210, 210)',
        									'color': 'black',
        									'fontWeight': 'bold'
    									}
			)
		])

# def register_callbacks(app):

#     @app.callback(Output('dd-output-container', 'children'), [Input('trade-type-dropdown', 'value')])
#     def updateTradeType(input_value):
#         if input_value=="Paper":
#             Paper = True
#         elif input_value == "Live":
#             Live = True
#         return f'You have Selected: {input_value}'
    
#     @app.callback(Output('key-output-container', 'children'), [Input('example-key-row', 'value')])
#     def updateAlpacaKey(input_key):
#         USER_ALPACA_KEY = input_key
    
#     @app.callback(Output('secret-output-container', 'children'), [Input('example-secret-row', 'value')])
#     def updateSecretKey(input_secret):
#         USER_ALPACA_SECRET = input_secret