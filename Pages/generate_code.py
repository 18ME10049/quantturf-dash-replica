import os
import json

from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from datetime import timedelta, datetime, date

import yfinance as yf
import pandas as pd

#Alpaca package
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame

with open('G:\Quanturf\quantturf-dash-replica\Alpaca_input_values.json') as infile:
    data = json.load(infile)

APCA_API_KEY_ID = data['ALPACA_KEY']#"PKWW7CAGNXC9BD8C1UEW"
APCA_API_SECRET_ID = data['ALPACA_SECRET']
BASE_URL = "https://paper-api.alpaca.markets"


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

company_list = symbolList #pd.read_csv('Static/sp500_companies.csv')['Symbol'].to_list()
frequencyList = ['Days', 'Ticks', 'MicroSeconds', 'Seconds', 'Minutes', 'Weeks', 'Months', 'Years', 'NoTimeFrame']

PRIMARY = '#FFFFFF'
SECONDARY = '#FFFFFF'
ACCENT = '#98C1D9'
SIDEBAR = '#F7F7F7'
DARK_ACCENT = '#474747'

today = date.today()
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

paper_live_code_generate = dbc.Col([
						html.Br(),
						dbc.Card(
							[
								dbc.CardHeader('Generate Algorithm Code - Backtest/Paper Trade', style={'color': DARK_ACCENT}),
								dbc.CardBody([
								# Generate code
								# html.Div([
								# 	html.Div('Algos:', className='four columns'),
								# 	dcc.Dropdown(id='module-gc', options=[], className='eight columns u-pull-right')
								# 	# dcc.Dropdown(
								# 	#     id='module',
								# 	#     options=[{'label': name, 'value': name} for name in oc.cfg['backtest']['modules'].split(',')],
								# 	#     className='eight columns u-pull-right')
								# ], className='row mb-10'),
								html.Div([
									'Enter Capital Value:',
									dcc.Input(id='cash', className='eight columns u-pull-right', value = 10000, style={'margin-left': '10px', 'width': '210px', 'font-size': '15px', 'font-weight': '5', 'border-radius': 5})
								], className='row mb-10'),
								html.Br(),
								html.Div([
									'Enter  Frequency:',
									dcc.Dropdown(frequencyList, value = 'Days', id='freqeuncy-selected-property', clearable=False, style={'width': '210px', 'border-radius': 5})
								], className='row mb-10'),
								html.Br(),
								html.Div([
									'Strategy Name:',
									#dcc.Dropdown(id='strategy', options=[], className='eight columns u-pull-right')
									dcc.Input(id='filename', className='eight columns u-pull-right', value = "MyStrategy", style={'margin-left': '10px', 'width': '210px', 'font-size': '15px', 'font-weight': '5', 'border-radius': 5})
								], className='row mb-10'),

								html.Br(),
								html.Button('Generate Code', id='Generate-Live-Code', n_clicks=0, className='eight columns u-pull-right', style={'font-size': '15px', 'font-weight': '7px', 'color': '#FAF18F', 'background-color': '#242324', "border-color":'#242324', 'border-radius': 5}),
							]),], color=PRIMARY, style={'border-radius': 10}
						),
						html.Div(id='code-generated-gc', style={'display': 'none'}),
						html.Div(id='code-generated2-gc', style={'display': 'none'}),
					], width=4)


backtest_code_generate = dbc.Col([
						html.Br(),
						dbc.Card(
							[
								dbc.CardHeader('Generate Algorithm Code - Backtest', style={'color': DARK_ACCENT}),
								dbc.Row([
									dbc.Col([
								dbc.CardBody([
								html.Div([
									dcc.Dropdown(frequencyList, value = 'Days', id='backtest-freqeuncy-selected-property', clearable=False, placeholder='Select Property...', style={ 'width': '210px', 'font-size': '15px', 'border-radius': 5})
								], className='row mb-10'),
								html.Br(),
								html.Div([
									html.Div('Capital:', className='four columns'),
									#dcc.Dropdown(id='strategy', options=[], className='eight columns u-pull-right')
									dcc.Input(id='cash', className='eight columns u-pull-right', value = 10000, style={'margin-left': '10px', 'width': '210px', 'font-size': '15px', 'font-weight': '5', 'border-radius': 5})
								], className='row mb-10'),
								html.Br(),
								html.Div([
										html.Div('Strategy Name:', className='four columns'),
										#dcc.Dropdown(id='strategy', options=[], className='eight columns u-pull-right')
										dcc.Input(id='filename', className='eight columns u-pull-right', value = "MyStrategy", style={'margin-left': '10px', 'width': '210px', 'font-size': '15px', 'font-weight': '5', 'border-radius': 5})
									], className='row mb-10'),
								html.Br(),
								html.Button('Generate Backtest Code', id='save-btn', n_clicks=0, className='eight columns u-pull-right', style={'font-size': '15px', 'font-weight': '5', 'color': PRIMARY, 'background-color': ACCENT, "border-color":ACCENT, 'border-radius': 5}),
							])]), dbc.Col([
								# dcc.DatePickerRange(
								# 		id='my-date-picker-range',
								# 		min_date_allowed=date(2000, 8, 5),
								# 		# max_date_allowed=date(2017, 9, 19),
								# 		start_date=previous_2,
								# 		# initial_visible_month=date(2022, 1, 1),
								# 		end_date=previous_1,
								# 		style={
								# 			'background-color': PRIMARY,
								# 			'color': 'black',
								# 			'zIndex': 100000
								# 		},
								# 		calendar_orientation='vertical',
								# 	)
							])])
							],color=PRIMARY, style={'border-radius': 10}),
						html.Div(id='code-generated-gc', style={'display': 'none'}),
						html.Div(id='code-generated2-gc', style={'display': 'none'}),
					], width=5)

ChooseAlgoCard = dbc.Col([dbc.Card([
							dbc.CardHeader('Choose Algorithm', style={'color': DARK_ACCENT}),
							dbc.CardBody([
								html.Div([
									dcc.Dropdown(id='module-gc', options=[], className='eight columns u-pull-right')
									# dcc.Dropdown(
									#     id='module',
									#     options=[{'label': name, 'value': name} for name in oc.cfg['backtest']['modules'].split(',')],
									#     className='eight columns u-pull-right')
								], className='row mb-10'),
							])
						], color=PRIMARY, style={'border-radius': 10, "width": "18rem"})], width=2)
						
ChooseEquityCard = dbc.Col([ dbc.Card([
							dbc.CardHeader('Choose Equity', style={'color': DARK_ACCENT}),
							dbc.CardBody([
								html.Div([
									dcc.Dropdown(
										value='AMZN',
										id='symbols',
										options=[{'label': name, 'value': name} for name in company_list],
										#options=['AAPL', 'TSLA', 'MSFT', 'AMZN'], #Replace this with list
										multi=True)
								],className='row mb-10'),
							])
						], color=PRIMARY, style={'border-radius': 10, "width": "18rem"})], width=2)

def make_layout():

	left_col = html.Div([
		dbc.Card(
			dbc.CardBody([
				html.Br(),
				dbc.Row([ChooseEquityCard, dbc.Col([html.Br()], width=1), ChooseAlgoCard], id='graph-container', style={'margin-bottom':'3rem'}),
				dbc.Row([
					paper_live_code_generate,
					#backtest_code_generate
				], id='graph-container', style={'margin-bottom':'30rem'})
			]),
		),
	])

	return left_col


def register_callbacks(app):
	# @app.server.route('{}<file>'.format(static_route))
	# def serve_file(file):
	# 	if file not in stylesheets and file not in jss:
	# 		raise Exception('"{}" is excluded from the allowed static css files'.format(file))
	# 	static_directory = os.path.join(root_directory, 'Static')
	# 	return flask.send_from_directory(static_directory, file)

    # update lago
	@app.callback(Output('module-gc', 'options'), [Input('symbols', 'value')])
	def update_algo_list(symbols):

		all_files = os.listdir("SampleStrategies") 
		algo_files = list(filter(lambda f: f.endswith('.py'), all_files))
		algo_avlb = [s.rsplit( ".", 1 )[ 0 ] for s in algo_files]
		#print(algo_avlb)    
		return algo_avlb

	# @app.callback(Output('strategy', 'options'), [Input('module', 'value')])
	# def update_strategy_list(module_name):
	#     data = ob.test_list(module_name)
	#     return [{'label': name, 'value': name} for name in data]

	@app.callback(Output('strategy-gc', 'options'), [Input('symbols', 'value')])
	def update_strategy_list(symbols):  
		print("strat called")
		all_files = os.listdir("MyStrategies")    
		backtest_files = list(filter(lambda f: f.endswith('.py'), all_files))
		backtest_avlb = [s.rsplit( ".", 1 )[ 0 ] for s in backtest_files]  
		#print(backtest_avlb) 
		return backtest_avlb

	# I think this callback is not needed. No html tag with id = 'params-table' is there
	# Commenting out it for now.

	# @app.callback(Output('params-table', 'columns'), [Input('module', 'value'), Input('strategy', 'value'), Input('symbols', 'value')])
	# def update_params_list(module_name, strategy_name, symbol):
	#     return ob.params_list(module_name, strategy_name, symbol)


	@app.callback(Output('strategy-gc', 'value'), [Input('strategy-gc', 'options')])
	def update_strategy_value(options):
		if len(options):
			#print(options)
			return options[0]
		return ''

	#Add code later to make sure that enter cash and symbols
	@app.callback(Output('code-generated-gc', 'children'),
				[
					Input('Generate-Live-Code', 'n_clicks')
				],
				[	State('symbols', 'value'),
					State('module-gc', 'value'),
					State('cash', 'value'),
					State('freqeuncy-selected-property','value'),
					State('filename', 'value'),
				])
	def create_code(n_clicks, symbol, algoName, cash, frequency, fileName):
		if n_clicks == 0:
			return '' 

		data = data2 = ""
		paperLiveCode =   f"""import alpaca_backtrader_api
import backtrader as bt
from datetime import datetime

# Your credentials here
ALPACA_API_KEY = "{APCA_API_KEY_ID}"
ALPACA_SECRET_KEY = "{APCA_API_SECRET_ID}"

IS_BACKTEST = False
IS_LIVE = False
symbol = "{symbol}"


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
            self.buy(data=data0, size=5)  # enter long

        # in the market & cross to the downside
        if self.positionsbyname[symbol].size and self.crossover0 <= 0:
            self.close(data=data0)  # close long position


if __name__ == '__main__':
    import logging
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaCross1)

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
    cerebro.adddata(data0)

    if IS_BACKTEST:
        # backtrader broker set initial simulated cash
        cerebro.broker.setcash(100000.0)

    print('Starting Portfolio Value: {{}}'.format(cerebro.broker.getvalue()))
    cerebro.run()
    print('Final Portfolio Value: {{}}'.format(cerebro.broker.getvalue()))
    cerebro.plot()"""
		#.format(APCA_API_KEY_ID = APCA_API_KEY_ID, APCA_API_SECRET_ID = APCA_API_SECRET_ID, live = live, symbol=symbol)
		strategy_file=algoName+".py"
		strategy_file = "SampleStrategies/"+strategy_file

		with open(strategy_file) as fp:
			data = fp.read()
		
		data += "\n"
		data = paperLiveCode
		path_dir = "MyLiveStrategies/"
		filename_save = fileName+".py"
		
		with open (os.path.join(path_dir, filename_save), 'w') as fp:
			fp.write(data)
		

		backTestCode =   f"""import alpaca_backtrader_api
import backtrader as bt
from datetime import datetime

# Your credentials here
ALPACA_API_KEY = "{APCA_API_KEY_ID}"
ALPACA_SECRET_KEY = "{APCA_API_SECRET_ID}"

IS_BACKTEST = True
IS_LIVE = False
symbol = "{symbol}"


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
            self.buy(data=data0, size=5)  # enter long

        # in the market & cross to the downside
        if self.positionsbyname[symbol].size and self.crossover0 <= 0:
            self.close(data=data0)  # close long position


if __name__ == '__main__':
    import logging
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaCross1)

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
    cerebro.adddata(data0)

    if IS_BACKTEST:
        # backtrader broker set initial simulated cash
        cerebro.broker.setcash({cash})

    print('Starting Portfolio Value: {{}}'.format(cerebro.broker.getvalue()))
    cerebro.run()
    print('Final Portfolio Value: {{}}'.format(cerebro.broker.getvalue()))
    cerebro.plot()"""
		#.format(APCA_API_KEY_ID = APCA_API_KEY_ID, APCA_API_SECRET_ID = APCA_API_SECRET_ID, live = live, symbol=symbol)
		strategy_file=algoName+".py"
		strategy_file = "SampleStrategies/"+strategy_file

		with open(strategy_file) as fp:
			data = fp.read()
		
		data += "\n"
		data = backTestCode
		path_dir = "MyBacktestStrategies/"
		filename_save = fileName+".py"
		
		with open (os.path.join(path_dir, filename_save), 'w') as fp:
			fp.write(data)

		return 0

		#####  Download Button #####

	# @app.callback(Output('code-generated2-gc', 'children'),
	# 			[
	# 				Input('download-btn', 'n_clicks'),
	# 				Input('symbols', 'value')
	# 			])
	# def download_data(n_clicks, symbols ):
	# 	if n_clicks == 0:
	# 		return '' 
	# 	#symbols = ['TSLA', 'GE']
	# 	print("testing Datas ") 
	# 	print(symbols)   
	# 	for s in symbols:
	# 			df = yf.download(s, start = "2018-01-01")
	# 			data_dir = "Data/"
	# 			filename = s +".csv"
	# 			df.to_csv(os.path.join(data_dir, filename)) 
	# 	#return dcc.send_data_frame(df.to_csv, filename) 
	# 	# module_name = "FromBackTrader"
	# 	# module = importlib.import_module(module_name)
	# 	# pnl, results = module.backtest()
	# 	#result = subprocess.getstatusoutput('python FromBackTrader.py' )  
	# 	return 0