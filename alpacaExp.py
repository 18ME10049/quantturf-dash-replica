#This file is only for testing out python not used in this app. 

import json
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
import pandas as pd
import os

import requests

# ALPACA_KEY = "PKWW7CAGNXC9BD8C1UEW"
# ALPACA_SECRET =  "iMz0aAFKlWrV4PqLKtUIFnJnyjtGthNDXQLoHckY"
# BASE_URL  =  "https://paper-api.alpaca.markets"

# #Use ALPACA Client
# api = tradeapi.REST(key_id = ALPACA_KEY, secret_key = ALPACA_SECRET, base_url = BASE_URL)
# # Get list of all the Symbols Available in Alpaca
# result = api.list_assets(status='active')
# result_df = pd.DataFrame(columns = ['class','exchange','symbol'])
# class_list = []
# exchange_list = []
# symbol_list = []

# for res in result:
#     #print(res.class)
#     #class_list.append(res.class)
#     exchange_list.append(res.exchange)
#     symbol_list.append(res.symbol)

# result_df = pd.DataFrame({ 'exchange': exchange_list, 'Symbol': symbol_list})
# symbolList = result_df.Symbol.unique().tolist()
# exchangeList = result_df.exchange.unique().tolist()

# # global yf_data
# # yf_data = pd.DataFrame()
# grouped = result_df.groupby('exchange')

# # create a dictionary with the categories as keys and the lists of values as values
# df_dict = grouped['Symbol'].apply(list).to_dict()
# keys = df_dict.keys()

# company_list = [1,2,3,4]

# options=[{'label': name, 'value': name} for name in company_list]
# print(options)

# print(keys)
# #print(df_dict['OTC'])
# print(exchangeList)


def deploy():
    AUTH_URL = "https://quanturf.com/api/auth/"  # API for authentication
    # API for uploading strategy and related files
    FILES_URL = "https://quanturf.com/api/files/"

    strategy = "MyStrategy"
    # upload all the files of the directory in which the CLI command is called.
    # path = os.getcwd()
    dir_list = os.listdir("MyLiveStrategies") 
    # dir_list = os.listdir(path)
    strategy_file=strategy+".py"
    path = "MyLiveStrategies/"+strategy_file
    files = []
    files.append(('file', open(path, "rb")))
    # for file in dir_list:
    #     if file == strategy:
    #         files.append(('file', open(path, "rb")))

    # taking quanturf username and password and then sending it to the quanturf server through the AUTH_URL API
    # print("Enter your Quanturf username and password...")

    # username=input("Enter Username: ")
    # password=maskpass.askpass(mask="*")
    password = "YXpLBqNzp2B79EG"
    username = "skumwt"
    user_auth = {'username': username, 'password': password}
    user_auth = json.dumps(user_auth)
    headers = {'Content-type': 'application/json'}
    auth_request = requests.post(url=AUTH_URL, data=user_auth, headers=headers)
    auth_response = auth_request.json()
    print(auth_response['message'])

    # sending the files to the quanturf server through the FILES_URL API
    file_upload_request = requests.post(url=FILES_URL, files=files)
    # print(file_upload_request.status_code)
    file_upload_response = file_upload_request.json()
    print(file_upload_response['message'])


deploy()