# import alpaca_backtrader_api
# import backtrader as bt
# from datetime import datetime

# class SmaCross1(bt.Strategy):
#     def notify_fund(self, cash, value, fundvalue, shares):
#         super().notify_fund(cash, value, fundvalue, shares)

#     def notify_store(self, msg, *args, **kwargs):
#         super().notify_store(msg, *args, **kwargs)
#         self.log(msg)

#     def notify_data(self, data, status, *args, **kwargs):
#         super().notify_data(data, status, *args, **kwargs)
#         print('*' * 5, 'DATA NOTIF:', data._getstatusname(status), *args)
#         if data._getstatusname(status) == "LIVE":
#             self.live_bars = True

#     # list of parameters which are configurable for the strategy
#     params = dict(
#         pfast=10,  # period for the fast moving average
#         pslow=30   # period for the slow moving average
#     )

#     def log(self, txt, dt=None):
#         dt = dt or self.data.datetime[0]
#         dt = bt.num2date(dt)
#         print('%s, %s' % (dt.isoformat(), txt))

#     def notify_trade(self, trade):
#         self.log("placing trade for {}. target size: {}".format(
#             trade.getdataname(),
#             trade.size))

#     def notify_order(self, order):
#         print(order)
#         print(f"Order notification. status {order.getstatusname()}.")
#         print(f"Order info. status {order.info}.")
#         #print(f'Order - {order.getordername()} {order.ordtypename()} {order.getstatusname()} for {order.size} shares @ ${order.price:.2f}')

#     def stop(self):
#         print('==================================================')
#         print('Starting Value - %.2f' % self.broker.startingcash)
#         print('Ending   Value - %.2f' % self.broker.getvalue())
#         print('==================================================')

#     def __init__(self):
#         self.live_bars = False
#         sma1 = bt.ind.SMA(self.data0, period=self.p.pfast)
#         sma2 = bt.ind.SMA(self.data0, period=self.p.pslow)
#         self.crossover0 = bt.ind.CrossOver(sma1, sma2)

#     def next(self):
#         #self.buy(data=data0, size=2)
#         if not self.live_bars and not IS_BACKTEST:
#             # only run code if we have live bars (today's bars).
#             # ignore if we are backtesting
#             return
#         # if fast crosses slow to the upside
#         if not self.positionsbyname[symbol].size and self.crossover0 > 0:
#             self.buy(data=data0, size=5)  # enter long

#         # in the market & cross to the downside
#         if self.positionsbyname[symbol].size and self.crossover0 <= 0:
#             self.close(data=data0)  # close long position