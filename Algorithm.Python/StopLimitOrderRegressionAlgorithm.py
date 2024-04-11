# QUANTCONNECT.COM - Democratizing Finance, Empowering Individuals.
# Lean Algorithmic Trading Engine v2.0. Copyright 2014 QuantConnect Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from AlgorithmImports import *

### <summary>
### Basic algorithm demonstrating how to place stop limit orders.
### </summary>
### <meta name="tag" content="trading and orders" />
### <meta name="tag" content="placing orders" />
### <meta name="tag" content="stop limit order"/>
class StopLimitOrderRegressionAlgorithm(QCAlgorithm):
    '''Basic algorithm demonstrating how to place stop limit orders.'''

    Tolerance = 0.001
    FastPeriod = 30
    SlowPeriod = 60

    def initialize(self):
        self.set_start_date(2013, 1, 1)
        self.set_end_date(2017, 1, 1)
        self.set_cash(100000)

        self._symbol = self.add_equity("SPY", Resolution.daily).symbol

        self._fast = self.EMA(self._symbol, self.fast_period, Resolution.daily)
        self._slow = self.EMA(self._symbol, self.slow_period, Resolution.daily)

        self._buy_order_ticket: OrderTicket = None
        self._sell_order_ticket: OrderTicket = None
        self._previous_slice: Slice = None

    def on_data(self, slice: Slice):
        if not self.is_ready():
            return

        security = self.securities[self._symbol]
        if self._buy_order_ticket is None and self.trend_is_up():
            self._buy_order_ticket = self.stop_limit_order(self._symbol, 100, stopPrice=security.high * 1.10, limitPrice=security.high * 1.11)
        elif self._buy_order_ticket.STATUS == OrderStatus.FILLED and self._sell_order_ticket is None and self.trend_is_down():
            self._sell_order_ticket = self.stop_limit_order(self._symbol, -100, stopPrice=security.low * 0.99, limitPrice=security.low * 0.98)

    def on_order_event(self, orderEvent: OrderEvent):
        if orderEvent.status == OrderStatus.FILLED:
            order: StopLimitOrder = self.transactions.get_order_by_id(orderEvent.order_id)
            if not order.stop_triggered:
                raise Exception("StopLimitOrder StopTriggered should haven been set if the order filled.")

            if orderEvent.direction == OrderDirection.buy:
                limitPrice = self._buy_order_ticket.get(OrderField.limit_price)
                if orderEvent.fill_price > limitPrice:
                    raise Exception(f"Buy stop limit order should have filled with price less than or equal to the limit price {limitPrice}. "
                                    f"Fill price: {orderEvent.fill_price}")
            else:
                limitPrice = self._sell_order_ticket.get(OrderField.limit_price)
                if orderEvent.fill_price < limitPrice:
                    raise Exception(f"Sell stop limit order should have filled with price greater than or equal to the limit price {limitPrice}. "
                                    f"Fill price: {orderEvent.fill_price}")

    def is_ready(self):
        return self._fast.is_ready and self._slow.is_ready

    def trend_is_up(self):
        return self.is_ready() and self._fast.current.value > self._slow.current.value * (1 + self.tolerance)

    def trend_is_down(self):
        return self.is_ready() and self._fast.current.value < self._slow.current.value * (1 + self.tolerance)
