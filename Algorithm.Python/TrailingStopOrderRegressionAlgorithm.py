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
### Basic algorithm demonstrating how to place trailing stop orders.
### </summary>
### <meta name="tag" content="trading and orders" />
### <meta name="tag" content="placing orders" />
### <meta name="tag" content="trailing stop order"/>
class TrailingStopOrderRegressionAlgorithm(QCAlgorithm):
    '''Basic algorithm demonstrating how to place trailing stop orders.'''

    BuyTrailingAmount = 2
    SellTrailingAmount = 0.5

    def initialize(self):

        self.set_start_date(2013,10, 7)
        self.set_end_date(2013,10,11)
        self.set_cash(100000)

        self._symbol = self.add_equity("SPY").symbol

        self._buy_order_ticket: OrderTicket = None
        self._sell_order_ticket: OrderTicket = None
        self._previous_slice: Slice = None

    def on_data(self, slice: Slice):
        if not slice.contains_key(self._symbol):
            return

        if self._buy_order_ticket is None:
            self._buy_order_ticket = self.trailing_stop_order(self._symbol, 100, trailingAmount=self.buy_trailing_amount, trailingAsPercentage=False)
        elif self._buy_order_ticket.STATUS != OrderStatus.filled:
            stopPrice = self._buy_order_ticket.get(OrderField.stop_price)

            # Get the previous bar to compare to the stop price,
            # because stop price update attempt with the current slice data happens after OnData.
            low = self._previous_slice.quote_bars[self._symbol].ask.low if self._previous_slice.quote_bars.contains_key(self._symbol) \
                  else self._previous_slice.bars[self._symbol].low

            stopPriceToMarketPriceDistance = stopPrice - low
            if stopPriceToMarketPriceDistance > self.buy_trailing_amount:
                raise Exception(f"StopPrice {stopPrice} should be within {self.buy_trailing_amount} of the previous low price {low} at all times.")

        if self._sell_order_ticket is None:
            if self.portfolio.invested:
                self._sell_order_ticket = self.trailing_stop_order(self._symbol, -100, trailingAmount=self.sell_trailing_amount, trailingAsPercentage=False)
        elif self._sell_order_ticket.STATUS != OrderStatus.filled:
            stopPrice = self._sell_order_ticket.get(OrderField.stop_price)

            # Get the previous bar to compare to the stop price,
            # because stop price update attempt with the current slice data happens after OnData.
            high = self._previous_slice.quote_bars[self._symbol].bid.high if self._previous_slice.quote_bars.contains_key(self._symbol) \
                   else self._previous_slice.bars[self._symbol].high
            stopPriceToMarketPriceDistance = high - stopPrice
            if stopPriceToMarketPriceDistance > self.sell_trailing_amount:
                raise Exception(f"StopPrice {stopPrice} should be within {self.sell_trailing_amount} of the previous high price {high} at all times.")

        self._previous_slice = slice

    def on_order_event(self, orderEvent: OrderEvent):
        if orderEvent.status == OrderStatus.filled:
            if orderEvent.direction == OrderDirection.buy:
                stopPrice = self._buy_order_ticket.get(OrderField.stop_price)
                if orderEvent.fill_price < stopPrice:
                    raise Exception(f"Buy trailing stop order should have filled with price greater than or equal to the stop price {stopPrice}. "
                                    f"Fill price: {orderEvent.fill_price}")
            else:
                stopPrice = self._sell_order_ticket.get(OrderField.stop_price)
                if orderEvent.fill_price > stopPrice:
                    raise Exception(f"Sell trailing stop order should have filled with price less than or equal to the stop price {stopPrice}. "
                                    f"Fill price: {orderEvent.fill_price}")
