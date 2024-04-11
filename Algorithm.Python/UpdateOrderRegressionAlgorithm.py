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
from math import copysign

### <summary>
### Provides a regression baseline focused on updating orders
### </summary>
### <meta name="tag" content="regression test" />
class UpdateOrderRegressionAlgorithm(QCAlgorithm):

    def initialize(self):
        '''Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'''

        self.set_start_date(2013,1,1)    #Set Start Date
        self.set_end_date(2015,1,1)      #Set End Date
        self.set_cash(100000)           #Set Strategy Cash
        # Find more symbols here: http://quantconnect.com/data

        self.security = self.add_equity("SPY", Resolution.daily)

        self.last_month = -1
        self.quantity = 100
        self.delta_quantity = 10

        self.stop_percentage = 0.025
        self.stop_percentage_delta = 0.005
        self.limit_percentage = 0.025
        self.limit_percentage_delta = 0.005

        OrderTypeEnum = [OrderType.MARKET, OrderType.LIMIT, OrderType.STOPMARKET, OrderType.STOPLIMIT, OrderType.MARKETONOPEN, OrderType.MARKETONCLOSE, OrderType.TRAILINGSTOP]
        self.order_types_queue = CircularQueue[OrderType](OrderTypeEnum)
        self.order_types_queue.circle_completed += self.on_circle_completed
        self.tickets = []


    def on_circle_completed(self, sender, event):
        '''Flip our signs when we've gone through all the order types'''
        self.quantity *= -1


    def on_data(self, data):
        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.'''
        if not data.contains_key("SPY"):
            return

        if self.time.MONTH != self.last_month:
            # we'll submit the next type of order from the queue
            orderType = self.order_types_queue.dequeue()
            #Log("")
            self.log("\r\n--------------MONTH: {0}:: {1}\r\n".format(self.time.strftime("%B"), orderType))
            #Log("")
            self.last_month = self.time.MONTH
            self.log("ORDER TYPE:: {0}".format(orderType))
            isLong = self.quantity > 0
            stopPrice = (1 + self.stop_percentage)*data["SPY"].high if isLong else (1 - self.stop_percentage)*data["SPY"].low
            limitPrice = (1 - self.limit_percentage)*stopPrice if isLong else (1 + self.limit_percentage)*stopPrice

            if orderType == OrderType.LIMIT:
                limitPrice = (1 + self.limit_percentage)*data["SPY"].high if not isLong else (1 - self.limit_percentage)*data["SPY"].low

            request = SubmitOrderRequest(orderType, self.security.symbol.SECURITYTYPE, "SPY", self.quantity, stopPrice, limitPrice, 0, 0.01, True,
                                         self.utc_time, str(orderType))
            ticket = self.transactions.add_order(request)
            self.tickets.append(ticket)

        elif len(self.tickets) > 0:
            ticket = self.tickets[-1]

            if self.time.DAY > 8 and self.time.DAY < 14:
                if len(ticket.update_requests) == 0 and ticket.status is not OrderStatus.FILLED:
                    self.log("TICKET:: {0}".format(ticket))
                    updateOrderFields = UpdateOrderFields()
                    updateOrderFields.quantity = ticket.quantity + copysign(self.delta_quantity, self.quantity)
                    updateOrderFields.tag = "Change quantity: {0}".format(self.time.DAY)
                    ticket.update(updateOrderFields)

            elif self.time.DAY > 13 and self.time.DAY < 20:
                if len(ticket.update_requests) == 1 and ticket.status is not OrderStatus.FILLED:
                    self.log("TICKET:: {0}".format(ticket))
                    updateOrderFields = UpdateOrderFields()
                    updateOrderFields.limit_price = self.security.price*(1 - copysign(self.limit_percentage_delta, ticket.quantity))
                    updateOrderFields.stop_price = self.security.price*(1 + copysign(self.stop_percentage_delta, ticket.quantity)) if ticket.order_type != OrderType.TRAILINGSTOP else None
                    updateOrderFields.tag = "Change prices: {0}".format(self.time.DAY)
                    ticket.update(updateOrderFields)
            else:
                if len(ticket.update_requests) == 2 and ticket.status is not OrderStatus.FILLED:
                    self.log("TICKET:: {0}".format(ticket))
                    ticket.cancel("{0} and is still open!".format(self.time.DAY))
                    self.log("CANCELLED:: {0}".format(ticket.cancel_request))


    def on_order_event(self, orderEvent):
        order = self.transactions.get_order_by_id(orderEvent.order_id)
        ticket = self.transactions.get_order_ticket(orderEvent.order_id)

        #order cancelations update CanceledTime
        if order.status == OrderStatus.CANCELED and order.canceled_time != orderEvent.utc_time:
            raise ValueError("Expected canceled order CanceledTime to equal canceled order event time.")

        #fills update LastFillTime
        if (order.status == OrderStatus.FILLED or order.status == OrderStatus.partially_filled) and order.last_fill_time != orderEvent.utc_time:
            raise ValueError("Expected filled order LastFillTime to equal fill order event time.")

        # check the ticket to see if the update was successfully processed
        if len([ur for ur in ticket.update_requests if ur.response is not None and ur.response.is_success]) > 0 and order.created_time != self.utc_time and order.last_update_time is None:
            raise ValueError("Expected updated order LastUpdateTime to equal submitted update order event time")

        if orderEvent.status == OrderStatus.FILLED:
            self.log("FILLED:: {0} FILL PRICE:: {1}".format(self.transactions.get_order_by_id(orderEvent.order_id), orderEvent.fill_price))
        else:
            self.log(orderEvent.to_string())
            self.log("TICKET:: {0}".format(ticket))
