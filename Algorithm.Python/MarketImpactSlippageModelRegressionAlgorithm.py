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

class MarketImpactSlippageModelRegressionAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2013, 10, 7)
        self.SetEndDate(2013, 10, 13)
        self.SetCash(1000000000)

        spy = self.AddEquity("SPY", Resolution.Daily)
        aapl = self.AddEquity("AAPL", Resolution.Daily)
        eem = self.AddEquity("EEM", Resolution.Daily)
        wm = self.AddEquity("WM", Resolution.Daily)

        spy.SetSlippageModel(MarketImpactSlippageModel(self))
        aapl.SetSlippageModel(MarketImpactSlippageModel(self))
        eem.SetSlippageModel(MarketImpactSlippageModel(self))
        wm.SetSlippageModel(MarketImpactSlippageModel(self))

        self.SetWarmUp(1)

    def OnData(self, data):
        self.SetHoldings("SPY", 0.25)
        self.SetHoldings("AAPL", 0.25)
        self.SetHoldings("EEM", 0.25)
        self.SetHoldings("WM", 0.25)

    def OnOrderEvent(self, orderEvent):
        if orderEvent.Status == OrderStatus.Filled:
            self.Debug(f"Price: {self.Securities[orderEvent.Symbol].Price}, filled price: {orderEvent.FillPrice}")
