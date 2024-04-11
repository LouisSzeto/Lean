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
from Selection.uncorrelated_universe_selection_model import UncorrelatedUniverseSelectionModel

class UncorrelatedUniverseSelectionFrameworkAlgorithm(QCAlgorithm):

    def initialize(self):

        self.universe_settings.resolution = Resolution.daily

        self.set_start_date(2018,1,1)   # Set Start Date
        self.set_cash(1000000)         # Set Strategy Cash


        benchmark = Symbol.create("SPY", SecurityType.EQUITY, Market.USA)
        self.set_universe_selection(UncorrelatedUniverseSelectionModel(benchmark))
        self.set_alpha(UncorrelatedUniverseSelectionAlphaModel())
        self.set_portfolio_construction(EqualWeightingPortfolioConstructionModel())
        self.set_execution(ImmediateExecutionModel())


class UncorrelatedUniverseSelectionAlphaModel(AlphaModel):
    '''Uses ranking of intraday percentage difference between open price and close price to create magnitude and direction prediction for insights'''

    def __init__(self, numberOfStocks = 10, predictionInterval = timedelta(1)):
        self.prediction_interval = predictionInterval
        self.number_of_stocks = numberOfStocks

    def update(self, algorithm, data):
        symbolsRet = dict()

        for kvp in algorithm.active_securities:
            security = kvp.value
            if security.has_data:
                open = security.open
                if open != 0:
                    symbolsRet[security.symbol] = security.close / open - 1

        # Rank on the absolute value of price change
        symbolsRet = dict(sorted(symbolsRet.items(), key=lambda kvp: abs(kvp[1]),reverse=True)[:self.number_of_stocks])

        insights = []
        for symbol, price_change in symbolsRet.items():
            # Emit "up" insight if the price change is positive and "down" otherwise
            direction = InsightDirection.UP if price_change > 0 else InsightDirection.down
            insights.append(Insight.price(symbol, self.prediction_interval, direction, abs(price_change), None))

        return insights
