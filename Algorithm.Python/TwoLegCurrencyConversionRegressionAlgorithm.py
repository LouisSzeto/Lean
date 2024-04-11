### QUANTCONNECT.COM - Democratizing Finance, Empowering Individuals.
### Lean Algorithmic Trading Engine v2.0. Copyright 2014 QuantConnect Corporation.
###
### Licensed under the Apache License, Version 2.0 (the "License");
### you may not use this file except in compliance with the License.
### You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
###
### Unless required by applicable law or agreed to in writing, software
### distributed under the License is distributed on an "AS IS" BASIS,
### WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
### See the License for the specific language governing permissions and
### limitations under the License.

from AlgorithmImports import *

### <summary>
### Regression algorithm which tests that a two leg currency conversion happens correctly
### </summary>
class TwoLegCurrencyConversionRegressionAlgorithm(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2018, 4, 4)
        self.set_end_date(2018, 4, 4)
        self.set_brokerage_model(BrokerageName.GDAX, AccountType.cash)
        # GDAX doesn't have LTCETH or ETHLTC, but they do have ETHUSD and LTCUSD to form a path between ETH and LTC
        self.set_account_currency("ETH")
        self.set_cash("ETH", 100000)
        self.set_cash("LTC", 100000)
        self.set_cash("USD", 100000)

        self._eth_usd_symbol = self.add_crypto("ETHUSD", Resolution.minute).symbol
        self._ltc_usd_symbol = self.add_crypto("LTCUSD", Resolution.minute).symbol

    def on_data(self, data):
        if not self.portfolio.invested:
            self.market_order(self._ltc_usd_symbol, 1)

    def on_end_of_algorithm(self):
        ltcCash = self.portfolio.cash_book["LTC"]

        conversionSymbols = [x.symbol for x in ltcCash.currency_conversion.conversion_rate_securities]

        if len(conversionSymbols) != 2:
            raise ValueError(
                f"Expected two conversion rate securities for LTC to ETH, is {len(conversionSymbols)}")

        if conversionSymbols[0] != self._ltc_usd_symbol:
            raise ValueError(
                f"Expected first conversion rate security from LTC to ETH to be {self._ltc_usd_symbol}, is {conversionSymbols[0]}")

        if conversionSymbols[1] != self._eth_usd_symbol:
            raise ValueError(
                f"Expected second conversion rate security from LTC to ETH to be {self._eth_usd_symbol}, is {conversionSymbols[1]}")

        ltcUsdValue = self.securities[self._ltc_usd_symbol].get_last_data().value
        ethUsdValue = self.securities[self._eth_usd_symbol].get_last_data().value

        expectedConversionRate = ltcUsdValue / ethUsdValue
        actualConversionRate = ltcCash.conversion_rate

        if actualConversionRate != expectedConversionRate:
            raise ValueError(
                f"Expected conversion rate from LTC to ETH to be {expectedConversionRate}, is {actualConversionRate}")
