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
### Regression algorithm asserting the behavior of Universe.selected collection
### </summary>
class UniverseSelectedRegressionAlgorithm(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2014, 3, 25)
        self.set_end_date(2014, 3, 27)

        self.universe_settings.resolution = Resolution.daily

        self.universe = self.add_universe(self.selection_function)
        self.selection_count = 0

    def selection_function(self, fundamentals):
        sortedByDollarVolume = sorted(fundamentals, key=lambda x: x.dollar_volume, reverse=True)

        sortedByDollarVolume = sortedByDollarVolume[self.selection_count:]
        self.selection_count = self.selection_count + 1

        # return the symbol objects of the top entries from our sorted collection
        return [ x.symbol for x in sortedByDollarVolume[:self.selection_count] ]

    def on_data(self, data):
        if Symbol.create("TSLA", SecurityType.EQUITY, Market.USA) in self.universe.selected:
            raise ValueError(f"TSLA shouldn't of been selected")

        self.buy(next(iter(self.universe.selected)), 1)

    def on_end_of_algorithm(self):
        if self.selection_count != 3:
            raise ValueError(f"Unexpected selection count {self.selection_count}")
        if self.universe.selected.count != 3 or self.universe.selected.count == self.universe.members.count:
            raise ValueError(f"Unexpected universe selected count {self.universe.selected.count}")
