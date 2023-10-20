/*
 * QUANTCONNECT.COM - Democratizing Finance, Empowering Individuals.
 * Lean Algorithmic Trading Engine v2.0. Copyright 2014 QuantConnect Corporation.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

using NUnit.Framework;
using QuantConnect.Algorithm;
using QuantConnect.Algorithm.Framework.Alphas;
using QuantConnect.Data;
using QuantConnect.Lean.Engine.DataFeeds;
using QuantConnect.Lean.Engine.HistoricalData;
using QuantConnect.Orders;
using QuantConnect.Orders.Slippage;
using QuantConnect.Securities;
using QuantConnect.Tests.Engine.DataFeeds;
using System;
using System.Collections.Generic;

namespace QuantConnect.Tests.Common.Orders.Slippage
{
    [TestFixture]
    public class MarketImpactSlippageModelTests
    {
        private QCAlgorithm _algorithm;
        private MarketImpactSlippageModel _slippageModel;
        private List<Security> _securities;

        [SetUp]
        public void Initialize()
        {
            _algorithm = new QCAlgorithm();
            _algorithm.SubscriptionManager.SetDataManager(new DataManagerStub(_algorithm));

            var historyProvider = new SubscriptionDataReaderHistoryProvider();
            historyProvider.Initialize(new HistoryProviderInitializeParameters(null, null,
                TestGlobals.DataProvider, TestGlobals.DataCacheProvider, TestGlobals.MapFileProvider, TestGlobals.FactorFileProvider,
                null, true, new DataPermissionManager(), _algorithm.ObjectStore));
            _algorithm.SetHistoryProvider(historyProvider);

            var optionContract = Symbol.CreateOption(Symbols.GOOG, Market.USA,
                OptionStyle.American, OptionRight.Call, 740, new DateTime(2015, 12, 24));

            _algorithm.SetDateTime(new DateTime(2015, 12, 23, 15, 0, 0));
            _securities = new List<Security>
            {
                _algorithm.AddEquity("SPY", Resolution.Daily),
                _algorithm.AddEquity("WM", Resolution.Daily),
                _algorithm.AddForex("EURUSD", Resolution.Daily),
                _algorithm.AddForex("GBPUSD", Resolution.Daily),
                _algorithm.AddCrypto("BTCUSD", Resolution.Daily, Market.GDAX),
                _algorithm.AddOptionContract(optionContract)
            };

            _algorithm.EnableAutomaticIndicatorWarmUp = true;

            _slippageModel = new MarketImpactSlippageModel(_algorithm);
        }

        // Test on buy & sell orders
        [TestCase(InsightDirection.Up)]
        [TestCase(InsightDirection.Down)]
        public void SizeSlippageComparisonTests(InsightDirection direction)
        {
            // Test on all liquid/illquid stocks/other asset classes
            foreach (var asset in _securities)
            {
                // A significantly large difference that noise cannot affect the result
                var smallBuyOrder = new MarketOrder(asset.Symbol, 10 * (int)direction, new DateTime(2015, 12, 22, 14, 50, 0));
                var largeBuyOrder = new MarketOrder(asset.Symbol, 10000000000 * (int)direction, new DateTime(2015, 12, 22, 14, 50, 0));

                var smallBuySlippage = _slippageModel.GetSlippageApproximation(asset, smallBuyOrder);
                var largeBuySlippage = _slippageModel.GetSlippageApproximation(asset, largeBuyOrder);

                // We expect small size order has less slippage than large size order on the same asset
                Assert.Less(smallBuySlippage, largeBuySlippage);
            }
        }

        // Order quantity large enough to create significant market impact
        // Test for buy & sell orders
        [TestCase(1000000)]
        [TestCase(-1000000)]
        [TestCase(1000000000)]
        [TestCase(-1000000000)]
        public void LiquiditySlippageComparisonTests(decimal orderQuantity)
        {
            var liquidAsset = _securities[0];
            var illquidAsset = _securities[1];

            var liquidOrder = new MarketOrder(liquidAsset.Symbol, orderQuantity, new DateTime(2015, 12, 22, 14, 50, 0));
            var illquidOrder = new MarketOrder(illquidAsset.Symbol, orderQuantity, new DateTime(2015, 12, 22, 14, 50, 0));

            var liquidSlippage = _slippageModel.GetSlippageApproximation(liquidAsset, liquidOrder);
            var illquidSlippage = _slippageModel.GetSlippageApproximation(illquidAsset, illquidOrder);

            // We expect same size order on liquid asset has less slippage than illquid asset
            Assert.Less(liquidSlippage, illquidSlippage);
        }

        // Test on buy & sell orders
        [TestCase(100000)]
        [TestCase(-100000)]
        public void TimeSlippageComparisonTests(decimal orderQuantity)
        {
            // Test on all liquid/illquid stocks/other asset classes
            foreach (var asset in _securities)
            {
                var fastFilledOrder = new MarketOrder(asset.Symbol, orderQuantity, new DateTime(2015, 12, 22, 14, 50, 0));
                var slowFilledOrder = new MarketOrder(asset.Symbol, orderQuantity, new DateTime(2015, 12, 22, 14, 50, 0));
                var fastFilledSlippage = _slippageModel.GetSlippageApproximation(asset, fastFilledOrder);
                var slowFilledSlippage = _slippageModel.GetSlippageApproximation(asset, slowFilledOrder);

                // We expect same size order on same asset has less slippage if filled quicker
                Assert.Less(fastFilledSlippage, slowFilledSlippage);
            }
        }

        // To test whether the slippage matches our expectation
        [TestCase(100000, 1)]
        [TestCase(100000, 2)]
        [TestCase(100000, 3)]
        [TestCase(100000, 4)]
        [TestCase(-100000, 1)]
        [TestCase(-100000, 2)]
        [TestCase(-100000, 3)]
        [TestCase(-100000, 4)]
        [TestCase(100000000, 1)]
        [TestCase(100000000, 2)]
        [TestCase(100000000, 3)]
        [TestCase(100000000, 4)]
        [TestCase(-100000000, 1)]
        [TestCase(-100000000, 2)]
        [TestCase(-100000000, 3)]
        [TestCase(-100000000, 4)]
        public void SlippageExpectationTests(decimal orderQuantity, int index, double expected)
        {
            var asset = _securities[index];
            
            var order = new MarketOrder(asset.Symbol, orderQuantity, new DateTime(2015, 12, 22, 14, 50, 0));
            var slippage = _slippageModel.GetSlippageApproximation(asset, order);

            Assert.AreEqual(expected, (double)slippage, 0.05d);
        }

        // Test on buy & sell orders
        [TestCase(1)]
        [TestCase(-1)]
        [TestCase(1000)]
        [TestCase(-1000)]
        [TestCase(1000000000)]
        [TestCase(-1000000000)]
        public void NonNegativeSlippageTests(decimal orderQuantity)
        {
            // Test on all liquid/illquid stocks/other asset classes
            foreach (var asset in _securities)
            {
                var order = new MarketOrder(asset.Symbol, orderQuantity, new DateTime(2015, 12, 22, 14, 50, 0));
                var slippage = _slippageModel.GetSlippageApproximation(asset, order);

                Assert.GreaterOrEqual(slippage, 0m);
            }
        }
    }
}
