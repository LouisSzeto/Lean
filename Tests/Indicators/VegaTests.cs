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
using QuantConnect.Data;
using QuantConnect.Indicators;
using System.IO;

namespace QuantConnect.Tests.Indicators
{
    [TestFixture]
    public class VegaTests : OptionBaseIndicatorTests<Vega>
    {
        protected override IndicatorBase<IndicatorDataPoint> CreateIndicator()
            => new Vega("testVegaIndicator", _symbol, 0.053m, 0.0153m);

        protected override OptionIndicatorBase CreateIndicator(IRiskFreeInterestRateModel riskFreeRateModel)
            => new Vega("testVegaIndicator", _symbol, riskFreeRateModel);

        protected override OptionIndicatorBase CreateIndicator(IRiskFreeInterestRateModel riskFreeRateModel, IDividendYieldModel dividendYieldModel)
            => new Vega("testVegaIndicator", _symbol, riskFreeRateModel, dividendYieldModel);


        protected override OptionIndicatorBase CreateIndicator(QCAlgorithm algorithm)
            => algorithm.V(_symbol);

        [SetUp]
        public void SetUp()
        {
            RiskFreeRateUpdatesPerIteration = 3;
            DividendYieldUpdatesPerIteration = 3;
        }

        [TestCase("SPX230811C04300000", 0.31)]
        [TestCase("SPX230811C04500000", 0.16)]
        [TestCase("SPX230811C04700000", 0.02)]
        [TestCase("SPX230811P04300000", 0.03)]
        [TestCase("SPX230811P04500000", 0.15)]
        [TestCase("SPX230811P04700000", 0.41)]
        [TestCase("SPX230901C04300000", 0.20)]
        [TestCase("SPX230901C04500000", 0.09)]
        [TestCase("SPX230901C04700000", 0.03)]
        [TestCase("SPX230901P04300000", 0.05)]
        [TestCase("SPX230901P04500000", 0.08)]
        [TestCase("SPX230901P04700000", 0.21)]
        public void ComparesAgainstExternalData(string fileName, double errorMargin = 0.005, int column = 5)
        {
            var path = Path.Combine("TestData", "greeks", $"{fileName}.csv");
            var symbol = ParseOptionSymbol(fileName);
            var underlying = symbol.Underlying;

            var indicator = new Vega(symbol, 0.053m, 0.0153m);
            RunTestIndicator(path, indicator, symbol, underlying, errorMargin, column);
        }

        [TestCase("SPY230811C00430000", 0.03)]
        [TestCase("SPY230811C00450000", 0.02)]
        [TestCase("SPY230811C00470000", 0.01)]
        [TestCase("SPY230811P00430000", 0.02)]
        [TestCase("SPY230811P00450000", 0.02)]
        [TestCase("SPY230811P00470000", 0.06)]
        [TestCase("SPY230901C00430000", 0.05)]
        [TestCase("SPY230901C00450000")]
        [TestCase("SPY230901C00470000", 0.04)]
        [TestCase("SPY230901P00430000", 0.05)]
        [TestCase("SPY230901P00450000")]
        [TestCase("SPY230901P00470000", 0.22)]
        public void ComparesAgainstExternalDataCRRModel(string fileName, double errorMargin = 0.005, int column = 5)
        {
            var path = Path.Combine("TestData", "greeks", $"{fileName}.csv");
            var symbol = ParseOptionSymbol(fileName);
            var underlying = symbol.Underlying;

            var indicator = new Vega(symbol, 0.053m, 0.0153m, OptionPricingModelType.BinomialCoxRossRubinstein,
                    OptionPricingModelType.BlackScholes);
            RunTestIndicator(path, indicator, symbol, underlying, errorMargin, column);

            indicator.Reset();
            RunTestIndicator(path, indicator, symbol, underlying, errorMargin, column);
        }

        [TestCase("SPX230811C04300000", 0.31)]
        [TestCase("SPX230811C04500000", 0.16)]
        [TestCase("SPX230811C04700000", 0.02)]
        [TestCase("SPX230811P04300000", 0.03)]
        [TestCase("SPX230811P04500000", 0.15)]
        [TestCase("SPX230811P04700000", 0.41)]
        [TestCase("SPX230901C04300000", 0.20)]
        [TestCase("SPX230901C04500000", 0.09)]
        [TestCase("SPX230901C04700000", 0.03)]
        [TestCase("SPX230901P04300000", 0.05)]
        [TestCase("SPX230901P04500000", 0.08)]
        [TestCase("SPX230901P04700000", 0.21)]
        public void ComparesAgainstExternalDataAfterReset(string fileName, double errorMargin = 0.005, int column = 5)
        {
            var path = Path.Combine("TestData", "greeks", $"{fileName}.csv");
            var symbol = ParseOptionSymbol(fileName);
            var underlying = symbol.Underlying;

            var indicator = new Vega(symbol, 0.053m, 0.0153m);
            RunTestIndicator(path, indicator, symbol, underlying, errorMargin, column);

            indicator.Reset();
            RunTestIndicator(path, indicator, symbol, underlying, errorMargin, column);
        }

        // Reference values from QuantLib
        [TestCase(23.753, 450.0, OptionRight.Call, 60, 0.7215)]
        [TestCase(35.830, 450.0, OptionRight.Put, 60, 0.7195)]
        [TestCase(33.928, 470.0, OptionRight.Call, 60, 0.6705)]
        [TestCase(6.428, 470.0, OptionRight.Put, 60, 0.6181)]
        [TestCase(3.219, 430.0, OptionRight.Call, 60, 0.5429)]
        [TestCase(47.701, 430.0, OptionRight.Put, 60, 0.6922)]
        [TestCase(16.528, 450.0, OptionRight.Call, 180, 1.1932)]
        [TestCase(21.784, 450.0, OptionRight.Put, 180, 1.2263)]
        [TestCase(35.207, 470.0, OptionRight.Call, 180, 1.0370)]
        [TestCase(0.409, 470.0, OptionRight.Put, 180, 0.3528)]
        [TestCase(2.642, 430.0, OptionRight.Call, 180, 0.9707)]
        [TestCase(27.772, 430.0, OptionRight.Put, 180, 1.1816)]
        public void ComparesVegaOnBSMModel(decimal price, decimal spotPrice, OptionRight right, int expiry, double refVega)
        {
            var symbol = Symbol.CreateOption("SPY", Market.USA, OptionStyle.American, right, 450m, _reference.AddDays(expiry));
            var indicator = new Vega(symbol, 0.053m, 0.0153m, optionModel: OptionPricingModelType.BlackScholes);

            var optionDataPoint = new IndicatorDataPoint(symbol, _reference, price);
            var spotDataPoint = new IndicatorDataPoint(symbol.Underlying, _reference, spotPrice);
            indicator.Update(optionDataPoint);
            indicator.Update(spotDataPoint);

            Assert.AreEqual(refVega, (double)indicator.Current.Value, 0.0001d);
        }

        // Reference values from QuantLib
        [TestCase(23.753, 450.0, OptionRight.Call, 60, 0.7206)]
        [TestCase(35.830, 450.0, OptionRight.Put, 60, 0.7189)]
        [TestCase(33.928, 470.0, OptionRight.Call, 60, 0.6791)]
        [TestCase(6.428, 470.0, OptionRight.Put, 60, 0.6308)]
        [TestCase(3.219, 430.0, OptionRight.Call, 60, 0.5725)]
        [TestCase(47.701, 430.0, OptionRight.Put, 60, 0.6909)]
        [TestCase(16.528, 450.0, OptionRight.Call, 180, 1.1918)]
        [TestCase(21.784, 450.0, OptionRight.Put, 180, 1.2238)]
        [TestCase(35.207, 470.0, OptionRight.Call, 180, 1.0129)]
        [TestCase(0.409, 470.0, OptionRight.Put, 180, 0.4070)]
        [TestCase(2.642, 430.0, OptionRight.Call, 180, 1.0433)]
        [TestCase(27.772, 430.0, OptionRight.Put, 180, 1.1293)]
        public void ComparesVegaOnCRRModel(decimal price, decimal spotPrice, OptionRight right, int expiry, double refVega)
        {
            var symbol = Symbol.CreateOption("SPY", Market.USA, OptionStyle.American, right, 450m, _reference.AddDays(expiry));
            var indicator = new Vega(symbol, 0.053m, 0.0153m, optionModel: OptionPricingModelType.BinomialCoxRossRubinstein,
                ivModel: OptionPricingModelType.BlackScholes);

            var optionDataPoint = new IndicatorDataPoint(symbol, _reference, price);
            var spotDataPoint = new IndicatorDataPoint(symbol.Underlying, _reference, spotPrice);
            indicator.Update(optionDataPoint);
            indicator.Update(spotDataPoint);

            Assert.AreEqual(refVega, (double)indicator.Current.Value, 0.005d);
        }
    }
}
