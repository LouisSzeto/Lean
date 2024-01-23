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

using System.IO;
using NUnit.Framework;
using QuantConnect.Algorithm;
using QuantConnect.Data;
using QuantConnect.Indicators;

namespace QuantConnect.Tests.Indicators
{
    [TestFixture]
    public class ThetaTests : OptionBaseIndicatorTests<Theta>
    {
        protected override IndicatorBase<IndicatorDataPoint> CreateIndicator()
            => new Theta("testThetaIndicator", _symbol, 0.04m);

        protected override OptionIndicatorBase CreateIndicator(IRiskFreeInterestRateModel riskFreeRateModel)
            => new Theta("testThetaIndicator", _symbol, riskFreeRateModel);

        protected override OptionIndicatorBase CreateIndicator(QCAlgorithm algorithm)
            => algorithm.Theta(_symbol);

        [SetUp]
        public void SetUp()
        {
            RiskFreeRateUpdatesPerIteration = 3;
        }

        [TestCase("SPX230811C04300000", 0.68)]
        [TestCase("SPX230811C04500000", 0.1)]
        [TestCase("SPX230811C04700000")]
        [TestCase("SPX230811P04300000", 0.08)]
        [TestCase("SPX230811P04500000", 0.37)]
        [TestCase("SPX230811P04700000", 0.71)]
        [TestCase("SPX230901C04300000", 0.25)]
        [TestCase("SPX230901C04500000", 0.14)]
        [TestCase("SPX230901C04700000")]
        [TestCase("SPX230901P04300000", 0.06)]
        [TestCase("SPX230901P04500000", 0.13)]
        [TestCase("SPX230901P04700000", 0.3)]
        public void ComparesAgainstExternalData(string fileName, double errorMargin = 0.05, int column = 6)
        {
            var path = Path.Combine("TestData", "greeks", $"{fileName}.csv");
            var symbol = ParseOptionSymbol(fileName);
            var underlying = symbol.Underlying;

            var indicator = new Theta(symbol, 0.04m);
            RunTestIndicator(path, indicator, symbol, underlying, errorMargin, column);
        }

        [TestCase("SPY230811C00430000", 0.12)]
        [TestCase("SPY230811C00450000")]
        [TestCase("SPY230811C00470000")]
        [TestCase("SPY230811P00430000")]
        [TestCase("SPY230811P00450000")]
        [TestCase("SPY230811P00470000")]
        [TestCase("SPY230901C00430000")]
        [TestCase("SPY230901C00450000")]
        [TestCase("SPY230901C00470000")]
        [TestCase("SPY230901P00430000")]
        [TestCase("SPY230901P00450000")]
        [TestCase("SPY230901P00470000")]
        public void ComparesAgainstExternalDataCRRModel(string fileName, double errorMargin = 0.05, int column = 6)
        {
            var path = Path.Combine("TestData", "greeks", $"{fileName}.csv");
            var symbol = ParseOptionSymbol(fileName);
            var underlying = symbol.Underlying;

            var indicator = new Theta(symbol, 0.04m, OptionPricingModelType.BinomialCoxRossRubinstein,
                    OptionPricingModelType.BlackScholes);
            RunTestIndicator(path, indicator, symbol, underlying, errorMargin, column);

            indicator.Reset();
            RunTestIndicator(path, indicator, symbol, underlying, errorMargin, column);
        }

        [TestCase("SPX230811C04300000", 0.68)]
        [TestCase("SPX230811C04500000", 0.1)]
        [TestCase("SPX230811C04700000")]
        [TestCase("SPX230811P04300000", 0.08)]
        [TestCase("SPX230811P04500000", 0.37)]
        [TestCase("SPX230811P04700000", 0.71)]
        [TestCase("SPX230901C04300000", 0.25)]
        [TestCase("SPX230901C04500000", 0.14)]
        [TestCase("SPX230901C04700000")]
        [TestCase("SPX230901P04300000", 0.06)]
        [TestCase("SPX230901P04500000", 0.13)]
        [TestCase("SPX230901P04700000", 0.3)]
        public void ComparesAgainstExternalDataAfterReset(string fileName, double errorMargin = 0.05, int column = 6)
        {
            var path = Path.Combine("TestData", "greeks", $"{fileName}.csv");
            var symbol = ParseOptionSymbol(fileName);
            var underlying = symbol.Underlying;

            var indicator = new Theta(symbol, 0.04m);
            RunTestIndicator(path, indicator, symbol, underlying, errorMargin, column);

            indicator.Reset();
            RunTestIndicator(path, indicator, symbol, underlying, errorMargin, column);
        }

        // Reference values from QuantLib
        [TestCase(23.753, 450.0, OptionRight.Call, 60, -0.2092)]
        [TestCase(35.830, 450.0, OptionRight.Put, 60, -0.2835)]
        [TestCase(33.928, 470.0, OptionRight.Call, 60, -0.1858)]
        [TestCase(6.428, 470.0, OptionRight.Put, 60, -0.0918)]
        [TestCase(3.219, 430.0, OptionRight.Call, 60, -0.0712)]
        [TestCase(47.701, 430.0, OptionRight.Put, 60, -0.2852)]
        [TestCase(16.528, 450.0, OptionRight.Call, 180, -0.0601)]
        [TestCase(21.784, 450.0, OptionRight.Put, 180, -0.0483)]
        [TestCase(35.207, 470.0, OptionRight.Call, 180, -0.0733)]
        [TestCase(0.409, 470.0, OptionRight.Put, 180, -0.0027)]
        [TestCase(2.642, 430.0, OptionRight.Call, 180, -0.0274)]
        [TestCase(27.772, 430.0, OptionRight.Put, 180, -0.0297)]
        public void ComparesThetaOnBSMModel(decimal price, decimal spotPrice, OptionRight right, int expiry, double refTheta)
        {
            var symbol = Symbol.CreateOption("SPY", Market.USA, OptionStyle.American, right, 450m, _reference.AddDays(expiry));
            var indicator = new Theta(symbol, 0.04m, optionModel: OptionPricingModelType.BlackScholes);

            var optionDataPoint = new IndicatorDataPoint(symbol, _reference, price);
            var spotDataPoint = new IndicatorDataPoint(symbol.Underlying, _reference, spotPrice);
            indicator.Update(optionDataPoint);
            indicator.Update(spotDataPoint);

            Assert.AreEqual(refTheta, (double)indicator.Current.Value, 0.001d);
        }

        // QuantLib/WolframAlpha does not have theta for binomial tree model
    }
}
