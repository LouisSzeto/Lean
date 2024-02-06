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
            => new Theta("testThetaIndicator", _symbol, 0.053m, 0.0153m);

        protected override OptionIndicatorBase CreateIndicator(IRiskFreeInterestRateModel riskFreeRateModel)
            => new Theta("testThetaIndicator", _symbol, riskFreeRateModel);

        protected override OptionIndicatorBase CreateIndicator(IRiskFreeInterestRateModel riskFreeRateModel, IDividendYieldModel dividendYieldModel)
            => new Theta("testThetaIndicator", _symbol, riskFreeRateModel, dividendYieldModel);

        protected override OptionIndicatorBase CreateIndicator(QCAlgorithm algorithm)
            => algorithm.T(_symbol);

        [SetUp]
        public void SetUp()
        {
            RiskFreeRateUpdatesPerIteration = 3;
            DividendYieldUpdatesPerIteration = 3;
        }

        [TestCase("SPX230811C04300000", 0.60)]
        [TestCase("SPX230811C04500000", 0.09)]
        [TestCase("SPX230811C04700000", 0.02)]
        [TestCase("SPX230811P04300000", 0.08)]
        [TestCase("SPX230811P04500000", 0.37)]
        [TestCase("SPX230811P04700000", 0.68)]
        [TestCase("SPX230901C04300000", 0.24)]
        [TestCase("SPX230901C04500000", 0.12)]
        [TestCase("SPX230901C04700000", 0.02)]
        [TestCase("SPX230901P04300000", 0.06)]
        [TestCase("SPX230901P04500000", 0.14)]
        [TestCase("SPX230901P04700000", 0.29)]
        public void ComparesAgainstExternalData(string fileName, double errorMargin, int column = 6)
        {
            var path = Path.Combine("TestData", "greeks", $"{fileName}.csv");
            var symbol = ParseOptionSymbol(fileName);
            var underlying = symbol.Underlying;

            var indicator = new Theta(symbol, 0.053m, 0.0153m);
            RunTestIndicator(path, indicator, symbol, underlying, errorMargin, column);
        }

        [TestCase("SPY230811C00430000", 0.06)]
        [TestCase("SPY230811C00450000", 0.03)]
        [TestCase("SPY230811C00470000", 0.005)]
        [TestCase("SPY230811P00430000", 0.02)]
        [TestCase("SPY230811P00450000", 0.04)]
        [TestCase("SPY230811P00470000", 0.07)]
        [TestCase("SPY230901C00430000", 0.01)]
        [TestCase("SPY230901C00450000", 0.005)]
        [TestCase("SPY230901C00470000", 0.005)]
        [TestCase("SPY230901P00430000", 0.005)]
        [TestCase("SPY230901P00450000", 0.01)]
        [TestCase("SPY230901P00470000", 0.05)]
        public void ComparesAgainstExternalDataCRRModel(string fileName, double errorMargin, int column = 6)
        {
            var path = Path.Combine("TestData", "greeks", $"{fileName}.csv");
            var symbol = ParseOptionSymbol(fileName);
            var underlying = symbol.Underlying;

            var indicator = new Theta(symbol, 0.053m, 0.0153m, OptionPricingModelType.BinomialCoxRossRubinstein,
                    OptionPricingModelType.BlackScholes);
            RunTestIndicator(path, indicator, symbol, underlying, errorMargin, column);

            indicator.Reset();
            RunTestIndicator(path, indicator, symbol, underlying, errorMargin, column);
        }

        [TestCase("SPX230811C04300000", 0.60)]
        [TestCase("SPX230811C04500000", 0.09)]
        [TestCase("SPX230811C04700000", 0.02)]
        [TestCase("SPX230811P04300000", 0.08)]
        [TestCase("SPX230811P04500000", 0.37)]
        [TestCase("SPX230811P04700000", 0.68)]
        [TestCase("SPX230901C04300000", 0.24)]
        [TestCase("SPX230901C04500000", 0.12)]
        [TestCase("SPX230901C04700000", 0.02)]
        [TestCase("SPX230901P04300000", 0.06)]
        [TestCase("SPX230901P04500000", 0.14)]
        [TestCase("SPX230901P04700000", 0.29)]
        public void ComparesAgainstExternalDataAfterReset(string fileName, double errorMargin, int column = 6)
        {
            var path = Path.Combine("TestData", "greeks", $"{fileName}.csv");
            var symbol = ParseOptionSymbol(fileName);
            var underlying = symbol.Underlying;

            var indicator = new Theta(symbol, 0.053m, 0.0153m);
            RunTestIndicator(path, indicator, symbol, underlying, errorMargin, column);

            indicator.Reset();
            RunTestIndicator(path, indicator, symbol, underlying, errorMargin, column);
        }

        // Reference values from QuantLib
        [TestCase(23.753, 450.0, OptionRight.Call, 60, -0.2075)]
        [TestCase(35.830, 450.0, OptionRight.Put, 60, -0.2828)]
        [TestCase(33.928, 470.0, OptionRight.Call, 60, -0.1842)]
        [TestCase(6.428, 470.0, OptionRight.Put, 60, -0.0920)]
        [TestCase(3.219, 430.0, OptionRight.Call, 60, -0.0705)]
        [TestCase(47.701, 430.0, OptionRight.Put, 60, -0.2843)]
        [TestCase(16.528, 450.0, OptionRight.Call, 180, -0.0583)]
        [TestCase(21.784, 450.0, OptionRight.Put, 180, -0.0481)]
        [TestCase(35.207, 470.0, OptionRight.Call, 180, -0.0715)]
        [TestCase(0.409, 470.0, OptionRight.Put, 180, -0.0028)]
        [TestCase(2.642, 430.0, OptionRight.Call, 180, -0.0265)]
        [TestCase(27.772, 430.0, OptionRight.Put, 180, -0.0294)]
        public void ComparesThetaOnBSMModel(decimal price, decimal spotPrice, OptionRight right, int expiry, double refTheta)
        {
            var symbol = Symbol.CreateOption("SPY", Market.USA, OptionStyle.American, right, 450m, _reference.AddDays(expiry));
            var indicator = new Theta(symbol, 0.053m, 0.0153m, optionModel: OptionPricingModelType.BlackScholes);

            var optionDataPoint = new IndicatorDataPoint(symbol, _reference, price);
            var spotDataPoint = new IndicatorDataPoint(symbol.Underlying, _reference, spotPrice);
            indicator.Update(optionDataPoint);
            indicator.Update(spotDataPoint);

            Assert.AreEqual(refTheta, (double)indicator.Current.Value, 0.0001d);
        }

        // Reference values from QuantLib
        [TestCase(23.753, 450.0, OptionRight.Call, 60, -0.2082)]
        [TestCase(35.830, 450.0, OptionRight.Put, 60, -0.2882)]
        [TestCase(33.928, 470.0, OptionRight.Call, 60, -0.1845)]
        [TestCase(6.428, 470.0, OptionRight.Put, 60, -0.0944)]
        [TestCase(3.219, 430.0, OptionRight.Call, 60, -0.0705)]
        [TestCase(47.701, 430.0, OptionRight.Put, 60, -0.2899)]
        [TestCase(16.528, 450.0, OptionRight.Call, 180, -0.0584)]
        [TestCase(21.784, 450.0, OptionRight.Put, 180, -0.0536)]
        [TestCase(35.207, 470.0, OptionRight.Call, 180, -0.0716)]
        [TestCase(0.409, 470.0, OptionRight.Put, 180, -0.0035)]
        [TestCase(2.642, 430.0, OptionRight.Call, 180, -0.0265)]
        [TestCase(27.772, 430.0, OptionRight.Put, 180, -0.0369)]
        public void ComparesThetaOnCRRModel(decimal price, decimal spotPrice, OptionRight right, int expiry, double refTheta)
        {
            var symbol = Symbol.CreateOption("SPY", Market.USA, OptionStyle.American, right, 450m, _reference.AddDays(expiry));
            var indicator = new Theta(symbol, 0.053m, 0.0153m, optionModel: OptionPricingModelType.BinomialCoxRossRubinstein,
                ivModel: OptionPricingModelType.BlackScholes);

            var optionDataPoint = new IndicatorDataPoint(symbol, _reference, price);
            var spotDataPoint = new IndicatorDataPoint(symbol.Underlying, _reference, spotPrice);
            indicator.Update(optionDataPoint);
            indicator.Update(spotDataPoint);

            Assert.AreEqual(refTheta, (double)indicator.Current.Value, 0.005d);
        }
    }
}
