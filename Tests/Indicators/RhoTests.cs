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

namespace QuantConnect.Tests.Indicators
{
    [TestFixture]
    public class RhoTests : OptionBaseIndicatorTests<Rho>
    {
        protected override IndicatorBase<IndicatorDataPoint> CreateIndicator()
            => new Rho("testRhoIndicator", _symbol, 0.053m, 0.0153m);

        protected override OptionIndicatorBase CreateIndicator(IRiskFreeInterestRateModel riskFreeRateModel)
            => new Rho("testRhoIndicator", _symbol, riskFreeRateModel);

        protected override OptionIndicatorBase CreateIndicator(IRiskFreeInterestRateModel riskFreeRateModel, IDividendYieldModel dividendYieldModel)
            => new Rho("testRhoIndicator", _symbol, riskFreeRateModel, dividendYieldModel);

        protected override OptionIndicatorBase CreateIndicator(QCAlgorithm algorithm)
            => algorithm.R(_symbol);

        [SetUp]
        public void SetUp()
        {
            RiskFreeRateUpdatesPerIteration = 3;
            DividendYieldUpdatesPerIteration = 3;
        }

        // No Rho value provided by IB API

        // Reference values from QuantLib
        [TestCase(23.753, 450.0, OptionRight.Call, 60, 0.3628)]
        [TestCase(35.830, 450.0, OptionRight.Put, 60, -0.3885)]
        [TestCase(33.928, 470.0, OptionRight.Call, 60, 0.4761)]
        [TestCase(6.428, 470.0, OptionRight.Put, 60, -0.2119)]
        [TestCase(3.219, 430.0, OptionRight.Call, 60, 0.1652)]
        [TestCase(47.701, 430.0, OptionRight.Put, 60, -0.4498)]
        [TestCase(16.528, 450.0, OptionRight.Call, 180, 1.2862)]
        [TestCase(21.784, 450.0, OptionRight.Put, 180, -1.0337)]
        [TestCase(35.207, 470.0, OptionRight.Call, 180, 1.5558)]
        [TestCase(0.409, 470.0, OptionRight.Put, 180, -0.1235)]
        [TestCase(2.642, 430.0, OptionRight.Call, 180, 0.5326)]
        [TestCase(27.772, 430.0, OptionRight.Put, 180, -1.3178)]
        public void ComparesRhoOnBSMModel(decimal price, decimal spotPrice, OptionRight right, int expiry, double refRho)
        {
            var symbol = Symbol.CreateOption("SPY", Market.USA, OptionStyle.American, right, 450m, _reference.AddDays(expiry));
            var indicator = new Rho(symbol, 0.053m, 0.0153m, optionModel: OptionPricingModelType.BlackScholes);

            var optionDataPoint = new IndicatorDataPoint(symbol, _reference, price);
            var spotDataPoint = new IndicatorDataPoint(symbol.Underlying, _reference, spotPrice);
            indicator.Update(optionDataPoint);
            indicator.Update(spotDataPoint);

            Assert.AreEqual(refRho, (double)indicator.Current.Value, 0.0001d);
        }

        // Reference values from QuantLib
        [TestCase(23.753, 450.0, OptionRight.Call, 60, 0.3628)]
        [TestCase(35.830, 450.0, OptionRight.Put, 60, -0.3253)]
        [TestCase(33.928, 470.0, OptionRight.Call, 60, 0.4761)]
        [TestCase(6.428, 470.0, OptionRight.Put, 60, -0.1869)]
        [TestCase(3.219, 430.0, OptionRight.Call, 60, 0.1648)]
        [TestCase(47.701, 430.0, OptionRight.Put, 60, -0.3643)]
        [TestCase(16.528, 450.0, OptionRight.Call, 180, 1.2861)]
        [TestCase(21.784, 450.0, OptionRight.Put, 180, -0.7927)]
        [TestCase(35.207, 470.0, OptionRight.Call, 180, 1.5558)]
        [TestCase(0.409, 470.0, OptionRight.Put, 180, -0.1173)]
        [TestCase(2.642, 430.0, OptionRight.Call, 180, 0.5306)]
        [TestCase(27.772, 430.0, OptionRight.Put, 180, -0.8524)]
        public void ComparesRhoOnCRRModel(decimal price, decimal spotPrice, OptionRight right, int expiry, double refRho)
        {
            var symbol = Symbol.CreateOption("SPY", Market.USA, OptionStyle.American, right, 450m, _reference.AddDays(expiry));
            var indicator = new Rho(symbol, 0.053m, 0.0153m, optionModel: OptionPricingModelType.BinomialCoxRossRubinstein,
                ivModel: OptionPricingModelType.BlackScholes);

            var optionDataPoint = new IndicatorDataPoint(symbol, _reference, price);
            var spotDataPoint = new IndicatorDataPoint(symbol.Underlying, _reference, spotPrice);
            indicator.Update(optionDataPoint);
            indicator.Update(spotDataPoint);

            Assert.AreEqual(refRho, (double)indicator.Current.Value, 0.005d);
        }
    }
}
