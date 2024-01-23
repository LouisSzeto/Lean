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
            => new Rho("testRhoIndicator", _symbol, 0.04m);

        protected override OptionIndicatorBase CreateIndicator(IRiskFreeInterestRateModel riskFreeRateModel)
            => new Rho("testRhoIndicator", _symbol, riskFreeRateModel);

        protected override OptionIndicatorBase CreateIndicator(QCAlgorithm algorithm)
            => algorithm.Rho(_symbol);

        [SetUp]
        public void SetUp()
        {
            RiskFreeRateUpdatesPerIteration = 3;
        }

        // IB does not provide option Rho data

        // Reference values from QuantLib
        [TestCase(23.753, 450.0, OptionRight.Call, 60, 0.3647)]
        [TestCase(35.830, 450.0, OptionRight.Put, 60, -0.3888)]
        [TestCase(33.928, 470.0, OptionRight.Call, 60, 0.4795)]
        [TestCase(6.428, 470.0, OptionRight.Put, 60, -0.2116)]
        [TestCase(3.219, 430.0, OptionRight.Call, 60, 0.1661)]
        [TestCase(47.701, 430.0, OptionRight.Put, 60, -0.4503)]
        [TestCase(16.528, 450.0, OptionRight.Call, 180, 1.3190)]
        [TestCase(21.784, 450.0, OptionRight.Put, 180, -1.0341)]
        [TestCase(35.207, 470.0, OptionRight.Call, 180, 1.5969)]
        [TestCase(0.409, 470.0, OptionRight.Put, 180, -0.1222)]
        [TestCase(2.642, 430.0, OptionRight.Call, 180, 0.5442)]
        [TestCase(27.772, 430.0, OptionRight.Put, 180, -1.3181)]
        public void ComparesRhoOnBSMModel(decimal price, decimal spotPrice, OptionRight right, int expiry, double refRho)
        {
            var symbol = Symbol.CreateOption("SPY", Market.USA, OptionStyle.American, right, 450m, _reference.AddDays(expiry));
            var indicator = new Rho(symbol, 0.04m, optionModel: OptionPricingModelType.BlackScholes);

            var optionDataPoint = new IndicatorDataPoint(symbol, _reference, price);
            var spotDataPoint = new IndicatorDataPoint(symbol.Underlying, _reference, spotPrice);
            indicator.Update(optionDataPoint);
            indicator.Update(spotDataPoint);

            Assert.AreEqual(refRho, (double)indicator.Current.Value, 0.001d);
        }

        // Reference values from WolframAlpha
        [TestCase(23.57, 450.0, OptionRight.Call, 60, 0.3712)]
        [TestCase(20.71, 450.0, OptionRight.Put, 60, -0.3092)]
        [TestCase(35.82, 470.0, OptionRight.Call, 60, 0.4750)]
        [TestCase(12.86, 470.0, OptionRight.Put, 60, -0.2360)]
        [TestCase(14.03, 430.0, OptionRight.Call, 60, 0.2631)]
        [TestCase(31.40, 430.0, OptionRight.Put, 60, -0.3604)]
        [TestCase(42.26, 450.0, OptionRight.Call, 180, 1.0819)]
        [TestCase(34.05, 450.0, OptionRight.Put, 180, -0.8701)]
        [TestCase(54.51, 470.0, OptionRight.Call, 180, 1.2610)]
        [TestCase(26.13, 470.0, OptionRight.Put, 180, -0.7622)]
        [TestCase(31.42, 430.0, OptionRight.Call, 180, 0.8964)]
        [TestCase(43.58, 430.0, OptionRight.Put, 180, -0.9480)]
        public void ComparesRhoOnAmericanOptions(decimal price, decimal spotPrice, OptionRight right, int expiry, double refRho)
        {
            var symbol = Symbol.CreateOption("SPY", Market.USA, OptionStyle.American, right, 450m, _reference.AddDays(expiry));
            var indicator = new Rho(symbol, 0.04m, optionModel: OptionPricingModelType.BlackScholes);

            var optionDataPoint = new IndicatorDataPoint(symbol, _reference, price);
            var spotDataPoint = new IndicatorDataPoint(symbol.Underlying, _reference, spotPrice);
            indicator.Update(optionDataPoint);
            indicator.Update(spotDataPoint);

            Assert.AreEqual(refRho, (double)indicator.Current.Value, 0.001d);
        }

        // QuantLib/WolframAlpha does not provide rho for binomial tree model
    }
}
