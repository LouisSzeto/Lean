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

using System;
using MathNet.Numerics.Distributions;
using Python.Runtime;
using QuantConnect.Data;

namespace QuantConnect.Indicators
{
    /// <summary>
    /// Option Gamma indicator that calculate the gamma of an option
    /// </summary>
    /// <remarks>derivative of option price change relative to $1 underlying changes</remarks>
    public class OptionGamma : OptionGreeksIndicatorBase
    {
        /// <summary>
        /// Initializes a new instance of the OptionGamma class
        /// </summary>
        /// <param name="name">The name of this indicator</param>
        /// <param name="option">The option to be tracked</param>
        /// <param name="riskFreeRateModel">Risk-free rate model</param>
        /// <param name="optionModel">The option pricing model used to estimate Gamma</param>
        /// <param name="ivModel">The option pricing model used to estimate IV</param>
        public OptionGamma(string name, Symbol option, IRiskFreeInterestRateModel riskFreeRateModel,
                OptionPricingModelType optionModel = OptionPricingModelType.BlackScholes, OptionPricingModelType? ivModel = null)
            : base(name, option, riskFreeRateModel, optionModel: optionModel, ivModel: ivModel)
        {
        }

        /// <summary>
        /// Initializes a new instance of the OptionGamma class
        /// </summary>
        /// <param name="option">The option to be tracked</param>
        /// <param name="riskFreeRateModel">Risk-free rate model</param>
        /// <param name="optionModel">The option pricing model used to estimate Gamma</param>
        /// <param name="ivModel">The option pricing model used to estimate IV</param>
        public OptionGamma(Symbol option, IRiskFreeInterestRateModel riskFreeRateModel,
            OptionPricingModelType optionModel = OptionPricingModelType.BlackScholes, OptionPricingModelType? ivModel = null)
            : this($"Gamma({optionModel})", option, riskFreeRateModel, optionModel, ivModel)
        {
        }

        /// <summary>
        /// Initializes a new instance of the OptionGamma class
        /// </summary>
        /// <param name="name">The name of this indicator</param>
        /// <param name="option">The option to be tracked</param>
        /// <param name="riskFreeRateModel">Risk-free rate model</param>
        /// <param name="optionModel">The option pricing model used to estimate Gamma</param>
        /// <param name="ivModel">The option pricing model used to estimate IV</param>
        public OptionGamma(string name, Symbol option, PyObject riskFreeRateModel,
            OptionPricingModelType optionModel = OptionPricingModelType.BlackScholes, OptionPricingModelType? ivModel = null)
            : base(name, option, riskFreeRateModel, optionModel: optionModel, ivModel: ivModel)
        {
        }

        /// <summary>
        /// Initializes a new instance of the OptionGamma class
        /// </summary>
        /// <param name="option">The option to be tracked</param>
        /// <param name="riskFreeRateModel">Risk-free rate model</param>
        /// <param name="optionModel">The option pricing model used to estimate Gamma</param>
        /// <param name="ivModel">The option pricing model used to estimate IV</param>
        public OptionGamma(Symbol option, PyObject riskFreeRateModel, OptionPricingModelType optionModel = OptionPricingModelType.BlackScholes,
            OptionPricingModelType? ivModel = null)
            : this($"Gamma({optionModel})", option, riskFreeRateModel, optionModel, ivModel)
        {
        }

        /// <summary>
        /// Initializes a new instance of the OptionGamma class
        /// </summary>
        /// <param name="option">The option to be tracked</param>am>
        /// <param name="riskFreeRate">Risk-free rate, as a constant</param>
        /// <param name="optionModel">The option pricing model used to estimate Gamma</param>
        /// <param name="ivModel">The option pricing model used to estimate IV</param>
        public OptionGamma(string name, Symbol option, decimal riskFreeRate = 0.05m,
            OptionPricingModelType optionModel = OptionPricingModelType.BlackScholes, OptionPricingModelType? ivModel = null)
            : base(name, option, riskFreeRate, optionModel: optionModel, ivModel: ivModel)
        {
        }

        /// <summary>
        /// Initializes a new instance of the OptionGamma class
        /// </summary>
        /// <param name="option">The option to be tracked</param>
        /// <param name="riskFreeRate">Risk-free rate, as a constant</param>
        /// <param name="optionModel">The option pricing model used to estimate Gamma</param>
        /// <param name="ivModel">The option pricing model used to estimate IV</param>
        public OptionGamma(Symbol option, decimal riskFreeRate = 0.05m, OptionPricingModelType optionModel = OptionPricingModelType.BlackScholes,
            OptionPricingModelType? ivModel = null)
            : this($"Gamma({optionModel})", option, riskFreeRate, optionModel, ivModel)
        {
        }

        // Calculate the theoretical option gamma
        private decimal TheoreticalGamma(decimal spotPrice, decimal timeToExpiration, decimal volatility, 
            OptionPricingModelType optionModel = OptionPricingModelType.BlackScholes)
        {
            var math = OptionGreekIndicatorsHelper.DecimalMath;
                
            switch (optionModel)
            {
                case OptionPricingModelType.BinomialCoxRossRubinstein:
                    var upFactor = math(Math.Exp, volatility * math(Math.Sqrt, timeToExpiration / 200));
                    if (upFactor == 1)
                    {
                        // provide a small step to estimate gamma
                        upFactor = 1.0001m;
                    }

                    // Finite differncing approach
                    var sU = spotPrice * upFactor * upFactor;
                    var sD = spotPrice / upFactor / upFactor;

                    var fU = OptionGreekIndicatorsHelper.CRRTheoreticalPrice(volatility, sU, Strike, timeToExpiration, RiskFreeRate, Right);
                    var fM = OptionGreekIndicatorsHelper.CRRTheoreticalPrice(volatility, spotPrice, Strike, timeToExpiration, RiskFreeRate, Right);
                    var fD = OptionGreekIndicatorsHelper.CRRTheoreticalPrice(volatility, sD, Strike, timeToExpiration, RiskFreeRate, Right);

                    var deltaU = (fU - fM) / (sU - spotPrice);
                    var deltaD = (fM - fD) / (spotPrice - sD);

                    return (deltaU - deltaD) * 2 / (sU - sD);

                case OptionPricingModelType.BlackScholes:
                default:
                    var norm = new Normal();
                    var d1 = OptionGreekIndicatorsHelper.CalculateD1(spotPrice, Strike, timeToExpiration, RiskFreeRate, volatility);

                    // allow at least 1% IV
                    return math(norm.Density, -d1) / spotPrice / Math.Max(volatility, 0.01m) / math(Math.Sqrt, timeToExpiration);
            }
        }

        // Calculate the Gamma of the option
        protected override decimal CalculateGreek(DateTime time)
        {
            var spotPrice = UnderlyingPrice.Current.Value;
            var timeToExpiration = Convert.ToDecimal((Expiry - time).TotalDays) / 365m;
            var volatility = ImpliedVolatility.Current.Value;

            return TheoreticalGamma(spotPrice, timeToExpiration, volatility, _optionModel);
        }
    }
}
