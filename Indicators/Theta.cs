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
    /// Option Theta indicator that calculate the theta of an option
    /// </summary>
    /// <remarks>derivative of option price change relative to $1 underlying changes</remarks>
    public class Theta : OptionGreeksIndicatorBase
    {
        /// <summary>
        /// Initializes a new instance of the Theta class
        /// </summary>
        /// <param name="name">The name of this indicator</param>
        /// <param name="option">The option to be tracked</param>
        /// <param name="riskFreeRateModel">Risk-free rate model</param>
        /// <param name="optionModel">The option pricing model used to estimate Theta</param>
        /// <param name="ivModel">The option pricing model used to estimate IV</param>
        public Theta(string name, Symbol option, IRiskFreeInterestRateModel riskFreeRateModel,
                OptionPricingModelType optionModel = OptionPricingModelType.BlackScholes, OptionPricingModelType? ivModel = null)
            : base(name, option, riskFreeRateModel, optionModel: optionModel, ivModel: ivModel)
        {
        }

        /// <summary>
        /// Initializes a new instance of the Theta class
        /// </summary>
        /// <param name="option">The option to be tracked</param>
        /// <param name="riskFreeRateModel">Risk-free rate model</param>
        /// <param name="optionModel">The option pricing model used to estimate Theta</param>
        /// <param name="ivModel">The option pricing model used to estimate IV</param>
        public Theta(Symbol option, IRiskFreeInterestRateModel riskFreeRateModel,
            OptionPricingModelType optionModel = OptionPricingModelType.BlackScholes, OptionPricingModelType? ivModel = null)
            : this($"Theta({optionModel})", option, riskFreeRateModel, optionModel, ivModel)
        {
        }

        /// <summary>
        /// Initializes a new instance of the Theta class
        /// </summary>
        /// <param name="name">The name of this indicator</param>
        /// <param name="option">The option to be tracked</param>
        /// <param name="riskFreeRateModel">Risk-free rate model</param>
        /// <param name="optionModel">The option pricing model used to estimate Theta</param>
        /// <param name="ivModel">The option pricing model used to estimate IV</param>
        public Theta(string name, Symbol option, PyObject riskFreeRateModel,
            OptionPricingModelType optionModel = OptionPricingModelType.BlackScholes, OptionPricingModelType? ivModel = null)
            : base(name, option, riskFreeRateModel, optionModel: optionModel, ivModel: ivModel)
        {
        }

        /// <summary>
        /// Initializes a new instance of the Theta class
        /// </summary>
        /// <param name="option">The option to be tracked</param>
        /// <param name="riskFreeRateModel">Risk-free rate model</param>
        /// <param name="optionModel">The option pricing model used to estimate Theta</param>
        /// <param name="ivModel">The option pricing model used to estimate IV</param>
        public Theta(Symbol option, PyObject riskFreeRateModel, OptionPricingModelType optionModel = OptionPricingModelType.BlackScholes,
            OptionPricingModelType? ivModel = null)
            : this($"Theta({optionModel})", option, riskFreeRateModel, optionModel, ivModel)
        {
        }

        /// <summary>
        /// Initializes a new instance of the Theta class
        /// </summary>
        /// <param name="option">The option to be tracked</param>am>
        /// <param name="riskFreeRate">Risk-free rate, as a constant</param>
        /// <param name="optionModel">The option pricing model used to estimate Theta</param>
        /// <param name="ivModel">The option pricing model used to estimate IV</param>
        public Theta(string name, Symbol option, decimal riskFreeRate = 0.05m,
            OptionPricingModelType optionModel = OptionPricingModelType.BlackScholes, OptionPricingModelType? ivModel = null)
            : base(name, option, riskFreeRate, optionModel: optionModel, ivModel: ivModel)
        {
        }

        /// <summary>
        /// Initializes a new instance of the Theta class
        /// </summary>
        /// <param name="option">The option to be tracked</param>
        /// <param name="riskFreeRate">Risk-free rate, as a constant</param>
        /// <param name="optionModel">The option pricing model used to estimate Theta</param>
        /// <param name="ivModel">The option pricing model used to estimate IV</param>
        public Theta(Symbol option, decimal riskFreeRate = 0.05m, OptionPricingModelType optionModel = OptionPricingModelType.BlackScholes,
            OptionPricingModelType? ivModel = null)
            : this($"Theta({optionModel})", option, riskFreeRate, optionModel, ivModel)
        {
        }

        // Calculate the theoretical option theta
        private decimal TheoreticalTheta(decimal spotPrice, decimal timeToExpiration, decimal volatility, 
            OptionPricingModelType optionModel = OptionPricingModelType.BlackScholes)
        {
            var math = OptionGreekIndicatorsHelper.DecimalMath;
                
            switch (optionModel)
            {
                case OptionPricingModelType.BinomialCoxRossRubinstein:
                    var deltaTime = timeToExpiration / 200;

                    var forwardPrice = OptionGreekIndicatorsHelper.CRRTheoreticalPrice(volatility, spotPrice, Strike, timeToExpiration - 2 * deltaTime, RiskFreeRate, Right);
                    var price = OptionGreekIndicatorsHelper.CRRTheoreticalPrice(volatility, spotPrice, Strike, timeToExpiration, RiskFreeRate, Right);

                    return (forwardPrice - price) * 0.5m / deltaTime / 365m;

                case OptionPricingModelType.BlackScholes:
                default:
                    var norm = new Normal();
                    var d1 = OptionGreekIndicatorsHelper.CalculateD1(spotPrice, Strike, timeToExpiration, RiskFreeRate, volatility);
                    var d2 = OptionGreekIndicatorsHelper.CalculateD2(d1, volatility, timeToExpiration);
                    var discount = math(Math.Exp, -RiskFreeRate * timeToExpiration);
                    // allow at least 1% IV
                    volatility = Math.Max(volatility, 0.01m);

                    var theta = -spotPrice * volatility * math(norm.Density, d1) * 0.5m / math(Math.Sqrt, timeToExpiration);

                    if (Right == OptionRight.Call)
                    {
                        theta -= RiskFreeRate * Strike * discount * math(norm.CumulativeDistribution, d2);
                    }
                    else
                    {
                        theta += RiskFreeRate * Strike * discount * math(norm.CumulativeDistribution, -d2);
                    }
                    return theta / 365m;
            }
        }

        // Calculate the Theta of the option
        protected override decimal CalculateGreek(DateTime time)
        {
            var spotPrice = UnderlyingPrice.Current.Value;
            var timeToExpiration = Convert.ToDecimal((Expiry - time).TotalDays) / 365m;
            var volatility = ImpliedVolatility.Current.Value;

            return TheoreticalTheta(spotPrice, timeToExpiration, volatility, _optionModel);
        }
    }
}
