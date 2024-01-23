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
    /// Option Vega indicator that calculate the vega of an option
    /// </summary>
    /// <remarks>derivative of option price change relative to $1 underlying changes</remarks>
    public class Vega : OptionGreeksIndicatorBase
    {
        /// <summary>
        /// Initializes a new instance of the Vega class
        /// </summary>
        /// <param name="name">The name of this indicator</param>
        /// <param name="option">The option to be tracked</param>
        /// <param name="riskFreeRateModel">Risk-free rate model</param>
        /// <param name="optionModel">The option pricing model used to estimate Vega</param>
        /// <param name="ivModel">The option pricing model used to estimate IV</param>
        public Vega(string name, Symbol option, IRiskFreeInterestRateModel riskFreeRateModel,
                OptionPricingModelType optionModel = OptionPricingModelType.BlackScholes, OptionPricingModelType? ivModel = null)
            : base(name, option, riskFreeRateModel, optionModel: optionModel, ivModel: ivModel)
        {
        }

        /// <summary>
        /// Initializes a new instance of the Vega class
        /// </summary>
        /// <param name="option">The option to be tracked</param>
        /// <param name="riskFreeRateModel">Risk-free rate model</param>
        /// <param name="optionModel">The option pricing model used to estimate Vega</param>
        /// <param name="ivModel">The option pricing model used to estimate IV</param>
        public Vega(Symbol option, IRiskFreeInterestRateModel riskFreeRateModel,
            OptionPricingModelType optionModel = OptionPricingModelType.BlackScholes, OptionPricingModelType? ivModel = null)
            : this($"Vega({optionModel})", option, riskFreeRateModel, optionModel, ivModel)
        {
        }

        /// <summary>
        /// Initializes a new instance of the Vega class
        /// </summary>
        /// <param name="name">The name of this indicator</param>
        /// <param name="option">The option to be tracked</param>
        /// <param name="riskFreeRateModel">Risk-free rate model</param>
        /// <param name="optionModel">The option pricing model used to estimate Vega</param>
        /// <param name="ivModel">The option pricing model used to estimate IV</param>
        public Vega(string name, Symbol option, PyObject riskFreeRateModel,
            OptionPricingModelType optionModel = OptionPricingModelType.BlackScholes, OptionPricingModelType? ivModel = null)
            : base(name, option, riskFreeRateModel, optionModel: optionModel, ivModel: ivModel)
        {
        }

        /// <summary>
        /// Initializes a new instance of the Vega class
        /// </summary>
        /// <param name="option">The option to be tracked</param>
        /// <param name="riskFreeRateModel">Risk-free rate model</param>
        /// <param name="optionModel">The option pricing model used to estimate Vega</param>
        /// <param name="ivModel">The option pricing model used to estimate IV</param>
        public Vega(Symbol option, PyObject riskFreeRateModel, OptionPricingModelType optionModel = OptionPricingModelType.BlackScholes,
            OptionPricingModelType? ivModel = null)
            : this($"Vega({optionModel})", option, riskFreeRateModel, optionModel, ivModel)
        {
        }

        /// <summary>
        /// Initializes a new instance of the Vega class
        /// </summary>
        /// <param name="option">The option to be tracked</param>am>
        /// <param name="riskFreeRate">Risk-free rate, as a constant</param>
        /// <param name="optionModel">The option pricing model used to estimate Vega</param>
        /// <param name="ivModel">The option pricing model used to estimate IV</param>
        public Vega(string name, Symbol option, decimal riskFreeRate = 0.05m,
            OptionPricingModelType optionModel = OptionPricingModelType.BlackScholes, OptionPricingModelType? ivModel = null)
            : base(name, option, riskFreeRate, optionModel: optionModel, ivModel: ivModel)
        {
        }

        /// <summary>
        /// Initializes a new instance of the Vega class
        /// </summary>
        /// <param name="option">The option to be tracked</param>
        /// <param name="riskFreeRate">Risk-free rate, as a constant</param>
        /// <param name="optionModel">The option pricing model used to estimate Vega</param>
        /// <param name="ivModel">The option pricing model used to estimate IV</param>
        public Vega(Symbol option, decimal riskFreeRate = 0.05m, OptionPricingModelType optionModel = OptionPricingModelType.BlackScholes,
            OptionPricingModelType? ivModel = null)
            : this($"Vega({optionModel})", option, riskFreeRate, optionModel, ivModel)
        {
        }

        // Calculate the theoretical option vega
        private decimal TheoreticalVega(decimal spotPrice, decimal timeToExpiration, decimal volatility, 
            OptionPricingModelType optionModel = OptionPricingModelType.BlackScholes)
        {
            var math = OptionGreekIndicatorsHelper.DecimalMath;
                
            switch (optionModel)
            {
                case OptionPricingModelType.BinomialCoxRossRubinstein:
                    // finite differencing method with 0.01% IV changes
                    var deltaSigma = 0.0001m;

                    var newPrice = OptionGreekIndicatorsHelper.CRRTheoreticalPrice(volatility + deltaSigma, spotPrice, Strike, timeToExpiration, RiskFreeRate, Right);
                    var price = OptionGreekIndicatorsHelper.CRRTheoreticalPrice(volatility, spotPrice, Strike, timeToExpiration, RiskFreeRate, Right);

                    return (newPrice - price) / deltaSigma / 100;

                case OptionPricingModelType.BlackScholes:
                default:
                    var norm = new Normal();
                    var d1 = OptionGreekIndicatorsHelper.CalculateD1(spotPrice, Strike, timeToExpiration, RiskFreeRate, volatility);

                    return spotPrice * math(Math.Sqrt, timeToExpiration) * math(norm.Density, d1) / 100;
            }
        }

        // Calculate the Vega of the option
        protected override decimal CalculateGreek(DateTime time)
        {
            var spotPrice = UnderlyingPrice.Current.Value;
            var timeToExpiration = Convert.ToDecimal((Expiry - time).TotalDays) / 365m;
            var volatility = ImpliedVolatility.Current.Value;

            return TheoreticalVega(spotPrice, timeToExpiration, volatility, _optionModel);
        }
    }
}
