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
using MathNet.Numerics.RootFinding;
using Python.Runtime;
using QuantConnect.Data;
using QuantConnect.Data.Consolidators;
using QuantConnect.Logging;
using QuantConnect.Python;
using QuantConnect.Util;

namespace QuantConnect.Indicators
{
    /// <summary>
    /// Implied Volatility indicator that calculate the IV of an option using Black-Scholes Model
    /// </summary>
    public class ImpliedVolatilityCustom : ImpliedVolatility
    {
        /// <summary>
        /// Initializes a new instance of the ImpliedVolatility class
        /// </summary>
        /// <param name="option">The option to be tracked</param>
        /// <param name="riskFreeRate">Risk-free rate, as a constant</param>
        /// <param name="dividendYield">Dividend yield, as a constant</param>
        /// <param name="mirrorOption">The mirror option for parity calculation</param>
        /// <param name="optionModel">The option pricing model used to estimate IV</param>
        /// <param name="period">The lookback period of historical volatility</param>
        public ImpliedVolatilityCustom(Symbol option, decimal riskFreeRate = 0.05m, decimal dividendYield = 0.0m, Symbol mirrorOption = null,
            OptionPricingModelType optionModel = OptionPricingModelType.BlackScholes, int period = 252)
            : base(option, riskFreeRate, dividendYield, mirrorOption, optionModel, period)
        {
            SetSmoothingFunction((iv, mirrorIV) => iv);
        }
        
        /// <summary>
        /// Computes the IV of the option
        /// </summary>
        /// <param name="timeTillExpiry">the time until expiration in years</param>
        /// <returns>Smoothened IV of the option</returns>
        protected override decimal CalculateIV(decimal timeTillExpiry)
        {
            var impliedVol = 0m;
            try
            {
                Func<double, double> f = (vol) => (double)(TheoreticalPrice(
                    Convert.ToDecimal(vol), UnderlyingPrice, Strike, timeTillExpiry, RiskFreeRate, DividendYield, Right, _optionModel)
                    + TheoreticalPrice(
                        Convert.ToDecimal(vol), UnderlyingPrice, Strike, timeTillExpiry, RiskFreeRate, DividendYield, _oppositeOptionSymbol.ID.OptionRight, _optionModel)
                    - Price - OppositePrice);
                impliedVol = Convert.ToDecimal(Brent.FindRoot(f, 1e-7d, 2.0d, 1e-4d, 100));
            }
            catch
            {
                Log.Error("ImpliedVolatility.CalculateIV(): Fail to converge, returning 0.");
            }

            return impliedVol;
        }
    }
}
