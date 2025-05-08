#ifndef DECISION_LOGIC_HPP
#define DECISION_LOGIC_HPP

#include <vector>
#include <string>
#include <iostream>
#include "indicators_rectified.hpp"

enum class Signal {
    BUY,
    SELL,
    HOLD
};

struct Decision {
    std::string date;
    Signal signal;
};

std::string signalToString(Signal signal) {
    switch (signal) {
        case Signal::BUY: return "BUY";
        case Signal::SELL: return "SELL";
        default: return "HOLD";
    }
}

// std::vector<Decision> generateDecisions(
//     const std::vector<std::string>& dates,
//     const std::vector<float>& close,
//     const std::vector<float>& high,
//     const std::vector<float>& low,
//     const std::vector<float>& volume,
//     bool use_gpu = false
// ) {
//     std::vector<Decision> decisions;

//     if (dates.empty() || close.empty()) {
//         std::cerr << "Error: Empty input data in generateDecisions" << std::endl;
//         return decisions;
//     }

//     std::cout << "Input data sizes - dates: " << dates.size()
//               << ", close: " << close.size()
//               << ", high: " << high.size()
//               << ", low: " << low.size()
//               << ", volume: " << volume.size() << std::endl;

//     int period = 14;

//     if (close.size() <= period) {
//         std::cerr << "Error: Not enough data points for technical analysis (need more than "
//                   << period << " points)" << std::endl;
//         return decisions;
//     }

//     std::vector<float> sma = computeSMA(close, period);
//     std::vector<float> ema = computeEMA(close, period);
//     std::vector<float> rsi = computeRSI(close, period);
//     std::vector<float> macd = computeMACD(close);
//     BollingerBands bb = computeBollingerBands(close);
//     std::vector<float> stoch = computeStochasticOscillator(close, high, low);

//     std::cout << "Indicator sizes - sma: " << sma.size()
//               << ", ema: " << ema.size()
//               << ", rsi: " << rsi.size()
//               << ", macd: " << macd.size()
//               << ", bb.middle: " << bb.middle.size()
//               << ", stoch: " << stoch.size() << std::endl;

//     size_t min_size = std::min({
//         sma.size(), ema.size(), rsi.size(),
//         macd.size(), bb.middle.size(), stoch.size()
//     });

//     std::cout << "Min indicator size: " << min_size << std::endl;

//     if (min_size == 0) {
//         std::cerr << "Error: One or more indicators have size 0" << std::endl;
//         return decisions;
//     }

//     int idx_offset = dates.size() - min_size;

//     if (idx_offset < 0 || idx_offset >= dates.size()) {
//         std::cerr << "Error: Invalid index offset: " << idx_offset << std::endl;
//         return decisions;
//     }

//     for (size_t i = 1; i < min_size; ++i) {
//         size_t date_idx = i + idx_offset;

//         if (date_idx >= dates.size() || i >= rsi.size() || i >= macd.size() || 
//             i >= stoch.size() || i >= bb.upper.size() || i >= ema.size()) {
//             std::cerr << "Index out of bounds at i=" << i << ", date_idx=" << date_idx << std::endl;
//             continue;
//         }

//         float c = close[date_idx];
//         float r = rsi[i];
//         float m = macd[i];
//         float s = stoch[i];
//         float e = ema[i];
//         float e_prev = ema[i - 1];
//         float m_prev = macd[i - 1];
//         float s_prev = stoch[i - 1];
//         float b_upper = bb.upper[i];
//         float b_lower = bb.lower[i];

//         // Scoring model
//         int score = 0;

//         // BUY-like indicators
//         if (c > e && close[date_idx - 1] <= e_prev) score++; // EMA crossover up
//         if (r < 30) score++;
//         if (m_prev < 0 && m > 0) score++; // MACD crossover up
//         if (c < b_lower) score++;
//         if (s_prev < 20 && s > s_prev) score++;

//         // SELL-like indicators
//         if (c < e && close[date_idx - 1] >= e_prev) score--;
//         if (r > 70) score--;
//         if (m_prev > 0 && m < 0) score--; // MACD crossover down
//         if (c > b_upper) score--;
//         if (s_prev > 80 && s < s_prev) score--;

//         Signal signal;
//         if (score >= 3)
//             signal = Signal::BUY;
//         else if (score <= -3)
//             signal = Signal::SELL;
//         else
//             signal = Signal::HOLD;

//         decisions.push_back({dates[date_idx], signal});

//         if (i == 1 || i == min_size - 1) {
//             std::cout << "Signal @ " << dates[date_idx] << " — "
//                       << "Close: " << c << ", EMA: " << e
//                       << ", RSI: " << r << ", MACD: " << m
//                       << ", BB Upper: " << b_upper << ", BB Lower: " << b_lower
//                       << ", Stoch: " << s << ", Score: " << score
//                       << ", Decision: " << signalToString(signal) << std::endl;
//         }
//     }

//     std::cout << "Generated " << decisions.size() << " decisions" << std::endl;
//     return decisions;
// }

// std::vector<Decision> generateDecisions(
//     const std::vector<std::string>& dates,
//     const std::vector<float>& close,
//     const std::vector<float>& high,
//     const std::vector<float>& low,
//     const std::vector<float>& volume,
//     bool use_gpu = false
// ) {
//     std::vector<Decision> decisions;

//     const int period = 14;

//     if (dates.size() <= period || close.size() <= period) {
//         std::cerr << "Not enough data for trend analysis." << std::endl;
//         return decisions;
//     }

//     std::vector<float> sma = computeSMA(close, period);
//     std::vector<float> ema = computeEMA(close, period);
//     std::vector<float> rsi = computeRSI(close, period);
//     std::vector<float> macd = computeMACD(close);
//     BollingerBands bb = computeBollingerBands(close);
//     std::vector<float> stoch = computeStochasticOscillator(close, high, low);

//     size_t usable_start = period; // Start after initial indicator warm-up
//     size_t usable_end = close.size();

//     for (size_t i = usable_start; i < usable_end; ++i) {
//         float c = close[i];
//         float r = rsi[i - period];     // aligned to indicator size
//         float m = macd[i - period];
//         float e = ema[i - period];
//         float s = stoch[i - period];
//         float b_upper = bb.upper[i - period];
//         float b_lower = bb.lower[i - period];

//         int score = 0;

//         // BUY-like signals
//         if (c > e && close[i - 1] <= ema[i - period - 1]) score++; // EMA cross up
//         if (r < 30) score++;
//         if (macd[i - period - 1] < 0 && m > 0) score++; // MACD cross up
//         if (c < b_lower) score++;
//         if (s < 20) score++;

//         // SELL-like signals
//         if (c < e && close[i - 1] >= ema[i - period - 1]) score--;
//         if (r > 70) score--;
//         if (macd[i - period - 1] > 0 && m < 0) score--; // MACD cross down
//         if (c > b_upper) score--;
//         if (s > 80) score--;

//         Signal signal;
//         if (score >= 3)
//             signal = Signal::BUY;
//         else if (score <= -3)
//             signal = Signal::SELL;
//         else
//             signal = Signal::HOLD;

//         decisions.push_back({dates[i], signal});

//         if (i == usable_start || i == usable_end - 1) {
//             std::cout << "Signal @ " << dates[i]
//                       << " — Close: " << c
//                       << ", EMA: " << e
//                       << ", RSI: " << r
//                       << ", MACD: " << m
//                       << ", Score: " << score
//                       << ", Decision: " << signalToString(signal)
//                       << std::endl;
//         }
//     }

//     std::cout << "Generated " << decisions.size() << " daily decisions" << std::endl;
//     return decisions;
// }

std::vector<Decision> generateDecisions(
    const std::vector<std::string>& dates,
    const std::vector<float>& close,
    const std::vector<float>& high,
    const std::vector<float>& low,
    const std::vector<float>& volume,
    bool use_gpu = false
) {
    std::vector<Decision> decisions;

    // Increase the lookback period to reduce noise
    const int period = 20; // Changed from 14
    // Add trend confirmation period
    const int trend_period = 50;

    if (dates.size() <= trend_period || close.size() <= trend_period) {
        std::cerr << "Not enough data for trend analysis." << std::endl;
        return decisions;
    }

    // Core indicators
    std::vector<float> sma = computeSMA(close, period);
    std::vector<float> ema = computeEMA(close, period);
    std::vector<float> rsi = computeRSI(close, period);
    MACDResult macd = computeMACD(close, 12, 26, 9); // Standard MACD parameters
    BollingerBands bb = computeBollingerBands(close, period, 2.0);
    StochasticResult stoch = computeStochasticOscillator(close, high, low, period, 3, 3); // Smoothed stochastic
    
    // Trend confirmation indicators
    std::vector<float> long_ema = computeEMA(close, trend_period);
    std::vector<float> atr = computeATR(high, low, close, period); // Assuming we have an ATR function

    // Find the start index that ensures all indicators have data available
    size_t sma_offset = close.size() - sma.size();
    size_t ema_offset = close.size() - ema.size();
    size_t long_ema_offset = close.size() - long_ema.size();
    size_t rsi_offset = close.size() - rsi.size();
    size_t macd_offset = close.size() - macd.macdLine.size();
    size_t macd_signal_offset = close.size() - macd.signalLine.size();
    size_t bb_offset = close.size() - bb.middle.size();
    size_t stoch_k_offset = close.size() - stoch.k.size();
    size_t stoch_d_offset = close.size() - stoch.d.size();
    size_t atr_offset = close.size() - atr.size();
    
    // Find the maximum offset to ensure all indicators are available
    size_t max_offset = std::max({sma_offset, ema_offset, long_ema_offset, rsi_offset, macd_offset, 
                                 macd_signal_offset, bb_offset, stoch_k_offset, stoch_d_offset, atr_offset});
    
    // Make sure we have enough data points
    if (max_offset >= close.size()) {
        std::cerr << "Not enough aligned data for decision generation." << std::endl;
        return decisions;
    }

    // Start after all indicators are available
    size_t usable_start = max_offset;
    size_t usable_end = close.size();

    std::cout << "Starting decision generation from index " << usable_start 
              << " (data point: " << dates[usable_start] << ")" << std::endl;
    
    // Track active positions for proper exit management
    bool in_position = false;
    float entry_price = 0.0f;
    size_t entry_idx = 0;
    float stop_loss = 0.0f;
    
    // Add volatility filter to avoid trading in highly volatile markets
    float avg_vol = 0.0f;
    for (size_t i = usable_start; i < usable_start + 10 && i < usable_end; ++i) {
        size_t atr_idx = i - atr_offset;
        if (atr_idx < atr.size()) {
            avg_vol += atr[atr_idx];
        }
    }
    avg_vol /= 10.0f;
    
    // Minimum score required for trade signals (higher = more conservative)
    const int buy_threshold = 4;   // Increased from 3
    const int sell_threshold = -4; // Decreased from -3

    for (size_t i = usable_start; i < usable_end; ++i) {
        float c = close[i];
        float prev_c = close[i-1];
        
        // Calculate relative indices for each indicator
        size_t ema_idx = i - ema_offset;
        size_t prev_ema_idx = (ema_idx > 0) ? (ema_idx - 1) : 0;
        size_t long_ema_idx = i - long_ema_offset;
        size_t rsi_idx = i - rsi_offset;
        size_t macd_idx = i - macd_offset;
        size_t prev_macd_idx = (macd_idx > 0) ? (macd_idx - 1) : 0;
        size_t signal_idx = i - macd_signal_offset;
        size_t prev_signal_idx = (signal_idx > 0) ? (signal_idx - 1) : 0;
        size_t bb_idx = i - bb_offset;
        size_t stoch_k_idx = i - stoch_k_offset;
        size_t stoch_d_idx = i - stoch_d_offset;
        size_t atr_idx = i - atr_offset;
        
        // Make sure we're not out of bounds
        if (ema_idx >= ema.size() || long_ema_idx >= long_ema.size() || 
            rsi_idx >= rsi.size() || macd_idx >= macd.macdLine.size() || 
            signal_idx >= macd.signalLine.size() || bb_idx >= bb.upper.size() || 
            stoch_k_idx >= stoch.k.size() || stoch_d_idx >= stoch.d.size() ||
            atr_idx >= atr.size()) {
            std::cerr << "Index out of bounds at date: " << dates[i] << std::endl;
            continue;
        }

        float e = ema[ema_idx];
        float prev_e = (prev_ema_idx < ema.size()) ? ema[prev_ema_idx] : e;
        float long_e = long_ema[long_ema_idx];
        float r = rsi[rsi_idx];
        float m = macd.macdLine[macd_idx];
        float prev_m = (prev_macd_idx < macd.macdLine.size()) ? macd.macdLine[prev_macd_idx] : m;
        float signal = macd.signalLine[signal_idx];
        float prev_signal = (prev_signal_idx < macd.signalLine.size()) ? macd.signalLine[prev_signal_idx] : signal;
        float b_upper = bb.upper[bb_idx];
        float b_middle = bb.middle[bb_idx];
        float b_lower = bb.lower[bb_idx];
        float k = stoch.k[stoch_k_idx];
        float d = stoch.d[stoch_d_idx];
        float current_atr = atr[atr_idx];
        
        // Volatility filter - don't trade if volatility is too high
        bool high_volatility = (current_atr > avg_vol * 1.5);
        
        // Calculate market trend strength
        bool strong_uptrend = (c > long_e) && (e > long_e);
        bool strong_downtrend = (c < long_e) && (e < long_e);
        
        // Check for stop loss if in position
        Signal signal_decision = Signal::HOLD;
        
        if (in_position && c <= stop_loss) {
            // Stop loss triggered
            signal_decision = Signal::SELL;
            in_position = false;
            
            std::cout << "STOP LOSS @ " << dates[i] 
                      << " — Entry: " << entry_price 
                      << ", Exit: " << c
                      << ", Loss: " << ((c / entry_price - 1) * 100) << "%"
                      << std::endl;
        }
        else {
            int score = 0;

            // Only evaluate buy signals if not in position and not in high volatility
            if (!in_position && !high_volatility) {
                // BUY signals - only consider in up trends or potential reversals
                if (strong_uptrend) score += 2; // Strong weight on trend
                
                // EMA crossover (more weight)
                if (c > e && prev_c <= prev_e) score += 2;
                
                // RSI oversold with stricter conditions
                if (r < 30 && r > r - 5) score++; // RSI reversing up from oversold
                
                // MACD signals (with more weight on crossovers)
                if (m > signal && prev_m <= prev_signal) score += 2; // MACD cross above signal
                if (m < 0 && m > prev_m && m - prev_m > 0.1 * current_atr) score++; // Strong MACD reversal while negative
                
                // Bollinger Band signals
                if (c < b_lower) score++; // Potential oversold
                if (c > b_lower && prev_c < b_lower) score++; // Reversal from oversold
                
                // Stochastic signals (with confirmation)
                if (k < 20 && k > d && k > stoch.k[stoch_k_idx-1]) score += 2; // Stochastic bullish crossover in oversold
                
                // Volume confirmation if available
                if (i > 0 && volume[i] > volume[i-1] * 1.2 && c > prev_c) score++;
                
                // Check for potential buy signal with higher threshold
                if (score >= buy_threshold) {
                    signal_decision = Signal::BUY;
                    in_position = true;
                    entry_price = c;
                    entry_idx = i;
                    
                    // Set stop loss at 2 ATR below entry price
                    stop_loss = entry_price * (1.0 - (2.0 * current_atr / entry_price));
                }
            }
            // Only evaluate sell signals if in position or for potential shorts
            else if (in_position || !high_volatility) {
                // SELL signals - for exiting positions or potential shorting opportunities
                if (strong_downtrend) score -= 2; // Strong weight on trend
                
                // EMA crossover (more weight)
                if (c < e && prev_c >= prev_e) score -= 2;
                
                // RSI overbought with confirmation
                if (r > 70 && r < r + 5) score--; // RSI reversing down from overbought
                
                // MACD signals
                if (m < signal && prev_m >= prev_signal) score -= 2; // MACD cross below signal
                if (m > 0 && m < prev_m && prev_m - m > 0.1 * current_atr) score--; // Strong MACD reversal while positive
                
                // Bollinger Band signals
                if (c > b_upper) score--; // Potential overbought
                if (c < b_upper && prev_c > b_upper) score--; // Reversal from overbought
                
                // Stochastic signals (with confirmation)
                if (k > 80 && k < d && k < stoch.k[stoch_k_idx-1]) score -= 2; // Stochastic bearish crossover in overbought
                
                // Profit taking rules (if in position)
                if (in_position) {
                    // Take profit if we have significant gain (more than 3 ATR)
                    if (c > entry_price * (1.0 + (3.0 * current_atr / entry_price))) {
                        score -= 4; // Strong signal to sell for profit taking
                    }
                    
                    // Exit if position has been held too long (more than 5 bars)
                    if (i - entry_idx > 5) {
                        score--;
                    }
                }
                
                // Check for potential sell signal
                if (score <= sell_threshold) {
                    signal_decision = Signal::SELL;
                    in_position = false;
                }
            }
        }

        decisions.push_back({dates[i], signal_decision});

        // Log important signals or first/last entries
        if (i == usable_start || i == usable_end - 1 || signal_decision != Signal::HOLD) {
            std::cout << "Signal @ " << dates[i]
                      << " — Close: " << c
                      << ", EMA: " << e
                      << ", Long EMA: " << long_e
                      << ", RSI: " << r
                      << ", MACD: " << m
                      << ", Signal: " << signal
                      << ", Stoch %K/%D: " << k << "/" << d
                      << ", ATR: " << current_atr
                      << ", Decision: " << signalToString(signal_decision);
            
            if (signal_decision == Signal::BUY) {
                std::cout << ", Stop Loss: " << stop_loss;
            }
            
            std::cout << std::endl;
        }
    }

    std::cout << "Generated " << decisions.size() << " decisions" << std::endl;
    return decisions;
}



#endif // DECISION_LOGIC_HPP
