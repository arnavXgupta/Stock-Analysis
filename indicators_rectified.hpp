#ifndef INDICATORS_HPP
#define INDICATORS_HPP

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
// #include <cuda_runtime.h>

// -------------------- CPU VERSIONS -------------------- //

std::vector<float> computeSMA(const std::vector<float>& prices, int period) {
    std::vector<float> sma;
    
    // Input validation
    if (prices.size() < period) {
        std::cerr << "Warning: Not enough data points for SMA calculation" << std::endl;
        return sma;  // Return empty vector
    }
    
    std::cout << "Computing SMA with period " << period << " on " << prices.size() << " prices" << std::endl;
    
    for (size_t i = period - 1; i < prices.size(); ++i) {
        float sum = 0;
        for (int j = 0; j < period; ++j)
            sum += prices[i - j];
        sma.push_back(sum / period);
    }
    
    std::cout << "SMA calculation complete. Output size: " << sma.size() << std::endl;
    return sma;
}

std::vector<float> computeEMA(const std::vector<float>& prices, int period) {
    std::vector<float> ema;
    
    // Input validation
    if (prices.size() < period) {
        std::cerr << "Warning: Not enough data points for EMA calculation" << std::endl;
        return ema;  // Return empty vector
    }
    
    std::cout << "Computing EMA with period " << period << " on " << prices.size() << " prices" << std::endl;
    
    // Calculate initial SMA as the first EMA value
    float firstEma = 0;
    for (int i = 0; i < period; ++i) {
        firstEma += prices[i];
    }
    firstEma /= period;
    
    ema.push_back(firstEma);
    
    // Calculate subsequent EMA values
    float multiplier = 2.0f / (period + 1);
    
    for (size_t i = period; i < prices.size(); ++i) {
        float value = (prices[i] - ema.back()) * multiplier + ema.back();
        ema.push_back(value);
    }
    
    std::cout << "EMA calculation complete. Output size: " << ema.size() << std::endl;
    return ema;
}

std::vector<float> computeRSI(const std::vector<float>& prices, int period = 14) {
    std::vector<float> rsi;
    
    // Input validation
    if (prices.size() <= period) {
        std::cerr << "Warning: Not enough data points for RSI calculation" << std::endl;
        return rsi;
    }
    
    std::cout << "Computing RSI with period " << period << " on " << prices.size() << " prices" << std::endl;
    
    // Calculate the initial average gain and loss
    float avgGain = 0, avgLoss = 0;
    
    for (int i = 1; i <= period; ++i) {
        float change = prices[i] - prices[i-1];
        if (change >= 0) {
            avgGain += change;
        } else {
            avgLoss -= change;  // Make positive
        }
    }
    
    avgGain /= period;
    avgLoss /= period;
    
    // Calculate the first RSI
    float rs = (avgLoss == 0) ? 100 : avgGain / avgLoss;
    rsi.push_back(100 - (100 / (1 + rs)));
    
    // Calculate subsequent RSI values
    for (size_t i = period + 1; i < prices.size(); ++i) {
        float change = prices[i] - prices[i-1];
        float gain = (change >= 0) ? change : 0;
        float loss = (change < 0) ? -change : 0;
        
        avgGain = ((period - 1) * avgGain + gain) / period;
        avgLoss = ((period - 1) * avgLoss + loss) / period;
        
        rs = (avgLoss == 0) ? 100 : avgGain / avgLoss;
        rsi.push_back(100 - (100 / (1 + rs)));
    }
    
    std::cout << "RSI calculation complete. Output size: " << rsi.size() << std::endl;
    return rsi;
}

// std::vector<float> computeMACD(const std::vector<float>& prices, int fast = 12, int slow = 26) {
//     // Input validation
//     if (prices.size() <= slow) {
//         std::cerr << "Warning: Not enough data points for MACD calculation" << std::endl;
//         return std::vector<float>();
//     }
    
//     std::cout << "Computing MACD with fast=" << fast << ", slow=" << slow << " on " << prices.size() << " prices" << std::endl;
    
//     std::vector<float> fastEMA = computeEMA(prices, fast);
//     std::vector<float> slowEMA = computeEMA(prices, slow);
    
//     std::vector<float> macd;
    
//     // Align the EMAs properly
//     size_t diff = fastEMA.size() - slowEMA.size();
    
//     for (size_t i = 0; i < slowEMA.size(); ++i) {
//         macd.push_back(fastEMA[i + diff] - slowEMA[i]);
//     }
    
//     std::cout << "MACD calculation complete. Output size: " << macd.size() << std::endl;
//     return macd;
// }

struct BollingerBands {
    std::vector<float> upper;
    std::vector<float> middle;
    std::vector<float> lower;
};

BollingerBands computeBollingerBands(const std::vector<float>& prices, int period = 20, float numStdDev = 2.0f) {
    BollingerBands bb;
    
    // Input validation
    if (prices.size() < period) {
        std::cerr << "Warning: Not enough data points for Bollinger Bands calculation" << std::endl;
        return bb;
    }
    
    std::cout << "Computing Bollinger Bands with period=" << period << ", stdDev=" << numStdDev 
              << " on " << prices.size() << " prices" << std::endl;
    
    for (size_t i = period - 1; i < prices.size(); ++i) {
        float sum = 0;
        for (int j = 0; j < period; ++j)
            sum += prices[i - j];
        float mean = sum / period;

        float variance = 0;
        for (int j = 0; j < period; ++j)
            variance += pow(prices[i - j] - mean, 2);
        float stddev = sqrt(variance / period);

        bb.middle.push_back(mean);
        bb.upper.push_back(mean + numStdDev * stddev);
        bb.lower.push_back(mean - numStdDev * stddev);
    }
    
    std::cout << "Bollinger Bands calculation complete. Size: " << bb.middle.size() << std::endl;
    return bb;
}

// std::vector<float> computeStochasticOscillator(const std::vector<float>& close, const std::vector<float>& high, const std::vector<float>& low, int period = 14) {
//     std::vector<float> stochastic;
    
//     // Input validation
//     size_t minSize = std::min({close.size(), high.size(), low.size()});
//     if (minSize < period) {
//         std::cerr << "Warning: Not enough data points for Stochastic Oscillator calculation" << std::endl;
//         return stochastic;
//     }
    
//     std::cout << "Computing Stochastic Oscillator with period=" << period << " on " << minSize << " data points" << std::endl;
    
//     for (size_t i = period - 1; i < minSize; ++i) {
//         float highest = *std::max_element(high.begin() + i - period + 1, high.begin() + i + 1);
//         float lowest = *std::min_element(low.begin() + i - period + 1, low.begin() + i + 1);
        
//         // Prevent division by zero
//         if (highest == lowest) {
//             stochastic.push_back(50.0f);  // Middle value if no range
//         } else {
//             float value = 100 * ((close[i] - lowest) / (highest - lowest));
//             stochastic.push_back(value);
//         }
//     }
    
//     std::cout << "Stochastic Oscillator calculation complete. Size: " << stochastic.size() << std::endl;
//     return stochastic;
// }


//MACD and Stochastic Oscillator claude version
// Corrected MACD implementation with signal line
struct MACDResult {
    std::vector<float> macdLine;     // MACD line (difference between fast and slow EMAs)
    std::vector<float> signalLine;   // Signal line (EMA of MACD line)
    std::vector<float> histogram;    // Histogram (MACD line - signal line)
};

MACDResult computeMACD(const std::vector<float>& prices, int fast = 12, int slow = 26, int signal = 9) {
    MACDResult result;
    
    // Input validation
    if (prices.size() <= slow) {
        std::cerr << "Warning: Not enough data points for MACD calculation" << std::endl;
        return result;
    }
    
    std::cout << "Computing MACD with fast=" << fast << ", slow=" << slow 
              << ", signal=" << signal << " on " << prices.size() << " prices" << std::endl;
    
    std::vector<float> fastEMA = computeEMA(prices, fast);
    std::vector<float> slowEMA = computeEMA(prices, slow);
    
    // Calculate MACD line (fast EMA - slow EMA)
    // We need to align the EMAs since they have different starting points
    size_t fastOffset = prices.size() - fastEMA.size();
    size_t slowOffset = prices.size() - slowEMA.size();
    size_t macdSize = prices.size() - std::max(fastOffset, slowOffset);
    
    for (size_t i = 0; i < macdSize; ++i) {
        size_t fastIndex = i + fastEMA.size() - macdSize;
        size_t slowIndex = i + slowEMA.size() - macdSize;
        result.macdLine.push_back(fastEMA[fastIndex] - slowEMA[slowIndex]);
    }
    
    // Calculate signal line (EMA of MACD line)
    if (result.macdLine.size() >= signal) {
        // Calculate initial SMA for signal line
        float signalSMA = 0;
        for (int i = 0; i < signal; ++i) {
            signalSMA += result.macdLine[i];
        }
        signalSMA /= signal;
        result.signalLine.push_back(signalSMA);
        
        // Calculate EMA of MACD line
        float multiplier = 2.0f / (signal + 1);
        for (size_t i = signal; i < result.macdLine.size(); ++i) {
            float value = (result.macdLine[i] - result.signalLine.back()) * multiplier + result.signalLine.back();
            result.signalLine.push_back(value);
        }
        
        // Calculate histogram (MACD line - signal line)
        size_t histStart = result.macdLine.size() - result.signalLine.size();
        for (size_t i = 0; i < result.signalLine.size(); ++i) {
            result.histogram.push_back(result.macdLine[i + histStart] - result.signalLine[i]);
        }
    }
    
    std::cout << "MACD calculation complete. MACD size: " << result.macdLine.size() 
              << ", Signal size: " << result.signalLine.size() 
              << ", Histogram size: " << result.histogram.size() << std::endl;
    
    return result;
}

// Corrected Stochastic Oscillator implementation with %K and %D
struct StochasticResult {
    std::vector<float> k;  // %K line (raw stochastic)
    std::vector<float> d;  // %D line (moving average of %K)
};

StochasticResult computeStochasticOscillator(
    const std::vector<float>& close, 
    const std::vector<float>& high, 
    const std::vector<float>& low, 
    int periodK = 14,     // Period for %K calculation
    int periodD = 3,      // Period for %D calculation (SMA of %K)
    int smoothK = 1       // Smoothing for %K (1 = no smoothing)
) {
    StochasticResult result;
    
    // Input validation
    size_t minSize = std::min({close.size(), high.size(), low.size()});
    if (minSize < periodK) {
        std::cerr << "Warning: Not enough data points for Stochastic Oscillator calculation" << std::endl;
        return result;
    }
    
    std::cout << "Computing Stochastic Oscillator with periodK=" << periodK 
              << ", smoothK=" << smoothK << ", periodD=" << periodD
              << " on " << minSize << " data points" << std::endl;
    
    // Calculate raw %K values
    std::vector<float> rawK;
    for (size_t i = periodK - 1; i < minSize; ++i) {
        float highest = *std::max_element(high.begin() + i - periodK + 1, high.begin() + i + 1);
        float lowest = *std::min_element(low.begin() + i - periodK + 1, low.begin() + i + 1);
        
        // Prevent division by zero
        if (highest == lowest) {
            rawK.push_back(50.0f);  // Middle value if no range
        } else {
            float value = 100 * ((close[i] - lowest) / (highest - lowest));
            rawK.push_back(value);
        }
    }
    
    // Apply smoothing to %K if requested (smoothK > 1)
    if (smoothK > 1 && rawK.size() >= smoothK) {
        for (size_t i = smoothK - 1; i < rawK.size(); ++i) {
            float sum = 0;
            for (int j = 0; j < smoothK; ++j) {
                sum += rawK[i - j];
            }
            result.k.push_back(sum / smoothK);
        }
    } else {
        // No smoothing required
        result.k = rawK;
    }
    
    // Calculate %D (SMA of %K)
    if (result.k.size() >= periodD) {
        for (size_t i = periodD - 1; i < result.k.size(); ++i) {
            float sum = 0;
            for (int j = 0; j < periodD; ++j) {
                sum += result.k[i - j];
            }
            result.d.push_back(sum / periodD);
        }
    }
    
    std::cout << "Stochastic Oscillator calculation complete. %K size: " << result.k.size() 
              << ", %D size: " << result.d.size() << std::endl;
    
    return result;
}


// ATR (Average True Range) calculation - add this function if not already present
std::vector<float> computeATR(const std::vector<float>& high, const std::vector<float>& low, 
    const std::vector<float>& close, int period = 14) {
std::vector<float> atr;

// Input validation
if (high.size() < period + 1 || low.size() < period + 1 || close.size() < period + 1) {
std::cerr << "Warning: Not enough data points for ATR calculation" << std::endl;
return atr;
}

std::cout << "Computing ATR with period " << period << " on " << high.size() << " data points" << std::endl;

std::vector<float> trueRange;

// Calculate True Range for each bar
for (size_t i = 1; i < high.size(); ++i) {
float highLowRange = high[i] - low[i];
float highClosePrevRange = std::abs(high[i] - close[i-1]);
float lowClosePrevRange = std::abs(low[i] - close[i-1]);

float tr = std::max({highLowRange, highClosePrevRange, lowClosePrevRange});
trueRange.push_back(tr);
}

// Calculate initial ATR (simple average of TR for first 'period' values)
float sum = 0;
for (int i = 0; i < period; ++i) {
sum += trueRange[i];
}
float firstATR = sum / period;
atr.push_back(firstATR);

// Calculate subsequent ATR values using EMA formula
for (size_t i = period; i < trueRange.size(); ++i) {
float currentATR = (atr.back() * (period - 1) + trueRange[i]) / period;
atr.push_back(currentATR);
}

std::cout << "ATR calculation complete. Output size: " << atr.size() << std::endl;
return atr;
}


// -------------------- GPU VERSION -------------------- //

// __global__ void smaKernel(float* prices, float* output, int n, int period) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= period - 1 && i < n) {
//         float sum = 0;
//         for (int j = 0; j < period; ++j)
//             sum += prices[i - j];
//         output[i - (period - 1)] = sum / period;
//     }
// }

// void computeSMA_GPU(const std::vector<float>& prices, std::vector<float>& result, int period) {
//     int n = prices.size();
//     int sizeOut = n - period + 1;

//     float *d_prices, *d_output;
//     cudaMalloc(&d_prices, n * sizeof(float));
//     cudaMalloc(&d_output, sizeOut * sizeof(float));

//     cudaMemcpy(d_prices, prices.data(), n * sizeof(float), cudaMemcpyHostToDevice);

//     int blockSize = 256;
//     int numBlocks = (n + blockSize - 1) / blockSize;
//     smaKernel<<<numBlocks, blockSize>>>(d_prices, d_output, n, period);
//     cudaDeviceSynchronize();

//     result.resize(sizeOut);
//     cudaMemcpy(result.data(), d_output, sizeOut * sizeof(float), cudaMemcpyDeviceToHost);

//     cudaFree(d_prices);
//     cudaFree(d_output);
// }

// __global__ void emaKernel(float* prices, float* output, int n, float multiplier) {
//     int i = threadIdx.x;
//     if (i == 0) output[0] = prices[0];
//     __syncthreads();
//     for (int i = 1; i < n; ++i)
//         output[i] = (prices[i] - output[i - 1]) * multiplier + output[i - 1];
// }

// void computeEMA_GPU(const std::vector<float>& prices, std::vector<float>& result, int period) {
//     int n = prices.size();
//     float multiplier = 2.0f / (period + 1);

//     float *d_prices, *d_output;
//     cudaMalloc(&d_prices, n * sizeof(float));
//     cudaMalloc(&d_output, n * sizeof(float));

//     cudaMemcpy(d_prices, prices.data(), n * sizeof(float), cudaMemcpyHostToDevice);

//     emaKernel<<<1, n>>>(d_prices, d_output, n, multiplier);
//     cudaDeviceSynchronize();

//     result.resize(n);
//     cudaMemcpy(result.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

//     cudaFree(d_prices);
//     cudaFree(d_output);
// }

#endif // INDICATORS_HPP