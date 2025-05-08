#ifndef FETCHER_HPP
#define FETCHER_HPP

#include <iostream>
#include <string>
#include <curl/curl.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <nlohmann/json.hpp>
#include <set>

using json = nlohmann::json;

// Global toggle for CPU/GPU mode (externally overridable)
inline bool use_gpu = false;

// --- Struct to hold parsed stock data ---
struct StockData
{
    std::vector<std::string> dates;
    std::vector<float> open;
    std::vector<float> high;
    std::vector<float> low;
    std::vector<float> close;
    std::vector<float> volume;
};

// --- Callback for libcurl ---
inline size_t WriteCallback(void *contents, size_t size, size_t nmemb, std::string *output)
{
    size_t totalSize = size * nmemb;
    output->append((char *)contents, totalSize);
    return totalSize;
}

// --- Fetch stock data from Alpha Vantage ---
// inline bool fetch_stock_data(const std::string &symbol, const std::string &interval, const std::string &apiKey, const std::string &filename)
// {
//     std::string url = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=" + symbol + "&interval=" + interval + "&apikey=" + apiKey + "&outputsize=full";

//     CURL *curl = curl_easy_init();
//     std::string response;

//     if (curl)
//     {
//         curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
//         curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
//         curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

//         CURLcode res = curl_easy_perform(curl);
//         if (res != CURLE_OK)
//         {
//             std::cerr << "CURL error: " << curl_easy_strerror(res) << "\n";
//             curl_easy_cleanup(curl);
//             return false;
//         }
//         curl_easy_cleanup(curl);
//     }

//     // Parse JSON response and write to CSV
//     try
//     {
//         auto jsonData = json::parse(response);
//         auto timeSeries = jsonData["Time Series Intraday"];

//         std::ofstream file(filename);
//         file << "Date,Open,High,Low,Close,Volume\n";
//         for (auto it = timeSeries.begin(); it != timeSeries.end(); ++it)
//         {
//             std::string date = it.key();
//             auto dayData = it.value();
//             // In fetch_stock_data function, modify the CSV writing part:
//             file << date << ","
//                  << (dayData["1. open"].is_null() ? "0.0" : dayData["1. open"].get<std::string>()) << ","
//                  << (dayData["2. high"].is_null() ? "0.0" : dayData["2. high"].get<std::string>()) << ","
//                  << (dayData["3. low"].is_null() ? "0.0" : dayData["3. low"].get<std::string>()) << ","
//                  << (dayData["4. close"].is_null() ? "0.0" : dayData["4. close"].get<std::string>()) << ","
//                  << (dayData["5. volume"].is_null() ? "0.0" : dayData["5. volume"].get<std::string>()) << "\n";
//         }

//         file.close();
//         std::cout << "Data saved to " << filename << "\n";
//         return true;
//     }
//     catch (std::exception &e)
//     {
//         std::cerr << "JSON parsing error: " << e.what() << "\n";
//         return false;
//     }
// }

inline bool fetch_stock_data(const std::string &symbol, const std::string &interval,
                             const std::string &apiKey, const std::string &filename)
{
    // Validate interval parameter
    const std::set<std::string> valid_intervals = {"1min", "5min", "15min", "30min", "60min"};
    if (valid_intervals.find(interval) == valid_intervals.end())
    {
        std::cerr << "Invalid interval parameter. Use: 1min, 5min, 15min, 30min, 60min\n";
        return false;
    }

    std::string url = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=" +
                      symbol + "&interval=" + interval + "&apikey=" + apiKey + "&outputsize=full";

    CURL *curl = curl_easy_init();
    std::string response;

    if (curl)
    {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0"); // Required by Alpha Vantage [2]

        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK)
        {
            std::cerr << "CURL error: " << curl_easy_strerror(res) << "\n";
            curl_easy_cleanup(curl);
            return false;
        }
        curl_easy_cleanup(curl);
    }

    try
    {
        auto jsonData = json::parse(response);

        // Check for API errors first [3][5]
        if (jsonData.contains("Error Message"))
        {
            std::cerr << "API Error: " << jsonData["Error Message"].get<std::string>() << "\n";
            return false;
        }

        // Dynamic time series key based on interval [2][3]
        std::string time_series_key = "Time Series (" + interval + ")";
        if (!jsonData.contains(time_series_key))
        {
            std::cerr << "Missing expected time series data in response\n";
            return false;
        }
        auto timeSeries = jsonData[time_series_key];

        std::ofstream file(filename);
        file << "Date,Open,High,Low,Close,Volume\n";
        for (auto it = timeSeries.begin(); it != timeSeries.end(); ++it)
        {
            std::string date = it.key();
            auto dayData = it.value();
            // In fetch_stock_data function, modify the CSV writing part:
            file << date << ","
                 << (dayData["1. open"].is_null() ? "0.0" : dayData["1. open"].get<std::string>()) << ","
                 << (dayData["2. high"].is_null() ? "0.0" : dayData["2. high"].get<std::string>()) << ","
                 << (dayData["3. low"].is_null() ? "0.0" : dayData["3. low"].get<std::string>()) << ","
                 << (dayData["4. close"].is_null() ? "0.0" : dayData["4. close"].get<std::string>()) << ","
                 << (dayData["5. volume"].is_null() ? "0.0" : dayData["5. volume"].get<std::string>()) << "\n";
        }

        file.close();
        std::cout << "Data saved to " << filename << "\n";

        return true;
    }
    catch (std::exception &e)
    {
        std::cerr << "JSON parsing error: " << e.what() << "\n";
        return false;
    }
}

// --- Parse close prices only ---
inline bool parse_csv_close_prices(const std::string &filename, std::vector<float> &closePrices)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filename << "\n";
        return false;
    }

    std::string line;
    getline(file, line); // skip header

    while (getline(file, line))
    {
        std::stringstream ss(line);
        std::string token;
        int col = 0;
        float close = 0.0f;

        while (getline(ss, token, ','))
        {
            if (col == 4)
            {
                try
                {
                    close = std::stof(token);
                }
                catch (...)
                {
                    close = 0.0f;
                }
                closePrices.push_back(close);
                break;
            }
            col++;
        }
    }

    file.close();
    return true;
}

// // --- Full CSV parser to StockData struct ---
// inline bool parse_csv_full(const std::string& filename, StockData& stockData) {
//     std::ifstream file(filename);
//     if (!file.is_open()) return false;

//     std::string line;
//     getline(file, line); // Skip header

//     while (getline(file, line)) {
//         std::stringstream ss(line);
//         std::string token;

//         std::string date;
//         float open, high, low, close, volume;

//         getline(ss, date, ',');
//         getline(ss, token, ','); open = std::stof(token);
//         getline(ss, token, ','); high = std::stof(token);
//         getline(ss, token, ','); low = std::stof(token);
//         getline(ss, token, ','); close = std::stof(token);
//         getline(ss, token, ','); volume = std::stof(token);

//         stockData.dates.push_back(date);
//         stockData.open.push_back(open);
//         stockData.high.push_back(high);
//         stockData.low.push_back(low);
//         stockData.close.push_back(close);
//         stockData.volume.push_back(volume);
//     }

//     file.close();
//     return true;
// }

// --- Full CSV parser to StockData struct ---
inline bool parse_csv_full(const std::string &filename, StockData &stockData)
{
    std::ifstream file(filename);
    if (!file.is_open())
        return false;

    std::string line;
    getline(file, line); // Skip header

    while (getline(file, line))
    {
        std::stringstream ss(line);
        std::string token;

        std::string date;
        float open, high, low, close, volume;

        getline(ss, date, ',');

        // Parse open value
        getline(ss, token, ',');
        open = (token == "NULL" || token.empty()) ? 0.0f : std::stof(token);

        // Parse high value
        getline(ss, token, ',');
        high = (token == "NULL" || token.empty()) ? 0.0f : std::stof(token);

        // Parse low value
        getline(ss, token, ',');
        low = (token == "NULL" || token.empty()) ? 0.0f : std::stof(token);

        // Parse close value
        getline(ss, token, ',');
        close = (token == "NULL" || token.empty()) ? 0.0f : std::stof(token);

        // Parse volume value
        getline(ss, token, ',');
        volume = (token == "NULL" || token.empty()) ? 0.0f : std::stof(token);

        stockData.dates.push_back(date);
        stockData.open.push_back(open);
        stockData.high.push_back(high);
        stockData.low.push_back(low);
        stockData.close.push_back(close);
        stockData.volume.push_back(volume);
    }

    file.close();
    return true;
}

// --- CPU-based processing (SMA) ---
inline void cpu_processing(const std::vector<float> &closePrices)
{
    std::cout << "Processing data on CPU...\n";
    for (size_t i = 0; i < closePrices.size() - 5; ++i)
    {
        float sum = 0;
        for (size_t j = 0; j < 5; ++j)
        {
            sum += closePrices[i + j];
        }
        float sma = sum / 5;
        std::cout << "SMA at index " << i << ": " << sma << "\n";
    }
}

inline bool fetchStockData(const std::string &symbol,
                           const std::string &interval,
                           const std::string &api_key,
                           StockData &outData,
                           bool use_gpu_flag)
{
    // Build filename
    std::string filename = symbol + "_data.csv";

    // Update global toggle
    use_gpu = use_gpu_flag;

    // Fetch and save data
    if (!fetch_stock_data(symbol, interval, api_key, filename))
    {
        return false;
    }

    // Parse full CSV
    return parse_csv_full(filename, outData);
}

// --- GPU processing (optional stubs) ---
// #include <cuda_runtime.h>
// inline float* prepare_for_gpu(const std::vector<float>& data) {
//     float* deviceData;
//     size_t size = data.size() * sizeof(float);
//     cudaMalloc((void**)&deviceData, size);
//     cudaMemcpy(deviceData, data.data(), size, cudaMemcpyHostToDevice);
//     return deviceData;
// }

// __global__ void gpu_processing(float* closePrices, size_t dataSize) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < dataSize - 5) {
//         float sum = 0;
//         for (int i = 0; i < 5; ++i) {
//             sum += closePrices[idx + i];
//         }
//         float sma = sum / 5;
//         if (idx == 0) {
//             printf("SMA at index %d: %f\n", idx, sma);
//         }
//     }
// }

#endif // FETCHER_HPP
