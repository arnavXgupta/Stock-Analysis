#include <iostream>
#include <vector>
#include <string>
#include "fetcher.hpp"
#include "indicators_rectified.hpp"
#include "decision_logic_rectified.hpp"
#include "backtester.hpp"
#include "export_decisions.hpp"

int main()
{
    // Configuration
    std::string api_key = "JJBVkPtrkQflCzlf";
    std::string symbol = "TSLA";
    std::string interval = "5min"; // daily interval
    bool use_gpu = false;          // Set to false since GPU functions are commented out

    std::cout << "Starting stock analysis program for " << symbol << std::endl;
    std::cout << "GPU acceleration: " << (use_gpu ? "enabled" : "disabled") << std::endl;

    // Fetch data
    StockData data;
    bool success = fetchStockData(symbol, interval, api_key, data, use_gpu);
    if (!success)
    {
        std::cerr << "Error: Failed to fetch data." << std::endl;
        return 1;
    }

    std::cout << "Successfully fetched " << data.dates.size() << " data points." << std::endl;

    if (data.dates.size() > 0)
    {
        std::cout << "Date range: " << data.dates.back() << " to " << data.dates.front() << std::endl;
    }

    // Generate decisions
    std::vector<Decision> decisions = generateDecisions(
        data.dates,
        data.close,
        data.high,
        data.low,
        data.volume,
        use_gpu
    );

    //save decision to csv
    exportDecisionCSV("signals.csv", decisions, data.close);

    Backtester bt(decisions, data.close, data.dates);
    bt.run();    // Simulate trading
    bt.exportBacktestCSV("backtest.csv");
    bt.report(); // Print summary

    // Output decisions
    std::cout << "Date       | Signal\n";
    std::cout << "-----------|--------\n";

    if (decisions.empty())
    {
        std::cout << "No trading signals generated." << std::endl;
    }
    else
    {
        // Print the most recent 10 decisions (or all if less than 10)
        size_t start_idx = (decisions.size() > 10) ? decisions.size() - 10 : 0;
        for (size_t i = start_idx; i < decisions.size(); ++i)
        {
            std::cout << decisions[i].date << " | " << signalToString(decisions[i].signal) << "\n";
        }

        // Count signals by type
        int buy_count = 0, sell_count = 0, hold_count = 0;
        for (const auto &d : decisions)
        {
            switch (d.signal)
            {
            case Signal::BUY:
                buy_count++;
                break;
            case Signal::SELL:
                sell_count++;
                break;
            case Signal::HOLD:
                hold_count++;
                break;
            }
        }

        std::cout << "\nSummary: " << decisions.size() << " total signals\n";
        std::cout << "BUY: " << buy_count << ", SELL: " << sell_count << ", HOLD: " << hold_count << std::endl;
    }

    return 0;
}