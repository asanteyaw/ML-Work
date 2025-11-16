/**
 * @file CSVParserTest.cpp
 * @brief Test and demonstration of the high-performance CSV parser with US Accidents dataset
 */

#include "CSVParser.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <map>
#include <algorithm>
#include <numeric>

// Helper function to print a row of CSV data
void printRow(const std::vector<csv::FieldType>& row, const std::vector<std::string>& headers = {}) {
    for (size_t i = 0; i < row.size(); ++i) {
        // Only print header if it exists for this column
        if (!headers.empty() && i < headers.size()) {
            std::cout << headers[i] << ": ";
        } else {
            std::cout << "Column " << i << ": ";
        }
        
        if (std::holds_alternative<std::nullptr_t>(row[i])) {
            std::cout << "NULL";
        } else if (std::holds_alternative<std::string_view>(row[i])) {
            std::cout << std::get<std::string_view>(row[i]);
        } else if (std::holds_alternative<std::string>(row[i])) {
            std::cout << std::get<std::string>(row[i]);
        } else if (std::holds_alternative<int64_t>(row[i])) {
            std::cout << std::get<int64_t>(row[i]);
        } else if (std::holds_alternative<double>(row[i])) {
            std::cout << std::fixed << std::setprecision(2) << std::get<double>(row[i]);
        } else if (std::holds_alternative<bool>(row[i])) {
            std::cout << (std::get<bool>(row[i]) ? "true" : "false");
        }
        
        std::cout << " | ";
        
        // Add a newline every 5 columns for better readability
        if ((i + 1) % 5 == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

// Function to get column index by name
size_t getColumnIndex(const std::vector<std::string>& headers, const std::string& columnName) {
    auto it = std::find(headers.begin(), headers.end(), columnName);
    if (it != headers.end()) {
        return std::distance(headers.begin(), it);
    }
    return std::numeric_limits<size_t>::max(); // Not found
}

// Function to analyze accident severity distribution
void analyzeAccidentSeverity(const std::vector<std::vector<csv::FieldType>>& data, 
                            const std::vector<std::string>& headers) {
    std::cout << "\nAnalyzing accident severity distribution..." << std::endl;
    
    size_t severityCol = getColumnIndex(headers, "Severity");
    if (severityCol == std::numeric_limits<size_t>::max()) {
        std::cerr << "Severity column not found!" << std::endl;
        return;
    }
    
    std::map<int, int> severityCounts;
    
    for (const auto& row : data) {
        if (row.size() > severityCol) {
            auto severityOpt = csv::TypeConverter::convert<int>(row[severityCol]);
            if (severityOpt) {
                severityCounts[*severityOpt]++;
            }
        }
    }
    
    std::cout << "Severity distribution:" << std::endl;
    for (const auto& [severity, count] : severityCounts) {
        std::cout << "Severity " << severity << ": " << count << " accidents (" 
                  << std::fixed << std::setprecision(2) 
                  << (100.0 * count / data.size()) << "%)" << std::endl;
    }
}

// Function to analyze accidents by state
void analyzeAccidentsByState(const std::vector<std::vector<csv::FieldType>>& data, 
                            const std::vector<std::string>& headers) {
    std::cout << "\nAnalyzing accidents by state..." << std::endl;
    
    size_t stateCol = getColumnIndex(headers, "State");
    if (stateCol == std::numeric_limits<size_t>::max()) {
        std::cerr << "State column not found!" << std::endl;
        return;
    }
    
    std::map<std::string, int> stateCounts;
    
    for (const auto& row : data) {
        if (row.size() > stateCol) {
            auto stateOpt = csv::TypeConverter::convert<std::string>(row[stateCol]);
            if (stateOpt) {
                stateCounts[*stateOpt]++;
            }
        }
    }
    
    // Find top 10 states by accident count
    std::vector<std::pair<std::string, int>> stateCountVec(stateCounts.begin(), stateCounts.end());
    std::sort(stateCountVec.begin(), stateCountVec.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    std::cout << "Top 10 states by accident count:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(10), stateCountVec.size()); ++i) {
        std::cout << i + 1 << ". " << stateCountVec[i].first << ": " 
                  << stateCountVec[i].second << " accidents (" 
                  << std::fixed << std::setprecision(2) 
                  << (100.0 * stateCountVec[i].second / data.size()) << "%)" << std::endl;
    }
}

// Function to analyze accidents by weather condition
void analyzeAccidentsByWeather(const std::vector<std::vector<csv::FieldType>>& data, 
                              const std::vector<std::string>& headers) {
    std::cout << "\nAnalyzing accidents by weather condition..." << std::endl;
    
    size_t weatherCol = getColumnIndex(headers, "Weather_Condition");
    if (weatherCol == std::numeric_limits<size_t>::max()) {
        std::cerr << "Weather_Condition column not found!" << std::endl;
        return;
    }
    
    std::map<std::string, int> weatherCounts;
    int totalWithWeather = 0;
    
    for (const auto& row : data) {
        if (row.size() > weatherCol) {
            auto weatherOpt = csv::TypeConverter::convert<std::string>(row[weatherCol]);
            if (weatherOpt && !weatherOpt->empty()) {
                weatherCounts[*weatherOpt]++;
                totalWithWeather++;
            }
        }
    }
    
    // Find top 10 weather conditions by accident count
    std::vector<std::pair<std::string, int>> weatherCountVec(weatherCounts.begin(), weatherCounts.end());
    std::sort(weatherCountVec.begin(), weatherCountVec.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    std::cout << "Top 10 weather conditions by accident count:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(10), weatherCountVec.size()); ++i) {
        std::cout << i + 1 << ". " << weatherCountVec[i].first << ": " 
                  << weatherCountVec[i].second << " accidents (" 
                  << std::fixed << std::setprecision(2) 
                  << (100.0 * weatherCountVec[i].second / totalWithWeather) << "%)" << std::endl;
    }
}

// Function to analyze accidents by time of day
void analyzeAccidentsByTime(const std::vector<std::vector<csv::FieldType>>& data, 
                           const std::vector<std::string>& headers) {
    std::cout << "\nAnalyzing accidents by time of day..." << std::endl;
    
    size_t timeCol = getColumnIndex(headers, "Start_Time");
    if (timeCol == std::numeric_limits<size_t>::max()) {
        std::cerr << "Start_Time column not found!" << std::endl;
        return;
    }
    
    std::vector<int> hourCounts(24, 0);
    int totalWithTime = 0;
    
    for (const auto& row : data) {
        if (row.size() > timeCol) {
            auto timeOpt = csv::TypeConverter::convert<std::string>(row[timeCol]);
            if (timeOpt && timeOpt->length() >= 13) { // Format: YYYY-MM-DD HH:MM:SS
                std::string hourStr = timeOpt->substr(11, 2);
                try {
                    int hour = std::stoi(hourStr);
                    if (hour >= 0 && hour < 24) {
                        hourCounts[hour]++;
                        totalWithTime++;
                    }
                } catch (...) {
                    // Skip invalid time formats
                }
            }
        }
    }
    
    std::cout << "Accidents by hour of day:" << std::endl;
    for (int hour = 0; hour < 24; ++hour) {
        std::cout << std::setw(2) << std::setfill('0') << hour << ":00-" 
                  << std::setw(2) << std::setfill('0') << hour << ":59: " 
                  << hourCounts[hour] << " accidents (" 
                  << std::fixed << std::setprecision(2) 
                  << (100.0 * hourCounts[hour] / totalWithTime) << "%)" << std::endl;
    }
    
    // Find peak hours
    auto peakHour = std::max_element(hourCounts.begin(), hourCounts.end()) - hourCounts.begin();
    std::cout << "\nPeak accident hour: " << std::setw(2) << std::setfill('0') << peakHour 
              << ":00-" << std::setw(2) << std::setfill('0') << peakHour << ":59" << std::endl;
}

// Function to demonstrate streaming processing with callbacks
void demonstrateStreamingProcessing(const std::string& filename) {
    std::cout << "\nDemonstrating streaming processing with callbacks..." << std::endl;
    
    csv::CSVParser parser;
    
    // Statistics to collect
    std::map<std::string, int> stateCounts;
    std::map<int, int> severityCounts;
    size_t totalRows = 0;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    parser.parseFileWithCallback(filename, [&](std::vector<csv::FieldType>& row, size_t lineNum) {
        // Skip header row
        if (lineNum == 1) {
            return;
        }
        
        totalRows++;
        
        // Process state (assuming column 16 is State)
        if (row.size() > 16) {
            auto stateOpt = csv::TypeConverter::convert<std::string>(row[16]);
            if (stateOpt) {
                stateCounts[*stateOpt]++;
            }
        }
        
        // Process severity (assuming column 3 is Severity)
        if (row.size() > 3) {
            auto severityOpt = csv::TypeConverter::convert<int>(row[3]);
            if (severityOpt) {
                severityCounts[*severityOpt]++;
            }
        }
        
        // Print progress every 1 million rows
        if (totalRows % 1000000 == 0) {
            std::cout << "Processed " << totalRows << " rows..." << std::endl;
        }
    });
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    
    std::cout << "Processed " << totalRows << " rows in " << duration << " ms" << std::endl;
    std::cout << "Processing speed: " << (totalRows * 1000.0 / duration) << " rows/second" << std::endl;
    
    // Print top 5 states
    std::vector<std::pair<std::string, int>> stateCountVec(stateCounts.begin(), stateCounts.end());
    std::sort(stateCountVec.begin(), stateCountVec.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    std::cout << "\nTop 5 states by accident count:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), stateCountVec.size()); ++i) {
        std::cout << i + 1 << ". " << stateCountVec[i].first << ": " 
                  << stateCountVec[i].second << " accidents" << std::endl;
    }
    
    // Print severity distribution
    std::cout << "\nSeverity distribution:" << std::endl;
    for (const auto& [severity, count] : severityCounts) {
        std::cout << "Severity " << severity << ": " << count << " accidents (" 
                  << std::fixed << std::setprecision(2) 
                  << (100.0 * count / totalRows) << "%)" << std::endl;
    }
}

// Function to benchmark parser performance
void benchmarkParser(const std::string& filename) {
    std::cout << "\nBenchmarking CSV parser performance..." << std::endl;
    
    // Test with different chunk sizes and thread counts
    std::vector<size_t> chunkSizes = {1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024};
    std::vector<size_t> threadCounts = {1, 2, 4, 8};
    
    for (size_t chunkSize : chunkSizes) {
        for (size_t threadCount : threadCounts) {
            csv::ParserOptions options;
            options.chunkSize = chunkSize;
            options.maxThreads = threadCount;
            
            csv::CSVParser parser(options);
            
            auto startTime = std::chrono::high_resolution_clock::now();
            
            // Use callback-based parsing to minimize memory usage
            size_t rowCount = 0;
            parser.parseFileWithCallback(filename, [&](std::vector<csv::FieldType>&, size_t lineNum) {
                if (lineNum > 1) { // Skip header
                    rowCount++;
                }
            });
            
            auto endTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
            
            std::cout << "Chunk size: " << (chunkSize / (1024 * 1024)) << " MB, Threads: " << threadCount
                      << ", Time: " << duration << " ms, Speed: " << (rowCount * 1000.0 / duration)
                      << " rows/second" << std::endl;
        }
    }
}

int main() {
    const std::string filename = "/Users/yawasante/Documents/Doctrate/Thesis/C++/second_paper/with_returns/Xperimental/utils/US_Accidents_Dec20.csv";
    
    try {
        std::cout << "=== US Accidents CSV Parser Demo ===" << std::endl;
        
        // Check if file exists
        std::ifstream fileCheck(filename);
        if (!fileCheck) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return 1;
        }
        fileCheck.close();
        
        std::cout << "File exists and can be opened." << std::endl;
        
        // Create parser with default options
        csv::ParserOptions options;
        options.detectTypes = true;  // Enable type detection
        options.trimSpaces = true;   // Trim spaces from fields
        options.maxThreads = 1;      // Use single thread for debugging
        csv::CSVParser parser(options);
        
        std::cout << "\nReading sample of data (first 5 rows)..." << std::endl;
        
        // Use a simpler approach first - read the whole file
        try {
            std::cout << "Attempting to read file directly..." << std::endl;
            auto allData = parser.parseFile(filename);
            std::cout << "Successfully read " << allData.size() << " rows." << std::endl;
            
            if (!allData.empty()) {
                std::cout << "First row has " << allData[0].size() << " columns." << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error reading file directly: " << e.what() << std::endl;
        }
        
        // Now try with callback
        std::cout << "\nTrying with callback approach..." << std::endl;
        std::vector<std::vector<csv::FieldType>> sampleData;
        std::vector<std::string> headers;
        bool headerProcessed = false;
        
        try {
            // First, explicitly read the header row
            std::cout << "Reading header row..." << std::endl;
            std::ifstream headerFile(filename);
            if (headerFile.is_open()) {
                std::string headerLine;
                if (std::getline(headerFile, headerLine)) {
                    // Parse the header line manually
                    std::istringstream ss(headerLine);
                    std::string token;
                    
                    while (std::getline(ss, token, ',')) {
                        // Remove quotes if present
                        if (!token.empty() && token.front() == '"' && token.back() == '"') {
                            token = token.substr(1, token.size() - 2);
                        }
                        headers.push_back(token);
                    }
                    
                    headerProcessed = true;
                    std::cout << "Manually extracted " << headers.size() << " headers" << std::endl;
                }
                headerFile.close();
            }
            
            // Now read the data rows
            parser.parseFileWithCallback(filename, [&](std::vector<csv::FieldType>& row, size_t lineNum) {
                if (lineNum > 1 && lineNum <= 6) { // Skip header, get first 5 data rows
                    sampleData.push_back(row);
                }
            });
            
            if (!headerProcessed) {
                std::cerr << "Warning: Header row was not processed" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in callback parsing: " << e.what() << std::endl;
        }
        
        // Print headers
        std::cout << "\nCSV Headers (" << headers.size() << " columns):" << std::endl;
        for (size_t i = 0; i < headers.size(); ++i) {
            std::cout << i << ": " << headers[i] << std::endl;
        }
        
        // Print sample data
        std::cout << "\nSample data (first 5 rows):" << std::endl;
        if (sampleData.empty()) {
            std::cout << "No sample data was collected." << std::endl;
        } else {
            for (size_t i = 0; i < sampleData.size(); ++i) {
                std::cout << "\nRow " << (i + 1) << ":" << std::endl;
                printRow(sampleData[i], headers);
            }
        }
        
        // Check if we have enough data to continue
        if (headers.empty()) {
            std::cerr << "Error: No headers were extracted. Cannot continue with analysis." << std::endl;
            return 1;
        }
        
        // Demonstrate streaming processing
        try {
            demonstrateStreamingProcessing(filename);
        } catch (const std::exception& e) {
            std::cerr << "Error in streaming processing: " << e.what() << std::endl;
            // Continue with the rest of the program
        }
        
        // Load a larger sample for analysis (first 100,000 rows)
        std::cout << "\nLoading larger sample for analysis (first 100,000 rows)..." << std::endl;
        
        std::vector<std::vector<csv::FieldType>> analysisData;
        size_t rowCount = 0;
        
        auto startTime = std::chrono::high_resolution_clock::now();
        
        try {
            parser.parseFileWithCallback(filename, [&](std::vector<csv::FieldType>& row, size_t lineNum) {
                if (lineNum > 1 && lineNum <= 100001) { // Skip header, get 100,000 rows
                    analysisData.push_back(row);
                    rowCount++;
                    
                    if (rowCount % 10000 == 0) {
                        std::cout << "Loaded " << rowCount << " rows..." << std::endl;
                    }
                }
            });
            
            auto endTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
            
            std::cout << "Loaded " << analysisData.size() << " rows in " << duration << " ms" << std::endl;
            
            // Perform analysis on the sample data if we have data
            if (analysisData.empty()) {
                std::cerr << "Warning: No analysis data was loaded." << std::endl;
            } else {
                analyzeAccidentSeverity(analysisData, headers);
                analyzeAccidentsByState(analysisData, headers);
                analyzeAccidentsByWeather(analysisData, headers);
                analyzeAccidentsByTime(analysisData, headers);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error loading or analyzing data: " << e.what() << std::endl;
        }
        
        // Benchmark parser performance
        try {
            benchmarkParser(filename);
        } catch (const std::exception& e) {
            std::cerr << "Error during benchmarking: " << e.what() << std::endl;
        }
        
        std::cout << "\nCSV Parser demo completed successfully!" << std::endl;
        
    } catch (const csv::ParseError& e) {
        std::cerr << "CSV parsing error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
