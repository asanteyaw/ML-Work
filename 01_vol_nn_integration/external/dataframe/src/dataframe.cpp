#include "dataframe.h"
#include "file_loader.h"

#include <iostream>              // Input/output
#include <fstream>               // File handling (CSV reading)
#include <sstream>               // String stream for parsing
#include <iomanip>               // Formatting for `head()`
#include <ranges>                // C++20 ranges & views
#include <algorithm>             // std::ranges::copy, std::ranges::transform
#include <iterator> 
#include <algorithm>


namespace pluss::table {

    // Declare static functions
static std::unordered_map<std::string, std::unique_ptr<FileLoader>> createLoaders();

// ✅ Constructor from unordered_map
DataFrame::DataFrame(const std::unordered_map<std::string, torch::Tensor>& data) : data(data) {}

DataFrame::DataFrame(const std::unordered_map<std::string, torch::Tensor>& data, const std::vector<std::string>& column_order)
    : data(data), column_order(column_order) {}

std::shared_ptr<DataFrame> DataFrame::load(const std::string& format, const std::string& filename) {
    static auto loaders = createLoaders();

    // Find the loader based on the format
    auto it = loaders.find(format);
    if (it == loaders.end()) {
        throw std::invalid_argument("Unsupported format: " + format);
    }

    // Validate file extension
    std::string ext = filename.substr(filename.find_last_of('.') + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext != it->second->extension()) {
        throw std::invalid_argument("File extension does not match the specified format '" + format + "'");
    }

    // Delegate the file loading to the appropriate loader
    return it->second->load(filename);
}

// ✅ Read CSV File
std::shared_ptr<DataFrame> DataFrame::read_csv(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file " + filename);
    }

    auto df = std::make_shared<DataFrame>();
    std::string line;

    // ✅ Step 1: Read column names
    if (!std::getline(file, line)) {
        throw std::runtime_error("Error: Empty CSV file!");
    }

    std::vector<std::string> column_names;
    std::stringstream header_stream(line);
    std::string cell;
    while (std::getline(header_stream, cell, ',')) {
        column_names.push_back(cell);
    }
    size_t num_cols = column_names.size();

    // ✅ Step 2: Read first row and infer column types
    std::vector<bool> is_integer(num_cols, true);
    std::vector<std::vector<int32_t>> int_columns(num_cols);
    std::vector<std::vector<float>> float_columns(num_cols);

    if (std::getline(file, line)) {
        std::stringstream line_stream(line);
        size_t col_idx = 0;
        while (std::getline(line_stream, cell, ',')) {
            if (col_idx >= num_cols) continue;

            bool has_decimal = cell.find('.') != std::string::npos || cell.find('e') != std::string::npos || cell.find('E') != std::string::npos;
            is_integer[col_idx] = !has_decimal;

            if (is_integer[col_idx]) {
                int_columns[col_idx].push_back(std::stoi(cell));
            } else {
                float_columns[col_idx].push_back(std::stof(cell));
            }

            col_idx++;
        }
    }

    // ✅ Step 3: Read remaining rows
    while (std::getline(file, line)) {
        std::stringstream line_stream(line);
        size_t col_idx = 0;
        while (std::getline(line_stream, cell, ',')) {
            if (col_idx >= num_cols) continue;

            if (is_integer[col_idx]) {
                int_columns[col_idx].push_back(std::stoi(cell));
            } else {
                float_columns[col_idx].push_back(std::stof(cell));
            }

            col_idx++;
        }
    }
    file.close();

    // ✅ Step 4: Convert to tensors
    for (size_t i = 0; i < num_cols; ++i) {
        if (is_integer[i]) {
            df->data[column_names[i]] = torch::tensor(int_columns[i], torch::kInt32);
        } else {
            df->data[column_names[i]] = torch::tensor(float_columns[i], torch::kFloat32);
        }
    }

    return df;
}


torch::Tensor DataFrame::read_matrix(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file " + filename);
    }

    std::vector<std::vector<float>> data;
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream line_stream(line);
        std::vector<float> row;
        std::string cell;
        while (std::getline(line_stream, cell, ',')) {
            row.push_back(std::stof(cell));
        }
        data.push_back(row);
    }
    file.close();

    size_t num_rows = data.size();
    size_t num_cols = data[0].size();

    torch::Tensor tensor_data = torch::empty({static_cast<long>(num_rows), static_cast<long>(num_cols)}, torch::kFloat32);
    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < num_cols; ++j) {
            tensor_data[i][j] = data[i][j];
        }
    }

    return tensor_data;
}

void DataFrame::head(int64_t n) const {
    if (data.empty()) {
        std::cout << "Empty DataFrame\n";
        return;
    }

    const int col_width = 12;
    int terminal_width = 200;  // Fallback to 100 columns if detection fails

    if (std::getenv("COLUMNS") != nullptr) {
        terminal_width = std::stoi(std::getenv("COLUMNS"));
    }

    int max_cols = terminal_width / col_width;

    std::vector<std::string> col_names = this->columns();  // Use existing method for order

    // Print column headers
    for (size_t i = 0; i < std::min(col_names.size(), static_cast<size_t>(max_cols)); ++i) {
        std::cout << std::setw(col_width) << std::left << col_names[i] << " ";
    }
    std::cout << "\n" << std::string(col_width * std::min(col_names.size(), static_cast<size_t>(max_cols)), '-') << "\n";

    // Print data rows
    int64_t rows_to_print = std::min(n, data.begin()->second.size(0));
    for (int64_t i = 0; i < rows_to_print; ++i) {
        for (size_t j = 0; j < std::min(col_names.size(), static_cast<size_t>(max_cols)); ++j) {
            const auto& tensor = data.at(col_names[j]);
            float value = tensor[i].item<float>();
            std::cout << std::setw(col_width) << std::fixed << std::setprecision((value == std::floor(value)) ? 0 : 4) << value << " ";
        }
        std::cout << "\n";
    }
}

// ✅ Print Last `n` Rows
void DataFrame::tail(int64_t n) const {
    if (data.empty()) {
        std::cout << "Empty DataFrame\n";
        return;
    }

    int64_t row_start = std::max<int64_t>(0, data.begin()->second.size(0) - n);
    auto df_tail = this->iloc(row_start, data.begin()->second.size(0));
    df_tail->head(n);  // Reuse `head()` for printing
}

// Add info() method to DataFrame
void DataFrame::info() const {
    std::cout << "<class 'pluss::table::DataFrame'>\n";
    size_t num_rows = data.empty() ? 0 : data.begin()->second.size(0);
    std::cout << "Index: 0 to " << (num_rows == 0 ? 0 : num_rows - 1) << "\n";
    std::cout << "Data columns (total " << data.size() << "):\n";

    // Print table header
    std::cout << std::left << std::setw(5) << "#" << std::setw(20) << "Column" << std::setw(20) << "Non-Null Count" << std::setw(10) << "Dtype" << "\n";
    std::cout << std::string(55, '-') << "\n";

    int idx = 0;
    if (!column_order.empty()) {
        for (const auto& col : column_order) {
            if (data.find(col) != data.end()) {
                const auto& tensor = data.at(col);
                std::cout << std::setw(5) << idx++ << std::setw(20) << col << std::setw(20) << tensor.numel() << std::setw(10) << (tensor.dtype() == torch::kInt32 ? "int32" : "float32") << "\n";
            }
        }
    } else {
        for (const auto& [col, tensor] : data) {
            std::cout << std::setw(5) << idx++ << std::setw(20) << col << std::setw(20) << tensor.numel() << std::setw(10) << (tensor.dtype() == torch::kInt32 ? "int32" : "float32") << "\n";
        }
    }

    size_t total_memory = 0;
    for (const auto& [_, tensor] : data) {
        total_memory += tensor.nbytes();
    }
    std::cout << "memory usage: " << total_memory / 1024.0 << " KB\n";
}

// DataFrame methods for describe and transpose
void DataFrame::describe() const {
    if (data.empty()) {
        std::cout << "Empty DataFrame\n";
        return;
    }

    std::vector<std::string> stat_cols = {"column", "count", "mean", "std", "min", "25%", "50%", "75%", "max"};
    std::unordered_map<std::string, std::vector<std::variant<std::string, float>>> stat_data;

    for (const auto& [col, tensor] : data) {
        auto values = tensor.to(torch::kFloat32);
        int64_t count = values.numel();
        float mean = values.mean().item<float>();
        float stddev = values.std().item<float>();
        float min = values.min().item<float>();
        float q25 = values.quantile(0.25).item<float>();
        float median = values.median().item<float>();
        float q75 = values.quantile(0.75).item<float>();
        float max = values.max().item<float>();

        stat_data[col] = {col, (float)count, mean, stddev, min, q25, median, q75, max};
    }

    // Display
    const int col_width = 14;
    std::cout << std::setw(col_width) << std::right << stat_cols[0];
    for (size_t i = 1; i < stat_cols.size(); ++i) {
        std::cout << std::setw(col_width) << std::right << stat_cols[i];
    }
    std::cout << "\n" << std::string(col_width * stat_cols.size(), '-') << "\n";

    for (const auto& [col, values] : stat_data) {
        std::cout << std::setw(col_width) << std::right << std::get<std::string>(values[0]);
        for (size_t i = 1; i < values.size(); ++i) {
            std::cout << std::setw(col_width) << std::scientific << std::setprecision(4) << std::get<float>(values[i]);
        }
        std::cout << "\n";
    }
}

// Corrected DataFrame::T method
std::shared_ptr<DataFrame> DataFrame::T() const {
    auto transposed = std::make_shared<DataFrame>();
    if (data.empty()) return transposed;

    size_t num_rows = data.begin()->second.size(0);
    size_t num_cols = data.size();

    for (size_t i = 0; i < num_rows; ++i) {
        std::vector<float> row_values;
        for (const auto& col : columns()) {
            row_values.push_back(data.at(col)[i].item<float>());
        }
        std::string row_name = "Row_" + std::to_string(i);
        transposed->data[row_name] = torch::tensor(row_values);
    }

    transposed->column_order.clear();
    for (size_t i = 0; i < num_rows; ++i) {
        transposed->column_order.push_back("Row_" + std::to_string(i));
    }

    return transposed;
}

std::vector<torch::Tensor> DataFrame::values() const {
    std::vector<torch::Tensor> vals;
    for (const auto& [_, tensor] : data) {
        vals.push_back(tensor);
    }
    return vals;
}

std::shared_ptr<DataFrame> DataFrame::set_cols(const std::vector<std::string>& column_names, const torch::TensorList& values) const {
    TORCH_CHECK(column_names.size() == values.size(), "Error: Number of column names must match number of tensors!");
    for (const auto& col_name : column_names) {
        TORCH_CHECK(!col_name.empty(), "Error: Column names cannot be empty!");
    }
    return set_columns(column_names, values); // Delegate to private method
}

std::shared_ptr<DataFrame> DataFrame::set_col(const std::string& col_name, const torch::Tensor& values) const {
    TORCH_CHECK(!col_name.empty(), "Error: Column name cannot be empty!");
    TORCH_CHECK(values.dim() == 1, "Error: Column tensor must be 1D!");
    return set_column(col_name, values); // Delegate to private method
}

// ✅ Get column names
std::vector<std::string> DataFrame::columns() const {
    std::vector<std::string> col_names;
    for (const auto& [col_name, _] : data) {
        col_names.push_back(col_name);
    }
    return col_names;
}

// ✅ Overloaded Operators for Column Selection
torch::Tensor DataFrame::get_col(const std::string& col_name) const {
    return select_one(col_name);
}

std::shared_ptr<DataFrame> DataFrame::get_cols(const std::vector<std::string>& column_names) const {
    return select_many(column_names);
}

// ✅ Move to a New Device
std::shared_ptr<DataFrame> DataFrame::to(const torch::Device& device) const {
    auto new_df = std::make_shared<DataFrame>();
    for (const auto& [key, tensor] : data) {
        new_df->data[key] = tensor.to(device);
    }
    return new_df;
}

// ✅ Save to CSV
void DataFrame::to_csv(const std::string& filename) const {
    std::ofstream file(filename);
    for (const auto& [key, _] : data) {
        file << key << ",";
    }
    file << "\n";

    int64_t num_rows = data.begin()->second.size(0);
    for (int64_t i = 0; i < num_rows; ++i) {
        for (const auto& [_, tensor] : data) {
            file << tensor[i].item<float>() << ",";
        }
        file << "\n";
    }
    file.close();
}

// ✅ Conditional filtering (boolean mask)
std::shared_ptr<DataFrame> DataFrame::loc(const torch::Tensor& condition) const {
    auto new_df = std::make_shared<DataFrame>();
    for (const auto& [col_name, tensor] : data) {
        new_df->data[col_name] = tensor.index({condition});
    }
    return new_df;
}

// ✅ Index-based selection (multi-row slice)
std::shared_ptr<DataFrame> DataFrame::iloc(int64_t row_start, int64_t row_end) const {
    auto new_df = std::make_shared<DataFrame>();
    for (const auto& [col_name, tensor] : data) {
        new_df->data[col_name] = tensor.index({torch::indexing::Slice(row_start, row_end)});
    }
    return new_df;
}

// ✅ Index-based selection (single row)
std::unordered_map<std::string, torch::Tensor> DataFrame::iloc(int64_t row) const {
    std::unordered_map<std::string, torch::Tensor> row_data;
    for (const auto& [col_name, tensor] : data) {
        row_data[col_name] = tensor.index({row});
    }
    return row_data;
}

// -----------------------------------Private Helper Functions---------------------

// ✅ Select a Single Column
torch::Tensor DataFrame::select_one(const std::string& col_name) const {
    if (data.find(col_name) == data.end()) {
        throw std::runtime_error("Error: Column '" + col_name + "' not found.");
    }
    return data.at(col_name);
}

// ✅ Select Multiple Columns
std::shared_ptr<DataFrame> DataFrame::select_many(const std::vector<std::string>& column_names) const {
    std::unordered_map<std::string, torch::Tensor> selected_data;
    for (const auto& col_name : column_names) {
        selected_data[col_name] = select_one(col_name);
    }
    return std::make_shared<DataFrame>(selected_data);
}

std::shared_ptr<DataFrame> DataFrame::set_column(const std::string& col_name, const torch::Tensor& values) const {
    auto new_df = std::make_shared<DataFrame>(*this);
    new_df->data[col_name] = values;
    return new_df;
}

std::shared_ptr<DataFrame> DataFrame::set_columns(const std::vector<std::string>& column_names, const torch::TensorList& values) const {
    auto new_df = std::make_shared<DataFrame>(*this);  // Copy dataframe
    for (size_t i = 0; i < column_names.size(); ++i) {
        new_df->data[column_names[i]] = values[i];  // Directly modify each column
    }
    return new_df;
}
// ----------------------------------End Private Helper Functions------------------

// ------------------------------------Static Helper Functions---------------------

// Helper function for lazy initialization of file loaders
static std::unordered_map<std::string, std::unique_ptr<FileLoader>> createLoaders() {
    std::unordered_map<std::string, std::unique_ptr<FileLoader>> loaders;
    loaders["csv"] = std::make_unique<CSVLoader>();
    loaders["excel"] = std::make_unique<ExcelLoader>();
    loaders["parquet"] = std::make_unique<ParquetLoader>();
    loaders["rdbms"] = std::make_unique<RDBMSLoader>();
    return loaders;
}

} // namespace pluss::table



// #include "dataframe.h"
// #include <fstream>
// #include <iostream>
// #include <sstream>
// #include <iomanip>
// #include <charconv>

// namespace pluss::table {

// // Constructor from unordered_map
// DataFrame::DataFrame(const std::unordered_map<std::string, torch::Tensor>& data) {
//     for (const auto& [col_name, tensor] : data) {
//         col_order.push_back(col_name);
//         col_index[col_name] = data_storage.size();
//         data_storage.push_back(tensor);
//     }
// }

// // Read CSV file
// std::shared_ptr<DataFrame> DataFrame::read_csv(const std::string& filename) {
//     std::ifstream file(filename);
//     if (!file.is_open()) {
//         throw std::runtime_error("Error: Could not open file " + filename);
//     }

//     auto df = std::make_shared<DataFrame>();
//     std::string line;

//     // ✅ Step 1: Read column names
//     if (!std::getline(file, line)) {
//         throw std::runtime_error("Error: Empty CSV file!");
//     }

//     std::vector<std::string> column_names;
//     std::stringstream header_stream(line);
//     std::string cell;
//     while (std::getline(header_stream, cell, ',')) {
//         column_names.push_back(cell);
//     }
//     df->col_order = column_names;
//     size_t num_cols = column_names.size();

//     // ✅ Step 2: Read first row and infer column types
//     std::vector<bool> is_integer(num_cols, true);
//     std::vector<std::vector<int32_t>> int_columns(num_cols);
//     std::vector<std::vector<float>> float_columns(num_cols);

//     if (std::getline(file, line)) {
//         std::stringstream line_stream(line);
//         size_t col_idx = 0;
//         while (std::getline(line_stream, cell, ',')) {
//             if (col_idx >= num_cols) continue; // Skip extra values

//             bool has_decimal = cell.find('.') != std::string::npos || cell.find('e') != std::string::npos || cell.find('E') != std::string::npos;
//             is_integer[col_idx] = !has_decimal;  // If first value is float, mark column as float

//             if (is_integer[col_idx]) {
//                 int_columns[col_idx].push_back(to_int(cell));
//             } else {
//                 float_columns[col_idx].push_back(std::stof(cell));
//             }

//             col_idx++;
//         }
//     }

//     // ✅ Step 3: Read remaining rows
//     while (std::getline(file, line)) {
//         std::stringstream line_stream(line);
//         size_t col_idx = 0;
//         while (std::getline(line_stream, cell, ',')) {
//             if (col_idx >= num_cols) continue;

//             if (is_integer[col_idx]) {
//                 int_columns[col_idx].push_back(to_int(cell));
//             } else {
//                 float_columns[col_idx].push_back(std::stof(cell));
//             }

//             col_idx++;
//         }
//     }
//     file.close();

//     // ✅ Step 4: Convert to tensors
//     for (size_t i = 0; i < num_cols; ++i) {
//         df->col_index[column_names[i]] = i;
//         if (is_integer[i]) {
//             df->data_storage.push_back(torch::tensor(int_columns[i], torch::kInt32));
//         } else {
//             df->data_storage.push_back(torch::tensor(float_columns[i], torch::kFloat32));
//         }
//     }

//     return df;
// }

// // Set column (return new DataFrame)
// std::shared_ptr<DataFrame> DataFrame::set_column(const std::string& col_name, const torch::Tensor& values) const {
//     auto new_df = std::make_shared<DataFrame>(*this);
//     auto it = new_df->col_index.find(col_name);
//     if (it != new_df->col_index.end()) {
//         new_df->data_storage[it->second] = values;
//     } else {
//         new_df->col_order.push_back(col_name);
//         new_df->col_index[col_name] = new_df->data_storage.size();
//         new_df->data_storage.push_back(values);
//     }
//     return new_df;
// }

// // Select multiple columns
// std::shared_ptr<DataFrame> DataFrame::select_many(const std::vector<std::string>& column_names) const {
//     auto new_df = std::make_shared<DataFrame>();
//     new_df->col_order.reserve(column_names.size());  // Reserve memory to avoid dynamic allocations
//     new_df->data_storage.reserve(column_names.size());

//     for (const auto& col_name : column_names) {
//         auto it = col_index.find(col_name);
//         if (it == col_index.end()) {
//             throw std::runtime_error("Error: Column '" + col_name + "' not found in DataFrame.");
//         }

//         int col_idx = it->second;
//         if (col_idx >= data_storage.size()) {  // 🚨 Safety check
//             throw std::runtime_error("Error: Column index out of bounds for '" + col_name + "'");
//         }

//         new_df->col_order.push_back(col_name);
//         new_df->col_index[col_name] = new_df->data_storage.size();
//         new_df->data_storage.push_back(data_storage[col_idx].clone());  // ✅ Use `.clone()` to avoid aliasing
//     }

//     return new_df;
// }

// // Select single column
// torch::Tensor DataFrame::select_one(const std::string& col_name) const {
//     auto it = col_index.find(col_name);
//     if (it == col_index.end()) {
//         throw std::runtime_error("Error: Column '" + col_name + "' not found in DataFrame.");
//     }
//     return data_storage[it->second];
// }

// // Single column selection
// torch::Tensor DataFrame::operator[](const std::string& col_name) const {
//     return select_one(col_name);  // Calls existing select() function
// }

// // Alternative accessor (alias for operator[])
// torch::Tensor DataFrame::at(const std::string& col_name) const {
//     return this->operator[](col_name);
// }

// // Multiple column selection
// std::shared_ptr<DataFrame> DataFrame::operator[](const std::vector<std::string>& column_names) const {
//     return select_many(column_names);
// }

// // Alternative accessor for multiple columns
// std::shared_ptr<DataFrame> DataFrame::at(const std::vector<std::string>& column_names) const {
//     return this->operator[](column_names);
// }

// // Move DataFrame to a different device
// std::shared_ptr<DataFrame> DataFrame::to(const torch::Device& device) const {
//     auto new_df = std::make_shared<DataFrame>();
//     new_df->col_order = col_order;
//     new_df->col_index = col_index;
//     new_df->data_storage.reserve(data_storage.size());
//     for (const auto& tensor : data_storage) {
//         new_df->data_storage.push_back(tensor.to(device));
//     }
//     return new_df;
// }

// // Display first n rows
// void DataFrame::head(int64_t n) const {
//     n = std::min(n, data_storage[0].size(0));

//     // Print column headers
//     for (const auto& col : col_order) {
//         std::cout << std::setw(12) << std::left << col << " ";
//     }
//     std::cout << "\n" << std::string(12 * col_order.size(), '-') << "\n";

//     // Print first `n` rows with float precision fixed at 2 decimal places
//     for (int64_t i = 0; i < n; ++i) {
//         for (size_t j = 0; j < data_storage.size(); ++j) {
//             float value = data_storage[j][i].item<float>();
//             std::cout << std::setw(12) << std::left << std::fixed << std::setprecision(2) << value << " ";
//         }
//         std::cout << "\n";
//     }
// }

// void DataFrame::tail(int64_t n) const {
//     int64_t row_start = std::max<int64_t>(0, data_storage[0].size(0) - n);
//     auto df_tail = this->iloc(row_start, data_storage[0].size(0));
//     df_tail->head(n);  // Reuse `head()` to print the last `n` rows
// }

// std::shared_ptr<DataFrame> DataFrame::loc(const torch::Tensor& condition) const {
//     TORCH_CHECK(condition.dtype() == torch::kBool, "Error: loc() condition must be a boolean mask!");
//     TORCH_CHECK(condition.sizes() == data_storage[0].sizes(), "Error: Condition size must match row count!");

//     // Create a new DataFrame for selected rows
//     auto new_df = std::make_shared<DataFrame>();
//     new_df->col_index = col_index;
//     new_df->col_order = col_order;

//     // Apply row filtering on each column
//     for (const auto& tensor : data_storage) {
//         new_df->data_storage.push_back(tensor.index({condition}));
//     }

//     return new_df;
// }

// std::shared_ptr<DataFrame> DataFrame::iloc(int64_t row_start, int64_t row_end) const {
//     TORCH_CHECK(row_start >= 0 && row_end <= data_storage[0].size(0),
//                 "Error: iloc() indices out of range!");

//     auto new_df = std::make_shared<DataFrame>();
//     new_df->col_index = col_index;
//     new_df->col_order = col_order;

//     // Slice the requested row range for each column
//     for (const auto& tensor : data_storage) {
//         new_df->data_storage.push_back(tensor.index({torch::indexing::Slice(row_start, row_end)}));
//     }

//     return new_df;
// }

// std::unordered_map<std::string, torch::Tensor> DataFrame::iloc(int64_t row) const {
//     TORCH_CHECK(row >= 0 && row < data_storage[0].size(0), "Error: Row index out of range!");

//     std::unordered_map<std::string, torch::Tensor> row_data;
//     for (size_t i = 0; i < col_order.size(); ++i) {
//         row_data[col_order[i]] = data_storage[i].index({row});
//     }

//     return row_data;
// }

// // Save DataFrame to CSV
// void DataFrame::to_csv(const std::string& filename) const {
//     std::ofstream file(filename);
//     if (!file.is_open()) {
//         throw std::runtime_error("Error: Could not open file " + filename);
//     }
//     for (size_t i = 0; i < col_order.size(); ++i) {
//         file << col_order[i] << (i < col_order.size() - 1 ? "," : "\n");
//     }
//     for (int64_t i = 0; i < data_storage[0].size(0); ++i) {
//         for (size_t j = 0; j < data_storage.size(); ++j) {
//             file << data_storage[j][i].item<float>() << (j < data_storage.size() - 1 ? "," : "\n");
//         }
//     }
//     file.close();
// }

// // Get column names
// std::vector<std::string> DataFrame::columns() const {
//     return col_order;
// }

// } // namespace pluss::table
