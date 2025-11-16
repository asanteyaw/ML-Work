#pragma once

#include <torch/torch.h>         // LibTorch Tensors
#include <unordered_map>         // Dictionary-based storage
#include <vector>                // Standard C++ containers
#include <string>                // String operations
#include <memory> 

namespace pluss::table {

class DataFrame {
public:
    // Constructors
    DataFrame() = default;
    explicit DataFrame(const std::unordered_map<std::string, torch::Tensor>& data);
    DataFrame(const std::unordered_map<std::string, torch::Tensor>& data, const std::vector<std::string>& column_order);
    DataFrame(const DataFrame& other) = default;
    DataFrame& operator=(const DataFrame& other) = default;

    // Loader
    static std::shared_ptr<DataFrame> load(const std::string& format, const std::string& filename);

    // Read CSV
    static std::shared_ptr<DataFrame> read_csv(const std::string& filename);
    static torch::Tensor read_matrix(const std::string& filename);

    // Get column names
    [[nodiscard]] std::vector<std::string> columns() const;

    // Conditional filtering
    [[nodiscard]] std::shared_ptr<DataFrame> loc(const torch::Tensor& condition) const;

    // Index-based selection
    [[nodiscard]] std::shared_ptr<DataFrame> iloc(int64_t row_start, int64_t row_end) const;
    [[nodiscard]] std::unordered_map<std::string, torch::Tensor> iloc(int64_t row) const;

    // Move to device (CPU/GPU)
    [[nodiscard]] std::shared_ptr<DataFrame> to(const torch::Device& device) const;

    [[nodiscard]] std::vector<torch::Tensor> values() const;

    [[nodiscard]] std::shared_ptr<DataFrame> T() const;

    // Display methods
    void head(int64_t n = 5) const;
    void tail(int64_t n = 5) const;
    void info() const;
    void describe() const;

    // Single column selection (returns Tensor)
    [[nodiscard]] torch::Tensor get_col(const std::string& col_name) const;

    // Multiple column selection (returns DataFrame)
    [[nodiscard]] std::shared_ptr<DataFrame> get_cols(const std::vector<std::string>& column_names) const;

    // single column creation
    [[nodiscard]] std::shared_ptr<DataFrame> set_col(const std::string& col_name, const torch::Tensor& values) const;

    // multiple column creation
    [[nodiscard]] std::shared_ptr<DataFrame> set_cols(const std::vector<std::string>& column_names, const torch::TensorList& values) const;


    // Save CSV
    void to_csv(const std::string& filename) const;

private:
    // Data members
    std::unordered_map<std::string, torch::Tensor> data;  // Dictionary storage

    std::vector<std::string> column_order;  // Maintain column insertion order

    // Member functions
    // Data selection
    [[nodiscard]] std::shared_ptr<DataFrame> select_many(const std::vector<std::string>& column_names) const;
    [[nodiscard]] torch::Tensor select_one(const std::string& col_name) const;

    // Data creations
    [[nodiscard]] std::shared_ptr<DataFrame> set_column(const std::string& col_name, const torch::Tensor& values) const;
    [[nodiscard]] std::shared_ptr<DataFrame> set_columns(const std::vector<std::string>& column_names, const torch::TensorList& values) const;
};

} // namespace pluss::table



// ----------------------------------Most recent-----------------------------------------
// #pragma once

// #include <torch/torch.h>
// #include <unordered_map>
// #include <vector>
// #include <string>
// #include <memory>
// #include <iostream>

// namespace pluss::table {

// class DataFrame {
// public:
//     // Constructors
//     DataFrame() = default;
//     explicit DataFrame(const std::unordered_map<std::string, torch::Tensor>& data);
//     DataFrame(const DataFrame& other) = default;
//     DataFrame& operator=(const DataFrame& other) = default;

//     // Read CSV
//     static std::shared_ptr<DataFrame> read_csv(const std::string& filename);

//     // Column operations (returns new DataFrame)
//     [[nodiscard]] std::shared_ptr<DataFrame> set_column(const std::string& col_name, const torch::Tensor& values) const;

//     // Data selection
//     [[nodiscard]] std::shared_ptr<DataFrame> select_many(const std::vector<std::string>& column_names) const;
//     [[nodiscard]] torch::Tensor select_one(const std::string& col_name) const;

//     // Get column names
//     [[nodiscard]] std::vector<std::string> columns() const;

//     // conditional filtering
//     [[nodiscard]] std::shared_ptr<DataFrame> loc(const torch::Tensor& condition) const;

//     // Index-based selection
//     [[nodiscard]] std::shared_ptr<DataFrame> iloc(int64_t row_start, int64_t row_end) const;
//     [[nodiscard]] std::unordered_map<std::string, torch::Tensor> iloc(int64_t row) const;

//     // Move to device (CPU/GPU)
//     [[nodiscard]] std::shared_ptr<DataFrame> to(const torch::Device& device) const;

//     // Display methods
//     void head(int64_t n = 5) const;
//     void tail(int64_t n = 5) const;

//     // Single column selection (returns Tensor)
//     [[nodiscard]] torch::Tensor operator[](const std::string& col_name) const;
//     [[nodiscard]] torch::Tensor at(const std::string& col_name) const;

//     // Multiple column selection (returns DataFrame)
//     [[nodiscard]] std::shared_ptr<DataFrame> operator[](const std::vector<std::string>& column_names) const;
//     [[nodiscard]] std::shared_ptr<DataFrame> at(const std::vector<std::string>& column_names) const;

//     // Save CSV
//     void to_csv(const std::string& filename) const;

// private:
//     std::unordered_map<std::string, int> col_index;
//     std::vector<std::string> col_order;
//     std::vector<torch::Tensor> data_storage;
// };

// } // namespace pluss::table
// --------------------------------------- end most recent------------------------------------------

// Another one before most recent

// #pragma once

// #include <torch/torch.h>
// #include <unordered_map>
// #include <deque>
// #include <vector>
// #include <string>
// #include <iostream>
// #include <fstream>
// #include <sstream>

// namespace pluss::table {

// class DataFrame {
// public:

//     // Constructors
//     DataFrame();  // Create an empty DataFrame
//     DataFrame(const std::unordered_map<std::string, torch::Tensor>& data);  // Create a DataFrame from an initializer list
//     // Explicit copy constructor
//     DataFrame(const DataFrame& other);

//     // Explicit move constructor
//     DataFrame(DataFrame&& other) noexcept;

//     // Read CSV file
//     static DataFrame read_csv(const std::string& filename);
//     // Read Matrix file
//     static DataFrame read_matrix(const std::string& filename);

//     // Display methods
//     void head(int64_t n = 5) const;
//     void tail(int64_t n = 5) const;

//     // Column operations
//     torch::Tensor& operator[](const std::string& col_name);
//     DataFrame operator[](std::initializer_list<std::string> column_names) const;
//     DataFrame operator[](const std::vector<std::string>& column_names) const;
//     DataFrame to_float32() const;
//     DataFrame to(const torch::Device& device) const;
//     DataFrame set_column(const std::string& col_name, const torch::Tensor& values) const;
//     size_t get_data_storage_size() const { return data_storage.size(); }

//     // Row and column selection
//     torch::Tensor to_tensor() const;
//     torch::Tensor iloc_t(int64_t row_start, int64_t row_end, int64_t col_start, int64_t col_end) const;
//     DataFrame iloc(int64_t row_index) const;
//     DataFrame iloc(torch::indexing::Slice, int64_t col_index) const;
//     torch::Tensor iloc(int64_t row_index, int64_t col_index) const;
//     DataFrame iloc(int64_t row_start, int64_t row_end, int64_t col_start, int64_t col_end) const;
//     DataFrame loc(const torch::Tensor& condition) const;

//     // Temporary debugging functions
//     std::vector<std::string> get_column_names() const;
//     std::unordered_map<std::string, int> get_column_index() const;

//     // save csv file
//     void to_csv(const std::string& filename) const;

// private:
//     std::unordered_map<std::string, int> col_index;
//     std::deque<std::string> col_order;
//     std::vector<torch::Tensor> data_storage;
// };

// } // namespace pluss::table

