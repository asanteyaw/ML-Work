// // dataframe.ixx
// export module dataframe;

// // Import your own modules
// import file_loader;

// // Include standard library headers (still required)
// #include <torch/torch.h>
// #include <iostream>
// #include <memory>
// #include <string>
// #include <unordered_map>
// #include <algorithm>

// namespace pluss::table {

// class DataFrame {
// public:
//    // Constructors
//    DataFrame() = default;
//    explicit DataFrame(const std::unordered_map<std::string, torch::Tensor>& data);
//    DataFrame(const DataFrame& other) = default;
//    DataFrame& operator=(const DataFrame& other) = default;

//    // loader
//    static std::shared_ptr<DataFrame> load(const std::string& format, const std::string& filename);

//    // Read CSV
//    static std::shared_ptr<DataFrame> read_csv(const std::string& filename);

//    // Get column names
//    [[nodiscard]] std::vector<std::string> columns() const;

//    // Conditional filtering
//    [[nodiscard]] std::shared_ptr<DataFrame> loc(const torch::Tensor& condition) const;

//    // Index-based selection
//    [[nodiscard]] std::shared_ptr<DataFrame> iloc(int64_t row_start, int64_t row_end) const;
//    [[nodiscard]] std::unordered_map<std::string, torch::Tensor> iloc(int64_t row) const;

//    // Move to device (CPU/GPU)
//    [[nodiscard]] std::shared_ptr<DataFrame> to(const torch::Device& device) const;

//    // Display methods
//    void head(int64_t n = 5) const;
//    void tail(int64_t n = 5) const;

//    // Single column selection (returns Tensor)
//    [[nodiscard]] torch::Tensor get_col(const std::string& col_name) const;

//    // Multiple column selection (returns DataFrame)
//    [[nodiscard]] std::shared_ptr<DataFrame> get_cols(const std::vector<std::string>& column_names) const;

//    // single column creation
//    [[nodiscard]] std::shared_ptr<DataFrame> set_col(const std::string& col_name, const torch::Tensor& values) const;

//    // multiple column creation
//    [[nodiscard]] std::shared_ptr<DataFrame> set_cols(const std::vector<std::string>& column_names, const torch::TensorList& values) const;


//    // Save CSV
//    void to_csv(const std::string& filename) const;

// private:
//     // Data members
//    std::unordered_map<std::string, torch::Tensor> data;  // Dictionary storage

//    // Member functions
//    // Data selection
//    [[nodiscard]] std::shared_ptr<DataFrame> select_many(const std::vector<std::string>& column_names) const;
//    [[nodiscard]] torch::Tensor select_one(const std::string& col_name) const;

//    // Data creations
//    [[nodiscard]] std::shared_ptr<DataFrame> set_column(const std::string& col_name, const torch::Tensor& values) const;
//    [[nodiscard]] std::shared_ptr<DataFrame> set_columns(const std::vector<std::string>& column_names, const torch::TensorList& values) const;
// };

// } // namespace pluss::table


