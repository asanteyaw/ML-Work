#include "file_loader.h"
#include "dataframe.h"

#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <ranges>
#include <algorithm>
#include <charconv>
#include <unordered_map>

namespace pluss::table {

   std::shared_ptr<DataFrame> CSVLoader::load(const std::string& filename) const {
      std::ifstream file(filename);
      if (!file.is_open()) {
          throw std::runtime_error("Error: Could not open file " + filename);
      }

      std::vector<std::string> column_names;
      std::string line;

      // ✅ Read header (column names)
      if (std::getline(file, line)) {
          std::stringstream header_stream(line);
          std::ranges::copy(
              std::views::split(line, ',') | std::views::transform([](auto&& rng) {
                  return std::string(rng.begin(), rng.end());
              }),
              std::back_inserter(column_names));
      }

      size_t num_cols = column_names.size();
      std::unordered_map<std::string, std::vector<std::string>> raw_data;
      std::vector<std::string> file_lines;

      while (std::getline(file, line)) {
          file_lines.push_back(line);
      }
      file.close();

      for (const auto& row : file_lines) {
          std::vector<std::string> row_values;
          std::ranges::copy(
              std::views::split(row, ',') | std::views::transform([](auto&& rng) {
                  return std::string(rng.begin(), rng.end());
              }),
              std::back_inserter(row_values));

          if (row_values.size() != num_cols) {
              throw std::runtime_error("Row has fewer columns than expected.");
          }

          for (size_t col = 0; col < num_cols; ++col) {
              raw_data[column_names[col]].push_back(row_values[col]);
          }
      }

      auto df = std::make_shared<DataFrame>();
      for (const auto& [col_name, values] : raw_data) {
          bool is_integer = std::ranges::all_of(values, [](const std::string& val) {
              return val.find('.') == std::string::npos && val.find('e') == std::string::npos;
          });

          if (is_integer) {
              std::vector<int32_t> int_values;
              for (const auto& val : values) {
                  try {
                      int_values.push_back(std::stoi(val));
                  } catch (...) {
                      int_values.push_back(0);
                  }
              }
              df = df->set_col(col_name, torch::tensor(int_values, torch::kInt32));
          } else {
              std::vector<float> float_values;
              for (const auto& val : values) {
                  try {
                      float_values.push_back(std::stof(val));
                  } catch (...) {
                      float_values.push_back(NAN);
                  }
              }
              df = df->set_col(col_name, torch::tensor(float_values, torch::kFloat32));
          }
      }

      return df;
  }

// std::shared_ptr<DataFrame> CSVLoader::load(const std::string& filename) const {
//    std::ifstream file(filename);
//    if (!file.is_open()) throw std::runtime_error("Error: Could not open file " + filename);

//    std::string line;
//    std::vector<std::string> headers;
//    if (std::getline(file, line)) {
//        std::ranges::transform(std::views::split(line, ','), std::back_inserter(headers), [](auto&& rng) { return std::string(rng.begin(), rng.end()); });
//    }

//    std::vector<std::string> lines;
//    std::ranges::copy(std::istream_iterator<std::string>(file), std::istream_iterator<std::string>(), std::back_inserter(lines));
//    file.close();

//    std::unordered_map<std::string, std::vector<std::string>> raw_data;
//    std::vector<std::string> column_order = headers;  // Populate column_order immediately

//    for (const auto& row : lines) {
//        std::vector<std::string> values;
//        std::ranges::transform(std::views::split(row, ','), std::back_inserter(values), [](auto&& rng) { return std::string(rng.begin(), rng.end()); });
//        std::ranges::for_each(std::views::iota(size_t{0}, headers.size()), [&](size_t i) { raw_data[headers[i]].push_back(values[i]); });
//    }

//    std::unordered_map<std::string, torch::Tensor> tensors;
//    for (const auto& [col, values] : raw_data) {
//        bool is_integer = std::ranges::all_of(values, [](const std::string& val) { return val.find('.') == std::string::npos && val.find('e') == std::string::npos && val.find('E') == std::string::npos; });
//        if (is_integer) {
//            std::vector<int32_t> int_values;
//            std::ranges::transform(values, std::back_inserter(int_values), [](const std::string& val) { return std::stoi(val); });
//            tensors[col] = torch::tensor(int_values, torch::kInt32);
//        } else {
//            std::vector<float> float_values;
//            std::ranges::transform(values, std::back_inserter(float_values), [](const std::string& val) { return std::stof(val); });
//            tensors[col] = torch::tensor(float_values, torch::kFloat32);
//        }
//    }

//    return std::make_shared<DataFrame>(tensors, column_order);
// }

  
  std::shared_ptr<DataFrame> ExcelLoader::load([[maybe_unused]] const std::string& filename) const {
      throw std::runtime_error("Excel loading not implemented yet.");
  }
  
  std::shared_ptr<DataFrame> ParquetLoader::load([[maybe_unused]] const std::string& filename) const {
      throw std::runtime_error("Parquet loading not implemented yet.");
  }
  
  std::shared_ptr<DataFrame> RDBMSLoader::load([[maybe_unused]] const std::string& filename) const {
      throw std::runtime_error("RDBMS loading not implemented yet.");
  }

} // namespace pluss::table