#pragma once

#include <memory>
#include <string>

namespace pluss::table {

class DataFrame;

class FileLoader {
public:
    virtual std::shared_ptr<DataFrame> load(const std::string& filename) const = 0;
    virtual std::string extension() const = 0;
    virtual ~FileLoader() = default;
};

// Concrete Loaders
class CSVLoader : public FileLoader {
    public:
    [[nodiscard]] std::shared_ptr<DataFrame> load(const std::string& filename) const override;
        std::string extension() const override { return "csv"; }
    };
    
class ExcelLoader : public FileLoader {
public:
    std::shared_ptr<DataFrame> load(const std::string& filename) const override;
    std::string extension() const override { return "xlsx"; }
};

class ParquetLoader : public FileLoader {
public:
    std::shared_ptr<DataFrame> load(const std::string& filename) const override;
    std::string extension() const override { return "parquet"; }
};

class RDBMSLoader : public FileLoader {
public:
    std::shared_ptr<DataFrame> load(const std::string& filename) const override;
    std::string extension() const override { return ""; }
};
    

} // namespace pluss::table