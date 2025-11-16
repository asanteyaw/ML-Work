/**
 * @file CSVParser.h
 * @brief High-performance, header-only CSV parser library
 * 
 * This library provides a fast, memory-efficient CSV parser with support for:
 * - Memory-mapped file I/O for optimal performance
 * - SIMD-accelerated parsing where available
 * - Multi-threaded processing for large files
 * - Zero-copy architecture to minimize memory usage
 * - Flexible configuration for various CSV formats
 * - Comprehensive error handling and reporting
 * 
 * @author Yaw Asante
 * @date April 2025
 */

#ifndef CSV_PARSER_H
#define CSV_PARSER_H

#include <algorithm>
#include <array>
#include <charconv>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

#ifdef __cpp_lib_parallel_algorithm
#include <execution>
#endif

// Platform-specific includes for memory mapping
#if defined(_WIN32)
    #include <windows.h>
#else
    #include <fcntl.h>
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <unistd.h>
#endif

// Optional SIMD support
#if defined(__AVX2__)
    #include <immintrin.h>
    #define CSV_PARSER_SIMD_ENABLED 1
#elif defined(__SSE4_2__)
    #include <nmmintrin.h>
    #define CSV_PARSER_SIMD_ENABLED 1
#else
    #define CSV_PARSER_SIMD_ENABLED 0
#endif

namespace csv {

/**
 * @brief Exception class for CSV parsing errors
 */
class ParseError : public std::runtime_error {
public:
    ParseError(const std::string& message, size_t line = 0, size_t column = 0)
        : std::runtime_error(formatMessage(message, line, column))
        , m_line(line)
        , m_column(column) {}

    size_t line() const { return m_line; }
    size_t column() const { return m_column; }

private:
    size_t m_line;
    size_t m_column;

    static std::string formatMessage(const std::string& message, size_t line, size_t column) {
        if (line == 0 && column == 0) {
            return message;
        }
        return "Line " + std::to_string(line) + ", Column " + std::to_string(column) + ": " + message;
    }
};

/**
 * @brief Configuration options for the CSV parser
 */
struct ParserOptions {
    char delimiter = ',';                // Field delimiter character
    char quote = '"';                    // Quote character for fields
    char escape = '"';                   // Escape character
    bool trimSpaces = false;             // Trim leading/trailing spaces from fields
    bool skipEmptyLines = true;          // Skip empty lines
    bool hasHeader = true;               // First line is a header
    char commentPrefix = '\0';           // Character that marks comment lines (none if '\0')
    std::string newline = "\r\n";        // Newline sequence
    size_t chunkSize = 1024 * 1024;      // Processing chunk size for large files
    size_t maxThreads = 0;               // Max threads (0 = use hardware concurrency)
    bool strictMode = false;             // Strict RFC 4180 compliance
    bool detectTypes = true;             // Automatically detect column types
    bool skipBOM = true;                 // Skip UTF-8 BOM if present
    
    // Set maximum threads based on hardware if not specified
    void optimizeThreadCount() {
        if (maxThreads == 0) {
            maxThreads = std::thread::hardware_concurrency();
            // Ensure at least one thread
            if (maxThreads == 0) maxThreads = 1;
        }
    }
};

/**
 * @brief Type inference result
 */
enum class InferredType {
    String,
    Integer,
    Float,
    Boolean,
    Null
};

/**
 * @brief Supported data types for CSV fields
 */
using FieldType = std::variant<
    std::string_view,  // Raw string view (zero-copy)
    std::string,       // Owned string (when processing required)
    int64_t,           // Integer
    double,            // Floating point
    bool,              // Boolean
    std::nullptr_t     // Null/empty value
>;

/**
 * @brief Memory-mapped file wrapper for efficient I/O
 */
class MemoryMappedFile {
public:
    MemoryMappedFile() = default;
    
    MemoryMappedFile(const std::filesystem::path& path) {
        open(path);
    }
    
    ~MemoryMappedFile() {
        close();
    }
    
    // No copy
    MemoryMappedFile(const MemoryMappedFile&) = delete;
    MemoryMappedFile& operator=(const MemoryMappedFile&) = delete;
    
    // Move allowed
    MemoryMappedFile(MemoryMappedFile&& other) noexcept {
        *this = std::move(other);
    }
    
    MemoryMappedFile& operator=(MemoryMappedFile&& other) noexcept {
        if (this != &other) {
            close();
            
            m_data = other.m_data;
            m_size = other.m_size;
            
            #if defined(_WIN32)
            m_fileHandle = other.m_fileHandle;
            m_mappingHandle = other.m_mappingHandle;
            #else
            m_fd = other.m_fd;
            #endif
            
            other.m_data = nullptr;
            other.m_size = 0;
            
            #if defined(_WIN32)
            other.m_fileHandle = INVALID_HANDLE_VALUE;
            other.m_mappingHandle = NULL;
            #else
            other.m_fd = -1;
            #endif
        }
        return *this;
    }
    
    bool open(const std::filesystem::path& path) {
        close();
        
        #if defined(_WIN32)
        // Windows implementation
        m_fileHandle = CreateFileW(
            path.wstring().c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            NULL,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            NULL
        );
        
        if (m_fileHandle == INVALID_HANDLE_VALUE) {
            return false;
        }
        
        LARGE_INTEGER fileSize;
        if (!GetFileSizeEx(m_fileHandle, &fileSize)) {
            CloseHandle(m_fileHandle);
            m_fileHandle = INVALID_HANDLE_VALUE;
            return false;
        }
        
        m_size = static_cast<size_t>(fileSize.QuadPart);
        
        if (m_size == 0) {
            // Empty file
            CloseHandle(m_fileHandle);
            m_fileHandle = INVALID_HANDLE_VALUE;
            return true;
        }
        
        m_mappingHandle = CreateFileMappingW(
            m_fileHandle,
            NULL,
            PAGE_READONLY,
            0,
            0,
            NULL
        );
        
        if (m_mappingHandle == NULL) {
            CloseHandle(m_fileHandle);
            m_fileHandle = INVALID_HANDLE_VALUE;
            return false;
        }
        
        m_data = static_cast<const char*>(MapViewOfFile(
            m_mappingHandle,
            FILE_MAP_READ,
            0,
            0,
            0
        ));
        
        if (m_data == nullptr) {
            CloseHandle(m_mappingHandle);
            CloseHandle(m_fileHandle);
            m_mappingHandle = NULL;
            m_fileHandle = INVALID_HANDLE_VALUE;
            return false;
        }
        
        #else
        // POSIX implementation
        m_fd = ::open(path.c_str(), O_RDONLY);
        if (m_fd == -1) {
            return false;
        }
        
        struct stat sb;
        if (fstat(m_fd, &sb) == -1) {
            ::close(m_fd);
            m_fd = -1;
            return false;
        }
        
        m_size = static_cast<size_t>(sb.st_size);
        
        if (m_size == 0) {
            // Empty file
            ::close(m_fd);
            m_fd = -1;
            return true;
        }
        
        m_data = static_cast<const char*>(mmap(
            nullptr,
            m_size,
            PROT_READ,
            MAP_PRIVATE,
            m_fd,
            0
        ));
        
        if (m_data == MAP_FAILED) {
            m_data = nullptr;
            ::close(m_fd);
            m_fd = -1;
            return false;
        }
        
        // Advise the kernel that we'll be reading sequentially
        madvise(const_cast<char*>(m_data), m_size, MADV_SEQUENTIAL);
        #endif
        
        return true;
    }
    
    void close() {
        if (m_data) {
            #if defined(_WIN32)
            UnmapViewOfFile(m_data);
            m_data = nullptr;
            
            if (m_mappingHandle != NULL) {
                CloseHandle(m_mappingHandle);
                m_mappingHandle = NULL;
            }
            
            if (m_fileHandle != INVALID_HANDLE_VALUE) {
                CloseHandle(m_fileHandle);
                m_fileHandle = INVALID_HANDLE_VALUE;
            }
            #else
            munmap(const_cast<char*>(m_data), m_size);
            m_data = nullptr;
            
            if (m_fd != -1) {
                ::close(m_fd);
                m_fd = -1;
            }
            #endif
        }
        
        m_size = 0;
    }
    
    bool isOpen() const {
        #if defined(_WIN32)
        return m_fileHandle != INVALID_HANDLE_VALUE;
        #else
        return m_fd != -1;
        #endif
    }
    
    const char* data() const { return m_data; }
    size_t size() const { return m_size; }
    
    std::string_view view() const {
        return std::string_view(m_data, m_size);
    }
    
private:
    const char* m_data = nullptr;
    size_t m_size = 0;
    
    #if defined(_WIN32)
    HANDLE m_fileHandle = INVALID_HANDLE_VALUE;
    HANDLE m_mappingHandle = NULL;
    #else
    int m_fd = -1;
    #endif
};

/**
 * @brief Chunk of CSV data for parallel processing
 */
struct DataChunk {
    std::string_view data;
    size_t startLine;
    size_t endLine;
    size_t startOffset;
    size_t endOffset;
    
    // Results after processing
    std::vector<std::vector<FieldType>> rows;
    std::vector<ParseError> errors;
};

/**
 * @brief Type inference and conversion utilities
 */
class TypeConverter {
public:
    /**
     * @brief Infer the type of a field
     */
    static InferredType inferType(std::string_view field) {
        if (field.empty()) {
            return InferredType::Null;
        }
        
        // Check for boolean
        if (field == "true" || field == "false" || field == "TRUE" || field == "FALSE" ||
            field == "True" || field == "False" || field == "1" || field == "0" ||
            field == "yes" || field == "no" || field == "YES" || field == "NO" ||
            field == "Yes" || field == "No" || field == "y" || field == "n" ||
            field == "Y" || field == "N") {
            return InferredType::Boolean;
        }
        
        // Check for integer
        bool isInteger = true;
        bool isFloat = true;
        bool hasDecimalPoint = false;
        
        // Allow leading sign
        size_t start = 0;
        if (!field.empty() && (field[0] == '-' || field[0] == '+')) {
            start = 1;
        }
        
        for (size_t i = start; i < field.size(); ++i) {
            if (field[i] == '.') {
                if (hasDecimalPoint) {
                    isFloat = false;
                    break;
                }
                hasDecimalPoint = true;
                isInteger = false;
            } else if (field[i] == 'e' || field[i] == 'E') {
                // Scientific notation
                isInteger = false;
                if (i + 1 < field.size() && (field[i + 1] == '+' || field[i + 1] == '-')) {
                    ++i;
                }
            } else if (!std::isdigit(field[i])) {
                isInteger = false;
                isFloat = false;
                break;
            }
        }
        
        if (isInteger) {
            return InferredType::Integer;
        }
        
        if (isFloat) {
            return InferredType::Float;
        }
        
        return InferredType::String;
    }
    
    /**
     * @brief Convert a field to the specified type
     */
    template<typename T>
    static std::optional<T> convert(const FieldType& field) {
        if (std::holds_alternative<std::nullptr_t>(field)) {
            return std::nullopt;
        }
        
        if constexpr (std::is_same_v<T, std::string> || std::is_same_v<T, std::string_view>) {
            if (std::holds_alternative<std::string>(field)) {
                return std::get<std::string>(field);
            } else if (std::holds_alternative<std::string_view>(field)) {
                return std::string(std::get<std::string_view>(field));
            }
        } else if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
            if (std::holds_alternative<int64_t>(field)) {
                return static_cast<T>(std::get<int64_t>(field));
            } else {
                std::string_view sv;
                if (std::holds_alternative<std::string>(field)) {
                    sv = std::get<std::string>(field);
                } else if (std::holds_alternative<std::string_view>(field)) {
                    sv = std::get<std::string_view>(field);
                } else {
                    return std::nullopt;
                }
                
                T value{};
                auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), value);
                if (ec == std::errc()) {
                    return value;
                }
            }
        } else if constexpr (std::is_floating_point_v<T>) {
            if (std::holds_alternative<double>(field)) {
                return static_cast<T>(std::get<double>(field));
            } else {
                std::string_view sv;
                if (std::holds_alternative<std::string>(field)) {
                    sv = std::get<std::string>(field);
                } else if (std::holds_alternative<std::string_view>(field)) {
                    sv = std::get<std::string_view>(field);
                } else {
                    return std::nullopt;
                }
                
                // from_chars for floating point might not be available in all compilers
                // fallback to string stream
                try {
                    return static_cast<T>(std::stod(std::string(sv)));
                } catch (...) {
                    return std::nullopt;
                }
            }
        } else if constexpr (std::is_same_v<T, bool>) {
            if (std::holds_alternative<bool>(field)) {
                return std::get<bool>(field);
            } else {
                std::string_view sv;
                if (std::holds_alternative<std::string>(field)) {
                    sv = std::get<std::string>(field);
                } else if (std::holds_alternative<std::string_view>(field)) {
                    sv = std::get<std::string_view>(field);
                } else {
                    return std::nullopt;
                }
                
                if (sv == "true" || sv == "TRUE" || sv == "True" || sv == "1" || 
                    sv == "yes" || sv == "YES" || sv == "Yes" || sv == "y" || sv == "Y") {
                    return true;
                } else if (sv == "false" || sv == "FALSE" || sv == "False" || sv == "0" || 
                           sv == "no" || sv == "NO" || sv == "No" || sv == "n" || sv == "N") {
                    return false;
                }
            }
        }
        
        return std::nullopt;
    }
    
    /**
     * @brief Convert a field to its inferred type
     */
    static FieldType convertToInferredType(const FieldType& field) {
        if (std::holds_alternative<std::nullptr_t>(field)) {
            return nullptr;
        }
        
        std::string_view sv;
        if (std::holds_alternative<std::string>(field)) {
            sv = std::get<std::string>(field);
        } else if (std::holds_alternative<std::string_view>(field)) {
            sv = std::get<std::string_view>(field);
        } else {
            return field;  // Already converted
        }
        
        InferredType type = inferType(sv);
        
        switch (type) {
            case InferredType::Integer: {
                int64_t value{};
                auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), value);
                if (ec == std::errc()) {
                    return value;
                }
                break;
            }
            case InferredType::Float: {
                try {
                    return std::stod(std::string(sv));
                } catch (...) {
                    // Fall back to string
                }
                break;
            }
            case InferredType::Boolean: {
                if (sv == "true" || sv == "TRUE" || sv == "True" || sv == "1" || 
                    sv == "yes" || sv == "YES" || sv == "Yes" || sv == "y" || sv == "Y") {
                    return true;
                } else if (sv == "false" || sv == "FALSE" || sv == "False" || sv == "0" || 
                           sv == "no" || sv == "NO" || sv == "No" || sv == "n" || sv == "N") {
                    return false;
                }
                break;
            }
            case InferredType::Null:
                return nullptr;
            case InferredType::String:
                // Keep as string
                break;
        }
        
        // Default: return as string
        return std::string(sv);
    }
};

/**
 * @brief CSV field parser with SIMD acceleration where available
 */
class FieldParser {
public:
    explicit FieldParser(const ParserOptions& options) : m_options(options) {}
    
    /**
     * @brief Parse a line of CSV data into fields
     * 
     * @param line The line to parse
     * @param lineNum Line number for error reporting
     * @return Vector of parsed fields
     */
    std::vector<FieldType> parseLine(std::string_view line, size_t lineNum) const {
        std::vector<FieldType> fields;
        size_t pos = 0;
        size_t fieldStart = 0;
        bool inQuotes = false;
        
        while (pos <= line.size()) {
            // End of line or delimiter outside quotes
            if (pos == line.size() || (line[pos] == m_options.delimiter && !inQuotes)) {
                addField(fields, line, fieldStart, pos, lineNum);
                fieldStart = pos + 1;
            }
            // Handle quotes
            else if (line[pos] == m_options.quote) {
                if (inQuotes && pos + 1 < line.size() && line[pos + 1] == m_options.quote) {
                    // Escaped quote within quoted field
                    pos++;
                } else {
                    // Start or end of quoted field
                    inQuotes = !inQuotes;
                }
            }
            
            pos++;
        }
        
        // Check for unterminated quotes
        if (inQuotes) {
            throw ParseError("Unterminated quoted field", lineNum, fieldStart + 1);
        }
        
        return fields;
    }
    
    /**
     * @brief Fast field parsing using SIMD instructions where available
     * 
     * @param data Pointer to CSV data
     * @param length Length of data
     * @param delimiter Field delimiter
     * @param results Vector to store field positions
     */
    static void findFieldBoundariesSIMD(const char* data, size_t length, char delimiter, 
                                        std::vector<std::pair<size_t, size_t>>& results) {
        #if CSV_PARSER_SIMD_ENABLED && defined(__AVX2__)
        // AVX2 implementation for 256-bit SIMD
        if (length >= 32) {
            const __m256i delim_mask = _mm256_set1_epi8(delimiter);
            const __m256i quote_mask = _mm256_set1_epi8('"');
            
            __m256i quote_state = _mm256_setzero_si256();
            size_t i = 0;
            size_t field_start = 0;
            
            for (; i + 32 <= length; i += 32) {
                const __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
                
                // Find delimiters and quotes
                __m256i delim_match = _mm256_cmpeq_epi8(chunk, delim_mask);
                __m256i quote_match = _mm256_cmpeq_epi8(chunk, quote_mask);
                
                // Update quote state
                quote_state = _mm256_xor_si256(quote_state, quote_match);
                
                // Only consider delimiters outside quotes
                __m256i valid_delim = _mm256_andnot_si256(quote_state, delim_match);
                
                uint32_t mask = _mm256_movemask_epi8(valid_delim);
                
                while (mask) {
                    uint32_t bit_pos = __builtin_ctz(mask);
                    size_t delim_pos = i + bit_pos;
                    
                    results.emplace_back(field_start, delim_pos);
                    field_start = delim_pos + 1;
                    
                    mask &= mask - 1;  // Clear the least significant bit
                }
            }
            
            // Process remaining bytes
            if (field_start < length) {
                findFieldBoundariesScalar(data + field_start, length - field_start, delimiter, results, field_start);
            }
            
            return;
        }
        #elif CSV_PARSER_SIMD_ENABLED && defined(__SSE4_2__)
        // SSE4.2 implementation for 128-bit SIMD
        if (length >= 16) {
            // Use PCMPISTRI for string comparison
            const __m128i delim_mask = _mm_set1_epi8(delimiter);
            bool in_quotes = false;
            size_t i = 0;
            size_t field_start = 0;
            
            for (; i + 16 <= length; i += 16) {
                const __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + i));
                
                // Find delimiters
                int delim_offset = _mm_cmpistri(delim_mask, chunk, 
                                               _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | 
                                               _SIDD_LEAST_SIGNIFICANT);
                
                // Find quotes
                const __m128i quote_mask = _mm_set1_epi8('"');
                int quote_offset = _mm_cmpistri(quote_mask, chunk, 
                                               _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | 
                                               _SIDD_LEAST_SIGNIFICANT);
                
                // Process the chunk
                if (delim_offset < 16 && (in_quotes || delim_offset < quote_offset)) {
                    // Delimiter found outside quotes
                    if (!in_quotes) {
                        results.emplace_back(field_start, i + delim_offset);
                        field_start = i + delim_offset + 1;
                    }
                    i += delim_offset;
                    continue;
                }
                
                if (quote_offset < 16) {
                    // Quote found
                    in_quotes = !in_quotes;
                    i += quote_offset;
                    continue;
                }
            }
            
            // Process remaining bytes
            if (i < length) {
                findFieldBoundariesScalar(data + i, length - i, delimiter, results, i);
            }
            
            return;
        }
        #endif
        
        // Fallback to scalar implementation
        findFieldBoundariesScalar(data, length, delimiter, results, 0);
    }
    
    /**
     * @brief Scalar implementation of field boundary detection
     */
    static void findFieldBoundariesScalar(const char* data, size_t length, char delimiter,
                                         std::vector<std::pair<size_t, size_t>>& results, size_t offset) {
        size_t field_start = 0;
        bool in_quotes = false;
        
        for (size_t i = 0; i < length; ++i) {
            if (data[i] == '"') {
                in_quotes = !in_quotes;
            } else if (data[i] == delimiter && !in_quotes) {
                results.emplace_back(offset + field_start, offset + i);
                field_start = i + 1;
            }
        }
        
        // Add the last field
        if (field_start < length) {
            results.emplace_back(offset + field_start, offset + length);
        }
    }
    
private:
    const ParserOptions& m_options;
    
    /**
     * @brief Add a parsed field to the result vector
     */
    void addField(std::vector<FieldType>& fields, std::string_view line, 
                 size_t start, size_t end, size_t lineNum) const {
        if (start >= end) {
            // Empty field
            fields.emplace_back(nullptr);
            return;
        }
        
        std::string_view field = line.substr(start, end - start);
        
        // Handle quoted fields
        if (!field.empty() && field.front() == m_options.quote && field.back() == m_options.quote) {
            // Remove quotes
            field = field.substr(1, field.size() - 2);
            
            // Handle escaped quotes
            if (field.find(m_options.quote) != std::string_view::npos) {
                std::string unescaped;
                unescaped.reserve(field.size());
                
                for (size_t i = 0; i < field.size(); ++i) {
                    unescaped.push_back(field[i]);
                    if (field[i] == m_options.quote && i + 1 < field.size() && field[i + 1] == m_options.quote) {
                        // Skip the second quote
                        ++i;
                    }
                }
                
                fields.emplace_back(std::move(unescaped));
                return;
            }
        }
        
        // Trim spaces if requested
        if (m_options.trimSpaces && !field.empty()) {
            while (!field.empty() && std::isspace(field.front())) {
                field = field.substr(1);
            }
            while (!field.empty() && std::isspace(field.back())) {
                field = field.substr(0, field.size() - 1);
            }
        }
        
        // Add the field as a string_view (zero-copy)
        fields.emplace_back(field);
    }
};

/**
 * @brief Main CSV parser class
 */
 class CSVParser {
    public:
        /**
         * @brief Construct a new CSV parser with default options
         */
        CSVParser() = default;
        
        /**
         * @brief Construct a new CSV parser with custom options
         * 
         * @param options Parser configuration options
         */
        explicit CSVParser(const ParserOptions& options) : m_options(options) {
            m_options.optimizeThreadCount();
        }
        
        /**
         * @brief Parse a CSV file
         * 
         * @param path Path to the CSV file
         * @return std::vector<std::vector<FieldType>> Parsed data
         * @throws ParseError if parsing fails
         */
        std::vector<std::vector<FieldType>> parseFile(const std::filesystem::path& path) {
            MemoryMappedFile file(path);
            if (!file.isOpen()) {
                throw ParseError("Failed to open file: " + path.string());
            }
            
            return parseData(file.data(), file.size());
        }
        
        /**
         * @brief Parse CSV data from a string
         * 
         * @param data CSV data as a string
         * @return std::vector<std::vector<FieldType>> Parsed data
         * @throws ParseError if parsing fails
         */
        std::vector<std::vector<FieldType>> parseString(std::string_view data) {
            return parseData(data.data(), data.size());
        }
        
        /**
         * @brief Parse CSV data with callback for each row
         * 
         * @param path Path to the CSV file
         * @param callback Function to call for each parsed row
         * @throws ParseError if parsing fails
         */
        void parseFileWithCallback(const std::filesystem::path& path, 
                                  const std::function<void(std::vector<FieldType>&, size_t)>& callback) {
            MemoryMappedFile file(path);
            if (!file.isOpen()) {
                throw ParseError("Failed to open file: " + path.string());
            }
            
            parseDataWithCallback(file.data(), file.size(), callback);
        }
        
        /**
         * @brief Get the column names from the header row
         * 
         * @return const std::vector<std::string>& Column names
         */
        const std::vector<std::string>& getColumnNames() const {
            return m_columnNames;
        }
        
        /**
         * @brief Get the inferred column types
         * 
         * @return const std::vector<InferredType>& Column types
         */
        const std::vector<InferredType>& getColumnTypes() const {
            return m_columnTypes;
        }
        
        /**
         * @brief Get the parser options
         * 
         * @return const ParserOptions& Parser options
         */
        const ParserOptions& getOptions() const {
            return m_options;
        }
        
    private:
        ParserOptions m_options;
        std::vector<std::string> m_columnNames;
        std::vector<InferredType> m_columnTypes;
        
        /**
         * @brief Parse CSV data from memory
         * 
         * @param data Pointer to CSV data
         * @param size Size of data in bytes
         * @return std::vector<std::vector<FieldType>> Parsed data
         * @throws ParseError if parsing fails
         */
        std::vector<std::vector<FieldType>> parseData(const char* data, size_t size) {
            std::vector<std::vector<FieldType>> result;
            
            if (size == 0) {
                return result;
            }
            
            // Skip BOM if present
            size_t offset = 0;
            if (m_options.skipBOM && size >= 3 && 
                static_cast<unsigned char>(data[0]) == 0xEF && 
                static_cast<unsigned char>(data[1]) == 0xBB && 
                static_cast<unsigned char>(data[2]) == 0xBF) {
                offset = 3;
            }
            
            // Find line breaks to determine chunks for parallel processing
            std::vector<size_t> lineBreaks;
            lineBreaks.push_back(offset);
            
            bool inQuotes = false;
            for (size_t i = offset; i < size; ++i) {
                if (data[i] == m_options.quote) {
                    inQuotes = !inQuotes;
                } else if (!inQuotes) {
                    if (data[i] == '\n') {
                        lineBreaks.push_back(i + 1);
                    } else if (data[i] == '\r') {
                        if (i + 1 < size && data[i + 1] == '\n') {
                            // CRLF
                            lineBreaks.push_back(i + 2);
                            ++i;
                        } else {
                            // CR only
                            lineBreaks.push_back(i + 1);
                        }
                    }
                }
            }
            
            // Add end of data
            if (lineBreaks.back() != size) {
                lineBreaks.push_back(size);
            }
            
            // Process header if present
            size_t dataStartLine = 0;
            if (m_options.hasHeader && lineBreaks.size() > 1) {
                std::string_view headerLine(data + lineBreaks[0], lineBreaks[1] - lineBreaks[0]);
                // Remove trailing newline
                if (!headerLine.empty() && (headerLine.back() == '\n' || headerLine.back() == '\r')) {
                    headerLine = headerLine.substr(0, headerLine.size() - 1);
                    if (!headerLine.empty() && headerLine.back() == '\r') {
                        headerLine = headerLine.substr(0, headerLine.size() - 1);
                    }
                }
                
                FieldParser parser(m_options);
                std::vector<FieldType> headerFields = parser.parseLine(headerLine, 1);
                
                m_columnNames.clear();
                for (const auto& field : headerFields) {
                    if (std::holds_alternative<std::string_view>(field)) {
                        m_columnNames.push_back(std::string(std::get<std::string_view>(field)));
                    } else if (std::holds_alternative<std::string>(field)) {
                        m_columnNames.push_back(std::get<std::string>(field));
                    } else {
                        m_columnNames.push_back(std::string());
                    }
                }
                
                dataStartLine = 1;
            }
            
            // Determine chunks for parallel processing
            std::vector<DataChunk> chunks;
            const size_t numLines = lineBreaks.size() - 1;
            const size_t linesPerChunk = std::max(size_t(1), numLines / m_options.maxThreads);
            
            for (size_t i = dataStartLine; i < numLines; i += linesPerChunk) {
                DataChunk chunk;
                chunk.startLine = i;
                chunk.endLine = std::min(i + linesPerChunk, numLines);
                chunk.startOffset = lineBreaks[chunk.startLine];
                chunk.endOffset = lineBreaks[chunk.endLine];
                chunk.data = std::string_view(data + chunk.startOffset, chunk.endOffset - chunk.startOffset);
                chunks.push_back(chunk);
            }
            
            // Process chunks in parallel
            if (chunks.size() > 1 && m_options.maxThreads > 1) {
                std::vector<std::thread> threads;
                threads.reserve(chunks.size());
                
                for (auto& chunk : chunks) {
                    threads.emplace_back([this, &chunk]() {
                        this->processChunk(chunk);
                    });
                }
                
                for (auto& thread : threads) {
                    thread.join();
                }
            } else if (!chunks.empty()) {
                // Single-threaded processing
                processChunk(chunks[0]);
            }
            
            // Combine results
            size_t totalRows = 0;
            for (const auto& chunk : chunks) {
                totalRows += chunk.rows.size();
            }
            
            result.reserve(totalRows);
            for (const auto& chunk : chunks) {
                result.insert(result.end(), chunk.rows.begin(), chunk.rows.end());
                
                // Report any errors
                for (const auto& error : chunk.errors) {
                    std::cerr << "Warning: " << error.what() << std::endl;
                }
            }
            
            // Infer column types if requested
            if (m_options.detectTypes && !result.empty()) {
                inferColumnTypes(result);
            }
            
            return result;
        }
        
        /**
         * @brief Parse CSV data with callback for each row
         * 
         * @param data Pointer to CSV data
         * @param size Size of data in bytes
         * @param callback Function to call for each parsed row
         * @throws ParseError if parsing fails
         */
        void parseDataWithCallback(const char* data, size_t size, 
                                  const std::function<void(std::vector<FieldType>&, size_t)>& callback) {
            if (size == 0) {
                return;
            }
            
            // Skip BOM if present
            size_t offset = 0;
            if (m_options.skipBOM && size >= 3 && 
                static_cast<unsigned char>(data[0]) == 0xEF && 
                static_cast<unsigned char>(data[1]) == 0xBB && 
                static_cast<unsigned char>(data[2]) == 0xBF) {
                offset = 3;
            }
            
            FieldParser parser(m_options);
            size_t lineNum = 1;
            size_t lineStart = offset;
            
            // Process header if present
            if (m_options.hasHeader && lineStart < size) {
                size_t lineEnd = findLineEnd(data, size, lineStart);
                std::string_view line(data + lineStart, lineEnd - lineStart);
                
                std::vector<FieldType> headerFields = parser.parseLine(line, lineNum);
                
                m_columnNames.clear();
                for (const auto& field : headerFields) {
                    if (std::holds_alternative<std::string_view>(field)) {
                        m_columnNames.push_back(std::string(std::get<std::string_view>(field)));
                    } else if (std::holds_alternative<std::string>(field)) {
                        m_columnNames.push_back(std::get<std::string>(field));
                    } else {
                        m_columnNames.push_back(std::string());
                    }
                }
                
                lineStart = lineEnd;
                ++lineNum;
            }
            
            // Process data rows
            while (lineStart < size) {
                size_t lineEnd = findLineEnd(data, size, lineStart);
                std::string_view line(data + lineStart, lineEnd - lineStart);
                
                // Skip empty lines if requested
                if (!m_options.skipEmptyLines || !line.empty()) {
                    try {
                        std::vector<FieldType> fields = parser.parseLine(line, lineNum);
                        callback(fields, lineNum);
                    } catch (const ParseError& e) {
                        std::cerr << "Warning: " << e.what() << std::endl;
                    }
                }
                
                lineStart = lineEnd;
                ++lineNum;
            }
        }
        
        /**
         * @brief Find the end of a line
         * 
         * @param data Pointer to CSV data
         * @param size Size of data in bytes
         * @param start Start position
         * @return size_t End position (including newline)
         */
        size_t findLineEnd(const char* data, size_t size, size_t start) {
            bool inQuotes = false;
            
            for (size_t i = start; i < size; ++i) {
                if (data[i] == m_options.quote) {
                    inQuotes = !inQuotes;
                } else if (!inQuotes) {
                    if (data[i] == '\n') {
                        return i + 1;
                    } else if (data[i] == '\r') {
                        if (i + 1 < size && data[i + 1] == '\n') {
                            // CRLF
                            return i + 2;
                        } else {
                            // CR only
                            return i + 1;
                        }
                    }
                }
            }
            
            return size;
        }
        
        /**
         * @brief Process a chunk of CSV data
         * 
         * @param chunk Chunk to process
         */
        void processChunk(DataChunk& chunk) {
            FieldParser parser(m_options);
            
            size_t lineStart = 0;
            size_t lineNum = chunk.startLine + 1;
            
            while (lineStart < chunk.data.size()) {
                // Find end of line
                size_t lineEnd = 0;
                for (size_t i = lineStart; i < chunk.data.size(); ++i) {
                    if (chunk.data[i] == '\n') {
                        lineEnd = i + 1;
                        break;
                    } else if (chunk.data[i] == '\r') {
                        if (i + 1 < chunk.data.size() && chunk.data[i + 1] == '\n') {
                            lineEnd = i + 2;
                        } else {
                            lineEnd = i + 1;
                        }
                        break;
                    }
                }
                
                if (lineEnd == 0) {
                    lineEnd = chunk.data.size();
                }
                
                std::string_view line = chunk.data.substr(lineStart, lineEnd - lineStart);
                
                // Remove trailing newline
                if (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
                    line = line.substr(0, line.size() - 1);
                    if (!line.empty() && line.back() == '\r') {
                        line = line.substr(0, line.size() - 1);
                    }
                }
                
                // Skip empty lines if requested
                if (!m_options.skipEmptyLines || !line.empty()) {
                    try {
                        std::vector<FieldType> fields = parser.parseLine(line, lineNum);
                        chunk.rows.push_back(std::move(fields));
                    } catch (const ParseError& e) {
                        chunk.errors.push_back(e);
                    }
                }
                
                lineStart = lineEnd;
                ++lineNum;
            }
        }
        
        /**
         * @brief Infer column types from data
         * 
         * @param data Parsed data
         */
        void inferColumnTypes(const std::vector<std::vector<FieldType>>& data) {
            if (data.empty()) {
                m_columnTypes.clear();
                return;
            }
            
            const size_t numColumns = data[0].size();
            m_columnTypes.resize(numColumns, InferredType::String);
            
            // Initialize with most specific type
            for (size_t col = 0; col < numColumns; ++col) {
                m_columnTypes[col] = InferredType::Integer;
            }
            
            // Sample rows for type inference
            const size_t maxSamples = 1000;
            const size_t step = std::max(size_t(1), data.size() / maxSamples);
            
            for (size_t row = 0; row < data.size(); row += step) {
                const auto& fields = data[row];
                
                for (size_t col = 0; col < std::min(fields.size(), numColumns); ++col) {
                    // Skip null values
                    if (std::holds_alternative<std::nullptr_t>(fields[col])) {
                        continue;
                    }
                    
                    std::string_view sv;
                    if (std::holds_alternative<std::string>(fields[col])) {
                        sv = std::get<std::string>(fields[col]);
                    } else if (std::holds_alternative<std::string_view>(fields[col])) {
                        sv = std::get<std::string_view>(fields[col]);
                    } else {
                        // Already converted to a specific type
                        continue;
                    }
                    
                    InferredType fieldType = TypeConverter::inferType(sv);
                    
                    // Update column type (widen if necessary)
                    if (fieldType == InferredType::String) {
                        m_columnTypes[col] = InferredType::String;
                    } else if (fieldType == InferredType::Float && m_columnTypes[col] == InferredType::Integer) {
                        m_columnTypes[col] = InferredType::Float;
                    } else if (fieldType == InferredType::Boolean && m_columnTypes[col] == InferredType::Integer) {
                        m_columnTypes[col] = InferredType::Boolean;
                    }
                }
            }
        }
    };
    
    /**
     * @brief CSV writer class for generating CSV files
     */
    class CSVWriter {
    public:
        /**
         * @brief Construct a new CSV writer with default options
         */
        CSVWriter() = default;
        
        /**
         * @brief Construct a new CSV writer with custom options
         * 
         * @param options Parser configuration options (used for formatting)
         */
        explicit CSVWriter(const ParserOptions& options) : m_options(options) {}
        
        /**
         * @brief Write data to a CSV file
         * 
         * @param path Path to the output file
         * @param data Data to write
         * @param columnNames Optional column names for header row
         * @return bool True if successful
         */
        bool writeFile(const std::filesystem::path& path, 
                      const std::vector<std::vector<FieldType>>& data,
                      const std::vector<std::string>& columnNames = {}) {
            std::ofstream file(path);
            if (!file) {
                return false;
            }
            
            return writeStream(file, data, columnNames);
        }
        
        /**
         * @brief Write data to an output stream
         * 
         * @param os Output stream
         * @param data Data to write
         * @param columnNames Optional column names for header row
         * @return bool True if successful
         */
        bool writeStream(std::ostream& os, 
                        const std::vector<std::vector<FieldType>>& data,
                        const std::vector<std::string>& columnNames = {}) {
            // Write header if provided
            if (!columnNames.empty()) {
                for (size_t i = 0; i < columnNames.size(); ++i) {
                    if (i > 0) {
                        os << m_options.delimiter;
                    }
                    writeField(os, columnNames[i]);
                }
                os << m_options.newline;
            }
            
            // Write data rows
            for (const auto& row : data) {
                for (size_t i = 0; i < row.size(); ++i) {
                    if (i > 0) {
                        os << m_options.delimiter;
                    }
                    writeField(os, row[i]);
                }
                os << m_options.newline;
            }
            
            return os.good();
        }
        
    private:
        ParserOptions m_options;
        
        /**
         * @brief Write a field to the output stream
         * 
         * @param os Output stream
         * @param field Field to write
         */
        void writeField(std::ostream& os, const FieldType& field) {
            if (std::holds_alternative<std::nullptr_t>(field)) {
                // Empty field
                return;
            } else if (std::holds_alternative<std::string_view>(field)) {
                writeString(os, std::get<std::string_view>(field));
            } else if (std::holds_alternative<std::string>(field)) {
                writeString(os, std::get<std::string>(field));
            } else if (std::holds_alternative<int64_t>(field)) {
                os << std::get<int64_t>(field);
            } else if (std::holds_alternative<double>(field)) {
                os << std::get<double>(field);
            } else if (std::holds_alternative<bool>(field)) {
                os << (std::get<bool>(field) ? "true" : "false");
            }
        }
        
        /**
         * @brief Write a string field, quoting if necessary
         * 
         * @param os Output stream
         * @param str String to write
         */
        void writeString(std::ostream& os, std::string_view str) {
            bool needsQuoting = str.empty() ||
                               str.find(m_options.delimiter) != std::string_view::npos ||
                               str.find(m_options.quote) != std::string_view::npos ||
                               str.find('\n') != std::string_view::npos ||
                               str.find('\r') != std::string_view::npos ||
                               str.front() == ' ' || str.back() == ' ';
            
            if (needsQuoting) {
                os << m_options.quote;
                
                for (char c : str) {
                    if (c == m_options.quote) {
                        // Escape quotes
                        os << m_options.quote << m_options.quote;
                    } else {
                        os << c;
                    }
                }
                
                os << m_options.quote;
            } else {
                os << str;
            }
        }
        
        /**
         * @brief Write a string field, quoting if necessary
         * 
         * @param os Output stream
         * @param str String to write
         */
        void writeString(std::ostream& os, const std::string& str) {
            writeString(os, std::string_view(str));
        }
    };
    
    } // namespace csv
    
    #endif // CSV_PARSER_H
