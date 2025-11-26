#include "../include/utils.h"

using torch::indexing::Slice;
using torch::indexing::None;



torch::Tensor sliding_windows(torch::Tensor ts, int64_t window_size) {
    std::vector<torch::Tensor> slices;
    for (int i = 0; i <= ts.size(0) - window_size; ++i) {
        slices.push_back(ts.slice(0, i, i + window_size));
    }
    return torch::stack(slices);
}
    
// Function to save the loss values to a CSV file
void to_csv(const std::vector<float>& loss_vec, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open the file!" << std::endl;
        return;
    }

    for (const auto& loss : loss_vec) {
        file << loss << "\n";
    }

    file.close();
}

torch::Tensor generate_random_values(double lower_bound, double upper_bound, size_t rows) {
    return lower_bound + (upper_bound - lower_bound) * torch::rand({static_cast<int64_t>(rows)}, torch::kFloat32);
}

void tensor_info(const torch::Tensor& tensor, const std::vector<std::string>& options = {"all"}) {
    // Map of info extractors
    std::unordered_map<std::string, std::function<std::any()>> info_extractors = {
        {"shape", [&]() { return tensor.sizes(); }},
        {"dtype", [&]() { return tensor.dtype(); }},
        {"device", [&]() { return tensor.device(); }},
        {"requires_grad", [&]() { return std::any(tensor.requires_grad()); }},
        {"is_leaf", [&]() { return tensor.is_leaf(); }},
        {"grad_fn", [&]() { return (tensor.grad_fn() ? tensor.grad_fn()->name() : "None");}}
    };

    // Dispatch table for type-specific printing
    auto print_any = [](const std::string& key, const std::any& value) {
        static const std::unordered_map<std::type_index, std::function<void(const std::any&)>> dispatch_table = {
            {typeid(c10::IntArrayRef), [](const std::any& val) {
                std::cout << std::any_cast<c10::IntArrayRef>(val) << std::endl;
            }},
            {typeid(torch::Dtype), [](const std::any& val) {
                std::cout << std::any_cast<torch::Dtype>(val) << std::endl;
            }},
            {typeid(torch::Device), [](const std::any& val) {
                std::cout << std::any_cast<torch::Device>(val) << std::endl;
            }},
            {typeid(bool), [](const std::any& val) {
                std::cout << (std::any_cast<bool>(val) ? "Yes" : "No") << std::endl;
            }},
            {typeid(std::string), [](const std::any& val) {
                std::cout << std::any_cast<std::string>(val) << std::endl;
            }}
        };

        std::cout << key << ": ";
        auto it = dispatch_table.find(value.type());
        if (it != dispatch_table.end()) {
            it->second(value);  // Call the associated function
        } else {
            std::cout << "Unknown" << std::endl;
        }
    };

    // Print selected info using std::for_each and ternary operator
    std::for_each(options.begin(), options.end(), [&](const std::string& key) {
    if (key == "all") {
        std::for_each(info_extractors.begin(), info_extractors.end(), [&](const auto& pair) {
            print_any(pair.first, pair.second());
        });
    } else if (info_extractors.count(key)) {
        print_any(key, info_extractors[key]());
    } else {
        // Key is invalid; do nothing or log
        std::cout << "Unknown or invalid option" << std::endl;
    }
    });
}


// Function to save checkpoint
void save_checkpoint(const std::string& path, torch::nn::Module& model, torch::optim::Optimizer& optimizer, int64_t epoch, float loss) {
    // Create an OutputArchive to save the model parameters
    torch::serialize::OutputArchive archive;
    model.save(archive);  // Save the model's state to the archive
    archive.save_to(path + "_model.pt");  // Save the archive to a file

    // Save the optimizer's state
    torch::serialize::OutputArchive optimizer_archive;
    optimizer.save(optimizer_archive);  // Save the optimizer's state to the archive
    optimizer_archive.save_to(path + "_optimizer.pt");  // Save the archive to a file

    // Save the epoch and loss in a text file
    std::ofstream epoch_file(path + "_epoch.txt");
    if (epoch_file.is_open()) {
        epoch_file << epoch << std::endl;
        epoch_file << loss << std::endl;
        epoch_file.close();
    }
}

void save_checkpoint(const std::string& path, torch::nn::Module& model) {
    // Create an OutputArchive to save the model parameters
    torch::serialize::OutputArchive archive;
    model.save(archive);  // Save the model's state to the archive
    archive.save_to(path);  // Save the archive to a file
}

// Function to split features (x) and labels (y) into train and test sets using test ratio
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
train_test_split(const torch::Tensor& x, const torch::Tensor& y, double test_ratio, int64_t dim) {
    // Validate the test ratio
    if (test_ratio < 0.0 || test_ratio > 1.0) {
        throw std::invalid_argument("test_ratio must be between 0.0 and 1.0");
    }

    // Ensure x and y have the same size along the specified dimension
    if (x.size(dim) != y.size(dim)) {
        throw std::invalid_argument("x and y must have the same size along the specified dimension");
    }

    // Get the total size along the specified dimension
    int64_t total_size = x.size(dim);

    // Compute the number of elements in the test set
    int64_t test_size = static_cast<int64_t>(total_size * test_ratio);

    // Compute the number of elements in the train set
    int64_t train_size = total_size - test_size;

    // Split features (x) and labels (y) into train and test sets
    torch::Tensor x_train = x.slice(dim, 0, train_size);
    torch::Tensor x_test = x.slice(dim, train_size, total_size);
    torch::Tensor y_train = y.slice(dim, 0, train_size);
    torch::Tensor y_test = y.slice(dim, train_size, total_size);

    // Return as a tuple
    return std::make_tuple(x_train, y_train, x_test, y_test);
}

std::unordered_map<std::string, torch::Tensor> load_trained_model(const std::string& path) {
    std::unordered_map<std::string, torch::Tensor> parameters;

    // Create an archive and load the state dictionary
    torch::serialize::InputArchive archive; 
    archive.load_from(path);
    torch::Tensor param{};

    // Retrieve each parameter by its name and store it in the map
    for (const auto& param_name : archive.keys()) {
        archive.read(param_name, param);
        parameters.emplace(param_name, param.cpu());
    }
    return parameters;
}

// Function to load checkpoint
bool load_checkpoint(const std::string& path, torch::nn::Module& model, torch::optim::Optimizer& optimizer, int64_t& epoch, float& loss) {
    try {
        // Create an InputArchive to load the model parameters
        torch::serialize::InputArchive archive;
        archive.load_from(path + "_model.pt");
        model.load(archive);  // Load the model's state from the archive

        // Load the optimizer's state
        torch::serialize::InputArchive optimizer_archive;
        optimizer_archive.load_from(path + "_optimizer.pt");
        optimizer.load(optimizer_archive);  // Load the optimizer's state from the archive

        // Load the epoch and loss from the text file
        std::ifstream epoch_file(path + "_epoch.txt");
        if (epoch_file.is_open()) {
            epoch_file >> epoch;
            epoch_file >> loss;
            epoch_file.close();
            return true;
        } else {
            return false;
        }
    } catch (...) {
        return false;
    }
}

// Function to load checkpoint
void load_checkpoint(const std::string& path, torch::nn::Module& model) {
    try {
	// Create an InputArchive to load the model parameters
        torch::serialize::InputArchive archive;
        archive.load_from(path);
        model.load(archive);  // Load the model's state from the archive
    } catch (...) {
        std::cout << "could not load model\n";
    }
}

// Function to map names to corresponding tensor values
std::unordered_map<std::string, torch::Tensor> name_tensor_map(
    const torch::Tensor& tensor, 
    const std::vector<std::string>& names) {
    
    // Ensure the size of names matches the tensor size
    if (tensor.size(0) != names.size()) {
        throw std::invalid_argument("The number of names must match the size of the tensor.");
    }

    // Create the unordered map
    std::unordered_map<std::string, torch::Tensor> named_tensor;

    // Populate the map
    for (size_t i = 0; i < names.size(); ++i) {
        named_tensor[names[i]] = tensor[i];
    }

    return named_tensor;
}
