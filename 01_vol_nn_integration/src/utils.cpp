#include "utils.h"


torch::Tensor neg_log_likelihood(const torch::Tensor& z, const torch::Tensor& h) {
    return 0.5 * torch::sum(torch::log(2.0*M_PI*h)+torch::pow(z, 2));
}

torch::Tensor sample_mixture_normal(
    const torch::Tensor& Z1,          // [N, D]
    const torch::Tensor& Z2,          // [N, D]
    const torch::TensorList params  // {w, m1, h1, m2, h2}
) {
    TORCH_CHECK(Z1.sizes() == Z2.sizes(), "Z1 and Z2 must be the same shape");
    TORCH_CHECK(params.size() == 5, "Expected 5 parameters: {w, m1, h1, m2, h2}");

    const auto w  = params[0];
    const auto m1 = params[1];
    const auto h1 = params[2];
    const auto m2 = params[3];
    const auto h2 = params[4];

    // Reproducible generator
    torch::Generator gen = torch::make_generator<torch::CPUGeneratorImpl>();
    gen.set_current_seed(1);

    //auto mask = (torch::rand({Z1.size(0)}, gen).to(Z1.device()).unsqueeze(1).expand_as(Z1)).lt(w);
    auto mask = (torch::rand(Z1.sizes(), gen)).to(Z1.device()).lt(w);
    // Print the first 10 rows and first 10 columns of the mask
    std::cout << "About to print out the mask\n";
    std::cout << "Mask (first 10x10):\n" << mask.slice(0, 0, 10).slice(1, 0, 10) << "\n";
    torch::Tensor comp1 = m1 + h1 * Z1;
    torch::Tensor comp2 = m2 + h2 * Z2;

    //return (w * comp1 + (1-w) * comp2);
    return torch::where(mask, comp1, comp2);
}

torch::Tensor unique_tensors(const torch::Tensor& input){

    // Find indices where consecutive elements are different and
    // Prepend the number -1 to the diff tensor
    auto diff = torch::cat({torch::tensor({-1}, input.options()), torch::diff(input, 1).squeeze(0)}, 0);
    
    // Replace non-zero values with 1
    auto mask = torch::where(diff != 0, torch::ones_like(diff), diff).to(torch::kBool);
    
    // Extract unique elements using the indices
    auto unique = torch::masked_select(input, mask);

    return unique;
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

void print_model_parameters(const torch::nn::Module& model) {
    // Iterate over named parameters and format the output as "name: value"
    for (const auto& named_param : model.named_parameters()) {
        std::cout << named_param.key() << ": " << named_param.value() << "\n";
    }
}


void report_nan(const torch::Tensor& tensor) {
    if (torch::isnan(tensor).any().item<bool>()) {
        std::cout << "Tensor contains NaN values at positions:\n";

        // Create a mask of NaNs
        torch::Tensor nan_mask = torch::isnan(tensor);

        // Get indices where mask is true
        torch::Tensor indices = torch::nonzero(nan_mask).squeeze();

        for (int64_t i = 0; i < indices.size(0); ++i) {
            int64_t idx = indices[i].item<int64_t>();
            std::cout << "Index " << idx << ": NaN\n";
        }
    } else {
        std::cout << "No NaNs\n";
    }
}


