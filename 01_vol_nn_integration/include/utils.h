#pragma once

#include <torch/torch.h>
#include <unordered_map>
#include <fstream>
#include <cassert>
#include "ATen/ops/div.h"
#include "ATen/ops/exp.h"

using torch::indexing::Slice;

// Declare your helper functions
torch::Tensor neg_log_likelihood(const torch::Tensor& z, const torch::Tensor& h);
torch::Tensor norm_pdf(torch::Tensor& x, torch::Tensor mu, torch::Tensor sig);
torch::Tensor penalty (torch::Tensor rate, torch::Tensor lb, torch::Tensor ub, torch::Tensor xp);
torch::Tensor unique_tensors(const torch::Tensor& input);
void print_model_parameters(const torch::nn::Module& model);
void to_csv(const std::vector<float>& loss_vec, const std::string& filename);
void load_trained_model(torch::nn::Module& model, const std::string& path);
void report_nan(const torch::Tensor& tensor);

template <typename ModelType>
torch::Tensor final_loss(ModelType& model, torch::Tensor inputs){
      auto [z, h] = model->forward(inputs);
      auto loss = neg_log_likelihood(z, h);
      std::cout << "Final loss: " << loss << "\n";
      return loss;
}

// template functions
template <typename ModelType, typename CriterionType, typename OptimType>
std::tuple<ModelType, std::vector<double>> train_model(ModelType& model, 
                                                   CriterionType& criterion, 
                                                   OptimType& optimizer, 
                                                   int64_t num_epochs, 
                                                   torch::Tensor inputs)
{

  //torch::Tensor loss{}, pen{};
  float running_loss = 0.0;
  std::vector<double> loss_vec{};

  model->train();

  // Training loop
  for (int64_t epoch{0}; epoch < num_epochs; ++epoch) {

    auto [z, h] = model->forward(inputs);
    auto loss = criterion(z, h);

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    auto current_loss = loss.template item<double>();
    loss_vec.push_back(current_loss);
    if (std::abs(running_loss - current_loss) <  1e-3){
        std::cout << "Breaking at epoch {}: " << epoch << "; " << "Loss: "<< current_loss << "\n";
        break;
    }
    running_loss = current_loss;
    std::cout << "Done with epoch {}: " << epoch << "; " << "Loss: "<< current_loss << "\n";
  }
  
  return std::make_tuple(model, loss_vec);
}

template <typename ModelType, typename CriterionType, typename OptimType>
std::tuple<ModelType, std::vector<double>> trainNet(ModelType& model, 
                                                   CriterionType& criterion, 
                                                   OptimType& optimizer, 
                                                   int64_t num_epochs, 
                                                   torch::TensorList inputs)
{

  //torch::Tensor loss{}, pen{};
  float running_loss = 0.0;
  std::vector<double> loss_vec{};

  model->train();

  // Training loop
  for (int64_t epoch{0}; epoch < num_epochs; ++epoch) {

    auto [z, h] = model->forward(inputs);
    auto loss = criterion(z, h);

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    auto current_loss = loss.template item<double>();
    loss_vec.push_back(current_loss);
    if (std::abs(running_loss - current_loss) <  1e-3){
        std::cout << "Breaking at epoch {}: " << epoch << "; " << "Loss: "<< current_loss << "\n";
        break;
    }
    running_loss = current_loss;
    std::cout << "Done with epoch {}: " << epoch << "; " << "Loss: "<< current_loss << "\n";
  }
  
  return std::make_tuple(model, loss_vec);
}

// CF-based option pricer for Heston–Nandi (or any model with cf(u, T))
template <typename ModelType>
torch::Tensor Pricer(
    const ModelType& model,
    const torch::Tensor& options_tensor, // [N, 2] => [K, T]
    bool is_call = true,
    int n_integration_points = 2000,
    double u_max = 200.0)
{
    // Ensure we’re on CPU and in double precision
    auto opts = options_tensor.to(torch::kDouble).cpu().contiguous();
    TORCH_CHECK(opts.dim() == 2 && opts.size(1) == 2,
                "options_tensor must be [N, 2] with columns [K, T].");

    const int64_t n_opts = opts.size(0);
    auto prices = torch::empty({n_opts}, torch::kDouble);

    // Make Simpson’s rule point count even
    if (n_integration_points % 2 != 0)
        n_integration_points += 1;

    for (int64_t i = 0; i < n_opts; ++i) {
        const double K = opts[i][0].item<double>();
        const double T = opts[i][1].item<double>();
        const double logK = std::log(K);

        // Discount factors
        const double disc_r = std::exp(-model.r * T);
        const double disc_q = std::exp(-model.q * T);

        // φ(-i) is E[e^{-i X_T}] with u = -i => E[S_T]
        // Used in the P1 integral
        const std::complex<double> minus_i(0.0, -1.0);
        const std::complex<double> phi_minus_i = model.cf(minus_i, T);

        // Defensive check (optional)
        // If |phi_minus_i| is extremely small, something is wrong.
        // You can replace this with a TORCH_CHECK if you prefer.
        if (std::abs(phi_minus_i) < 1e-12) {
            throw std::runtime_error("phi(-i) is too small (numerical issue).");
        }

        // Simpson’s rule integrator on (0, u_max].
        auto simpson = [&](auto&& f) -> double {
            const double a = 1e-6;  // avoid u=0 singularity
            const double b = u_max;
            const int N = n_integration_points;
            const double h = (b - a) / static_cast<double>(N);

            double sum = f(a) + f(b);
            for (int k = 1; k < N; ++k) {
                const double x = a + h * static_cast<double>(k);
                sum += (k % 2 ? 4.0 : 2.0) * f(x);
            }
            return sum * h / 3.0;
        };

        // Integrand for P2
        auto integrand_P2 = [&](double u) -> double {
            std::complex<double> iu(0.0, u);
            std::complex<double> u_c(u, 0.0);

            std::complex<double> phi_u = model.cf(u_c, T);
            std::complex<double> e_minus_iulogK =
                std::exp(std::complex<double>(0.0, -u * logK));

            // e^{-iu log K} * φ(u) / (i u)
            std::complex<double> val = e_minus_iulogK * phi_u / iu;
            return val.real(); // Re[ ... ]
        };

        // Integrand for P1
        auto integrand_P1 = [&](double u) -> double {
            std::complex<double> iu(0.0, u);
            std::complex<double> u_minus_i(u, -1.0);

            std::complex<double> phi_u_minus_i = model.cf(u_minus_i, T);
            std::complex<double> e_minus_iulogK =
                std::exp(std::complex<double>(0.0, -u * logK));

            // e^{-iu log K} * φ(u - i) / (i u φ(-i))
            std::complex<double> val =
                e_minus_iulogK * phi_u_minus_i / (iu * phi_minus_i);
            return val.real(); // Re[ ... ]
        };

        // Perform numerical integration
        const double I1 = simpson(integrand_P1);
        const double I2 = simpson(integrand_P2);

        const double P1 = 0.5 + I1 / M_PI;
        const double P2 = 0.5 + I2 / M_PI;

        double call_price = model.S0 * disc_q * P1 - K * disc_r * P2;
        double price = call_price;

        if (!is_call) {
            // Put via put-call parity: P = C - S0 e^{-qT} + K e^{-rT}
            price = call_price - model.S0 * disc_q + K * disc_r;
        }

        prices[i] = price;
    }

    return prices;
}

template <typename ModelType>
void replace_params(ModelType& model, const torch::Tensor& new_params) {
    auto parameters = model->parameters();
    TORCH_CHECK(parameters.size() == new_params.numel(), "Number of provided parameters does not match the number of model parameters.");

    int64_t offset = 0;
    for (auto& param : parameters) {
        auto param_size = param.numel();
        param.set_data(new_params.slice(0, offset, offset + param_size).view(param.sizes()).clone());
        offset += param_size;
    }
}

template <typename ModelType>
void update_params(ModelType& model, const std::vector<torch::Tensor>& new_params) {
    auto parameters = model->parameters();
    TORCH_CHECK(parameters.size() == new_params.size(), "Number of provided tensors does not match the number of model parameters.");

    for (size_t i = 0; i < parameters.size(); ++i) {
        TORCH_CHECK(parameters[i].sizes() == new_params[i].sizes(), "Size mismatch between model parameter and provided tensor at index ", i);
        parameters[i].set_data(new_params[i].clone());
    }
}