#include <torch/torch.h>
#include "TFTModel.h"

// This example trains TFTNGARCHModelImpl to recover NGARCH parameters
// from simulated NGARCH returns and variances.

// Simple container for dataset tensors
struct NGARCHDataset {
    torch::Tensor inputs;      // [N, T, input_size]
    torch::Tensor returns;     // [N, T]
    torch::Tensor variances;   // [N, T]
};

// Generate a synthetic NGARCH dataset using the model's simulate_ngarch helper
static NGARCHDataset generate_ngarch_dataset(TFTNGARCHModel &model,
                                             const torch::Tensor &true_params,
                                             int64_t n_samples,
                                             int64_t T) {
    // true_params: [5]
    TORCH_CHECK(true_params.dim() == 1 && true_params.size(0) == 5,
                "true_params must be 1D of size 5");

    auto options = true_params.options();

    // Repeat true params across samples: [N, 5]
    auto params_batch = true_params.unsqueeze(0)
                            .expand({n_samples, -1})
                            .contiguous();

    // Simulate NGARCH paths: returns x and variances h
    auto sim = model->simulate_ngarch(params_batch, static_cast<size_t>(T));
    auto returns = sim.first;   // [N, T]
    auto variances = sim.second; // [N, T]

    // ----- Multi-feature engineering for TFT stability -----
    auto r = returns;      // [N, T]
    auto h = variances;    // [N, T]
    int64_t N = r.size(0);
    int64_t TT = r.size(1);

    // Lag-1 return
    auto r_lag1 = torch::cat({torch::zeros({N, 1}, r.options()),
                              r.index({torch::indexing::Slice(), torch::indexing::Slice(0, -1)})}, 1);

    // Lag-2 return
    auto r_lag2 = torch::cat({torch::zeros({N, 2}, r.options()),
                              r.index({torch::indexing::Slice(), torch::indexing::Slice(0, -2)})}, 1);

    // Absolute and squared returns
    auto abs_r = r.abs();
    auto sq_r  = r * r;

    // Lagged conditional variance
    auto h_lag1 = torch::cat({torch::zeros({N, 1}, h.options()),
                              h.index({torch::indexing::Slice(), torch::indexing::Slice(0, -1)})}, 1);

    // Assemble final input features
    auto f1 = r.unsqueeze(-1);
    auto f2 = h.unsqueeze(-1);
    auto f3 = r_lag1.unsqueeze(-1);
    auto f4 = r_lag2.unsqueeze(-1);
    auto f5 = abs_r.unsqueeze(-1);
    auto f6 = sq_r.unsqueeze(-1);
    auto f7 = h_lag1.unsqueeze(-1);

    auto inputs = torch::cat({f1, f2, f3, f4, f5, f6, f7}, -1);  // [N, T, 7]

    return {inputs, returns, variances};
}

int main() {
    try {
        torch::manual_seed(0);

        // --- Configure TFT for NGARCH parameter learning ---
        tft::TFTConfig config;
        config.total_time_steps = 192;   // full sequence length T
        config.num_encoder_steps = 168;  // encoder context (can be tuned)
        config.input_size = 7;   // we now feed 7 engineered features
        config.static_input_loc.clear();
        config.input_obs_loc = {0,1,2,3,4,5,6};
        config.output_size = 5;          // 5 NGARCH parameters
        config.quantiles = {0.5f};       // single (mean-like) output per param

        auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        std::cout << "Using device: " << (device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;

        // Instantiate NGARCH model with explicit arguments
        auto model = TFTNGARCHModel(
                                        "NGARCH",            // vol_type
                                        config.input_size,    // input_size
                                        64,                   // hidden_size (choose a reasonable value)
                                        config.num_encoder_steps, // num_encoder_steps
                                        config.output_size,   // output_size
                                        0.1f,                 // dropout_rate
                                        4                     // num_heads
                                    );
        model->to(device);

        // --- True NGARCH parameters used for simulation ---
        // [omega, alpha, phi, lambda, gamma]
        auto true_params = torch::tensor({0.01, 0.05, 0.90, 0.10, -0.30},
                                         torch::TensorOptions().dtype(torch::kDouble));

        int64_t n_samples = 512;
        int64_t T = config.total_time_steps;

        // --- Generate synthetic dataset ---
        std::cout << "Generating NGARCH dataset..." << std::endl;
        auto dataset = generate_ngarch_dataset(model, true_params, n_samples, T);
        
        // Ensure dataset tensors are float32 for the TFT model
        dataset.inputs    = dataset.inputs.to(torch::kFloat);
        dataset.returns   = dataset.returns.to(torch::kFloat);
        dataset.variances = dataset.variances.to(torch::kFloat);

        // Split into train / valid (simple 80/20 split)
        int64_t n_train = static_cast<int64_t>(n_samples * 0.8);
        int64_t n_valid = n_samples - n_train;

        auto train_inputs   = dataset.inputs.narrow(0, 0, n_train);
        auto train_returns  = dataset.returns.narrow(0, 0, n_train);
        auto train_variance = dataset.variances.narrow(0, 0, n_train);

        auto valid_inputs   = dataset.inputs.narrow(0, n_train, n_valid);
        auto valid_returns  = dataset.returns.narrow(0, n_train, n_valid);
        auto valid_variance = dataset.variances.narrow(0, n_train, n_valid);

        // --- Optimizer ---
        torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));

        int64_t num_epochs = 30;

        for (int64_t epoch = 0; epoch < num_epochs; ++epoch) {
            model->train();

            // For simplicity we train in a single batch with all training samples
            auto x_batch = train_inputs.to(device).to(torch::kFloat);              // [N_train, T, 1]
            auto r_batch = train_returns.unsqueeze(-1).to(device).to(torch::kFloat);   // [N_train, T, 1]
            auto h_batch = train_variance.unsqueeze(-1).to(device).to(torch::kFloat);  // [N_train, T, 1]
            
            // Forward: predict NGARCH parameters per sample
            auto predicted_params = model->forward(x_batch);     // expected [N_train, 5]
            
            // Compute loss between implied and true variances
            auto loss = model->params_loss(predicted_params, r_batch, h_batch);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            // --- Validation ---
            model->eval();
            auto vx = valid_inputs.to(device).to(torch::kFloat);
            auto vr = valid_returns.unsqueeze(-1).to(device).to(torch::kFloat);
            auto vh = valid_variance.unsqueeze(-1).to(device).to(torch::kFloat);

            auto vparams = model->forward(vx);
            auto vloss   = model->params_loss(vparams, vr, vh);

            std::cout << "Epoch " << (epoch + 1)
                      << " / " << num_epochs
                      << " - train loss: " << loss.item<double>()
                      << ", valid loss: " << vloss.item<double>()
                      << std::endl;
        }

        // --- Inspect learned parameters on a single sample ---
        model->eval();

        auto sample_input = dataset.inputs[0].unsqueeze(0).to(device).to(torch::kFloat);   // [1, T, 1]
        auto learned_params = model->forward(sample_input);              // [1, 5]

        std::cout << "True NGARCH params:   " << true_params << std::endl;
        std::cout << "Learned NGARCH params:" << learned_params.squeeze(0).cpu() << std::endl;

    } catch (const c10::Error &e) {
        std::cerr << "LibTorch error: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception &e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
