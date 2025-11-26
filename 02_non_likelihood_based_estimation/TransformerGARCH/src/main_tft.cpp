#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "../include/TFTModel.h"
#include "../include/utils.h"

void save_tensor_as_csv(const torch::Tensor& tensor, const std::string& filename) {
    // Ensure the tensor is on the CPU and detached
    auto cpu_tensor = tensor.to(torch::kCPU).detach();

    // Open a file stream
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    // Get the tensor dimensions
    auto sizes = cpu_tensor.sizes();
    if (sizes.size() != 2) {
        std::cerr << "Error: Only 2D tensors can be saved as CSV." << std::endl;
        file.close();
        return;
    }

    // Write tensor values to the CSV file
    for (int64_t i = 0; i < sizes[0]; ++i) {
        for (int64_t j = 0; j < sizes[1]; ++j) {
            file << cpu_tensor.index({i, j}).item<float>();
            if (j < sizes[1] - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
    std::cout << "Tensor saved as CSV to " << filename << std::endl;
}

// TFT-compatible training function
template <typename ModelType, typename OptimType>
ModelType trainTFTNet(ModelType model, OptimType& optimizer, torch::TensorList data, int64_t num_epochs, 
                      const std::string& checkpoint_path, const std::string& temp_path){

   torch::Tensor loss{};
   float running_loss{0.0}, current_val_loss{}, best_val_loss{232566.9};
   int64_t start_epoch{0}, patience{5}, patience_counter{0};
   float last_loss = 0.0;
   std::vector<std::pair<float, float>> losses{};

    // Try to load the checkpoint
    if (load_checkpoint(checkpoint_path, *model, optimizer, start_epoch, last_loss)) {
        std::cout << "Resumed from epoch " << start_epoch << " with loss " << last_loss << std::endl;
    } else {
        std::cout << "Starting TFT training from scratch" << std::endl;
    }

   // prepare inputs
   auto features = data[0];
   auto targets = data[1];
   auto val_features = data[2];
   auto val_targets = data[3]; 
   auto ground_truth = data[4];

   std::cout << "TFT Features Size: " << features.sizes() << "\n";
   std::cout << "TFT Targets Size: " << targets.sizes() << "\n";

   // Training loop
   for (int64_t epoch{start_epoch > 0 ? start_epoch + 1 : start_epoch}; epoch < num_epochs; ++epoch) {
      model->train();
      
      // Forward pass through TFT model
      auto params = model->forward(features);   // output parameters
      auto loss = model->params_loss(params, features, targets);
      std::cout << "Training TFT Params :\n" << params.slice(0, 0, 5) << "\n";

      optimizer.zero_grad();
      loss.backward();
      optimizer.step();

      auto current_loss = loss.template item<float>();
    
      // Validation
      {
         model->eval();
         torch::NoGradGuard no_grad;
         
         auto val_params = model->forward(val_features);            
         auto val_loss = model->params_loss(val_params, val_features, val_targets);
         std::cout << "TFT Val Params :\n" << val_params.slice(0, 0, 5) << "\n";
         current_val_loss = val_loss.template item<float>();
         if (current_val_loss < best_val_loss) {
             best_val_loss = current_val_loss;
             patience_counter = 0; // Reset patience
         } else {
             patience_counter++;
             if (patience_counter >= patience) {
                 std::cout << "Early stopping at epoch: " << epoch << "; Loss: "<< current_loss << "\n";
                 break;
             }
         }
      }

      losses.push_back({current_loss, current_val_loss}); 
      to_csv(losses, temp_path);

      // Save checkpoint every 10 epochs
      if ((epoch+1) % 10 == 0) {
          save_checkpoint(checkpoint_path, *model, optimizer, epoch, current_loss);
          std::cout << "TFT Checkpoint saved at epoch: " << epoch << "\n";
      }
      std::cout << "TFT Epoch: " << epoch << "; train loss: "<< current_loss<< "; val loss: "<< current_val_loss << "\n";
   }

   return model;
}

int main() {

   auto cuda_available = torch::cuda::is_available();
   torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
   std::cout << (cuda_available ? "CUDA available. Training TFT on GPU." : "Training TFT on CPU.") << '\n';

   torch::manual_seed(42);

   std::cout << "Training TFT-NGARCH Model!!\n";
  
   const std::string c_path = "../checkpoints/ngarch/synthetic/tft/";
   const std::string t_path = "../losses/ngarch_tft_loss.csv";
   int64_t n_epochs = 500;
   std::string vol_type = "NGARCH";
   int64_t in_size = 1;
   int64_t hid_size = 64;        // Larger hidden size for TFT
   int64_t num_encoder_steps = 250;  // Sequence length for TFT
   int64_t out_size = 5;         // 5 NGARCH parameters
   size_t ts_len = 8'000;        // simulate data points
   double tr = 0.1;

   std::vector<float> loss_vals{};
   size_t n_batch = 10'000;      // Number of batches

   // Initialize the TFT-NGARCH model
   auto model = TFTNGARCHModel(vol_type, in_size, hid_size, num_encoder_steps, out_size, 0.1f, 4);
   model->to(device);
   
   // define optimizer with different learning rate for TFT
   torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.0005));

   // Generate random NGARCH parameters for simulation
   torch::Tensor true_params = torch::cat({
        generate_random_values(0.000121, 0.000324, n_batch).unsqueeze(1),     // omega
        generate_random_values(0.035, 0.047, n_batch).unsqueeze(1),           // alpha
        generate_random_values(0.85, 0.95, n_batch).unsqueeze(1),             // phi
        generate_random_values(0.045, 0.057, n_batch).unsqueeze(1),           // lambda
        generate_random_values(0.7, 2.0, n_batch).unsqueeze(1)                // gamma
   }, 1); // Concatenate along dimension 1

   // Simulate NGARCH data
   auto [true_returns, true_variances] = model->simulate_ngarch(true_params.to(device), ts_len);

   std::cout << "Generated " << n_batch << " batches of NGARCH data with " << ts_len << " time steps each\n";
   
   // Save generated data
   save_tensor_as_csv(true_params.cpu(), "../data/tft_ngarch_params.csv");
   save_tensor_as_csv(true_returns.cpu(), "../data/tft_log_returns.csv");
   save_tensor_as_csv(true_variances.cpu(), "../data/tft_conditional_variances.csv");

   utilities::Normalizer x_norm;
   utilities::Normalizer y_norm;

   // obtain train-test split
   auto [X_train, Y_train, x_test, y_test] = train_test_split(true_returns, true_variances, tr);

   // obtain train-val split
   auto [x_train, y_train, x_val, y_val] = train_test_split(X_train, Y_train, tr);

   // Prepare data for TFT: [batch_size, seq_len, input_size]
   x_train = (x_norm.fit_transform(x_train)).unsqueeze(-1).to(device);  // Add feature dimension
   y_train = (y_norm.fit_transform(y_train)).unsqueeze(-1).to(device);
   x_val = (x_norm.transform(x_val)).unsqueeze(-1).to(device);
   y_val = (y_norm.transform(y_val)).unsqueeze(-1).to(device);
   x_test = (x_norm.transform(x_test)).unsqueeze(-1).to(device);
   y_test = y_test.unsqueeze(-1).to(device);

   // Get corresponding true parameters for training batches
   auto true_params_slice = true_params.slice(0, 0, x_train.size(0)).to(device);
   
   // train TFT model
   auto trained_model = trainTFTNet(model, optimizer, {x_train, y_train, x_val, y_val, true_params_slice}, n_epochs, c_path, t_path);
   torch::save(trained_model, "../models/model_tft_ngarch.pt");

   // test TFT model
   {
      torch::NoGradGuard no_grad;

      torch::load(model, "../models/model_tft_ngarch.pt");
      model->eval();
      auto params = model->predict(x_test);   // output parameters
      
      // ensure constraints are satisfied
      auto test_loss = model->params_loss(params, 
                                          x_norm.inverse_transform(x_test.squeeze(-1)), 
                                          y_test);

      std::cout << "TFT Final Test loss: " << test_loss << "\n";
      std::cout << "TFT Predicted params sample:\n" << params.slice(0, 0, 5) << "\n";
      std::cout << "TFT True params sample:\n" << true_params.slice(0, x_train.size(0), x_train.size(0) + 5) << "\n";
   }

   return 0;
}
