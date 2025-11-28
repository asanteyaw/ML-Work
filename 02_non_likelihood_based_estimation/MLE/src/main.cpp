#include <iostream>
#include <vector>
#include "Model.h"
#include "utils.h"


int main() {

   auto cuda_available = torch::cuda::is_available();
   torch::Device device(cuda_available ? torch::kCPU : torch::kCPU);
   std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

   torch::manual_seed(1);  
   
   const std::string c_path = "../checkpoints/ngarch/synthetic/ml/";
   const std::string t_path = "../losses/ngarch_loss2.csv";
   int64_t n_epochs = 6'000;
   std::string vol_type = "EGARCH";
   size_t ts_len = 20'000;     // simulate 50,000 returns and variance data points
   double tr = 0.1;

   std::vector<float> loss_vals{};

   size_t n_batch = 50'000; // Number of rows
   
   // Initialize the Econometric model
   auto model = EconometricModel(vol_type, n_batch);
   model->to(device);

   {
      torch::NoGradGuard no_grad;

      //std::cout << "Params before scaling:\n" << model->parameters() << "\n";
      model->scale_parameters();  // Scale parameters after model instantiation
      //std::cout << "Params after scaling:\n" << model->parameters() << "\n";
   }

   // define optimizer
   torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.001));

   // get data
   //auto [true_returns, true_variances] = model->simulate_ngarch(true_params.to(device), ts_len);

   torch::Tensor true_returns{}, true_variances{}, true_params{};
   torch::load(true_returns, "../../Enc_Dec/data/log_returns.pt");
   torch::load(true_variances, "../../Enc_Dec/data/conditional_variance.pt");
   torch::load(true_params, "../../Enc_Dec/data/ngarch_params.pt");
   std::cout << "Returns data size:\n" << true_returns.sizes() << "\n";
   std::cout << "Variance data size:\n" << true_variances.sizes() << "\n";
   std::cout << "Parameter size:\n" << true_params.sizes() << "\n";
   //std::abort();

   // train model
   auto trained_model = trainNet(model, optimizer, {true_returns,}, n_epochs, c_path, t_path);

   {
       torch::NoGradGuard no_grad;
       model->eval();
       model->get_unscaled_params(0);
   }
   torch::save(trained_model, "../models/model_synthetic_ml.pt");

   // test model
   {
      torch::NoGradGuard no_grad;
     
      torch::save(model, "../models/model_synthetic_ml.pt");
      model->eval();
      //auto params = model.predict(x_test);   // output parameters

      // ensure constraints are satisfied
      //auto test_loss = model.params_loss(params, x_test, y_test);
      //std::cout << "Test loss: " << test_loss << "\n";
   
   }

   return 0;
}



