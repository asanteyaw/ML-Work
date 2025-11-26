#pragma once

#include <torch/torch.h>
#include <any>
#include <map>
#include <unordered_map>
#include <typeindex>
#include <fstream>
#include <functional>
#include <algorithm> // for std::for_each


// Declare your helper functions
torch::Tensor sliding_windows(torch::Tensor ts, int64_t window_size);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
train_test_split(const torch::Tensor& x, const torch::Tensor& y, double test_ratio, int64_t dim = 0);
void to_csv(const std::vector<float>& loss_vec, const std::string& filename);
std::unordered_map<std::string, torch::Tensor> load_trained_model(const std::string& path);
void save_checkpoint(const std::string& path, torch::nn::Module& model, torch::optim::Optimizer& optimizer, int64_t epoch, float loss);
void save_checkpoint(const std::string& path, torch::nn::Module& model);
bool load_checkpoint(const std::string& path, torch::nn::Module& model, torch::optim::Optimizer& optimizer, int64_t& epoch, float& loss);
void load_checkpoint(const std::string& path, torch::nn::Module& model);
std::unordered_map<std::string, torch::Tensor> name_tensor_map(const torch::Tensor& tensor, const std::vector<std::string>& names);
//void tensor_info(const torch::Tensor& tensor);
void tensor_info(const torch::Tensor& tensor, const std::vector<std::string>& options);
torch::Tensor generate_random_values(double lower_bound, double upper_bound, size_t rows);

// template functions
template <typename ModelType, typename OptimType>
ModelType trainNet(ModelType model, OptimType& optimizer, torch::TensorList data, int64_t num_epochs, 
                   const std::string& checkpoint_path, const std::string& temp_path){

   torch::Tensor loss{};
   float running_loss{0.0};
   int64_t start_epoch{0};
   float last_loss = 0.0;
   std::vector<float> loss_vec{};

    // Try to load the checkpoint
    if (load_checkpoint(checkpoint_path, *model, optimizer, start_epoch, last_loss)) {
        std::cout << "Resumed from epoch " << start_epoch << " with loss " << last_loss << std::endl;
    } else {
	std::cout << "Starting training from scratch" << std::endl;
    }

   // prepare inputs
   auto returns = data[0];

   //std::cout << "Features Size: " << features.sizes() << "\n";

   // Training loop
   for (int64_t epoch{start_epoch > 0 ? start_epoch + 1 : start_epoch}; epoch < num_epochs; ++epoch) {
      model->train();
      auto [z, h] = model->forward(returns);   // output parameters
      auto loss = model->loglikelihood(z, h);

      optimizer.zero_grad();
      loss.backward();
      optimizer.step();

      auto current_loss = loss.template item<float>();
      loss_vec.push_back(current_loss);
      to_csv(loss_vec, temp_path);   // intermittently save loss

      // Save checkpoint every 5 epochs
      if ((epoch+1) % 5 == 0) {
          save_checkpoint(checkpoint_path, *model, optimizer, epoch, current_loss);
          std::cout << "Checkpoint saved at epoch: " << epoch << "\n";
      }

      if (std::abs(running_loss - current_loss) <  1e-3){
         std::cout << "Breaking at epoch {}: " << epoch << "; " << "Loss: "<< current_loss << "\n";
         break;
      }

      running_loss = current_loss;
      std::cout << "Done with epoch {}: " << epoch << "; " << "Loss: "<< current_loss << "\n";
   }

   return model;
}

template <typename ModelType>
void final_loss(ModelType model, const torch::TensorList&){

//      auto [h1, z] = IVRMSE(res, opt, news, model);
      auto loss = torch::tensor(0.0);
      std::cout << "Final loss: " << loss << "\n";
}














