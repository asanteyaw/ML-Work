#pragma once

#include <torch/torch.h>
#include <any>
#include <map>
#include <unordered_map>
#include <typeindex>
#include <fstream>
#include <functional>
#include <algorithm> // for std::for_each

namespace utilities {

    class Normalizer {

        public:
            // Fit the normalizer: calculate min and max from the training data
            void fit(const torch::Tensor& train_data) {
                data_min = train_data.min().item<float>();
                data_max = train_data.max().item<float>();
                is_fitted = true;
            }

            // Transform data: normalize it to the range [a, b]
            torch::Tensor transform(const torch::Tensor& data, float a = 0.0f, float b = 1.0f) const {
                if (!is_fitted) {
                    throw std::runtime_error("Normalizer must be fitted before calling transform.");
                }
                return ((data - data_min) / (data_max - data_min)) * (b - a) + a;
            }

            // Fit and transform: combine fit and transform for a single dataset
            torch::Tensor fit_transform(const torch::Tensor& train_data, float a = 0.0f, float b = 1.0f) {
                fit(train_data);
                return transform(train_data, a, b);
            }

            // Inverse transform: return data to its original scale
            torch::Tensor inverse_transform(const torch::Tensor& normalized_data, float a = 0.0f, float b = 1.0f) const {
                if (!is_fitted) {
                    throw std::runtime_error("Normalizer must be fitted before calling inverse_transform.");
                }
                return ((normalized_data - a) / (b - a)) * (data_max - data_min) + data_min;
            }

            // Getters for debugging or inspection
            float get_min() const {
                if (!is_fitted) {
                    throw std::runtime_error("Normalizer must be fitted before calling get_min.");
                }
                return data_min;
            }

            float get_max() const {
                if (!is_fitted) {
                    throw std::runtime_error("Normalizer must be fitted before calling get_max.");
                }
                return data_max;
            }

            // Save normalization parameters to a file
            void save(const std::string& filename) const {
                std::ofstream out_file(filename, std::ios::binary);
                if (!out_file) {
                    throw std::runtime_error("Failed to open file for saving: " + filename);
                }
                out_file.write(reinterpret_cast<const char*>(&data_min), sizeof(data_min));
                out_file.write(reinterpret_cast<const char*>(&data_max), sizeof(data_max));
                out_file.write(reinterpret_cast<const char*>(&is_fitted), sizeof(is_fitted));
                out_file.close();
            }

            // Load normalization parameters from a file
            void load(const std::string& filename) {
                std::ifstream in_file(filename, std::ios::binary);
                if (!in_file) {
                    throw std::runtime_error("Failed to open file for loading: " + filename);
                }
                in_file.read(reinterpret_cast<char*>(&data_min), sizeof(data_min));
                in_file.read(reinterpret_cast<char*>(&data_max), sizeof(data_max));
                in_file.read(reinterpret_cast<char*>(&is_fitted), sizeof(is_fitted));
                in_file.close();
            }

        private:
            float data_min;
            float data_max;
            bool is_fitted = false;    
    };

}  // namespace utilities

// Declare your helper functions
torch::Tensor sliding_windows(torch::Tensor ts, int64_t window_size);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
train_test_split(const torch::Tensor& x, const torch::Tensor& y, double test_ratio, int64_t dim = 0);
void to_csv(const std::vector<std::pair<float, float>>& loss_vec, const std::string& filename);
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
                   const std::string& checkpoint_path, const std::string& temp_path, float tfr = 0.5, bool dynamic_tf = true){

   torch::Tensor loss{};
   float running_loss{0.0}, current_val_loss{}, best_val_loss{232566.9};
   int64_t start_epoch{0}, patience{3}, patience_counter{0};
   float last_loss = 0.0;
   std::vector<std::pair<float, float>> losses{};

    // Try to load the checkpoint
    if (load_checkpoint(checkpoint_path, *model, optimizer, start_epoch, last_loss)) {
        std::cout << "Resumed from epoch " << start_epoch << " with loss " << last_loss << std::endl;
    } else {
	std::cout << "Starting training from scratch" << std::endl;
    }

   // prepare inputs
   auto features = data[0];
   auto targets = data[1];
   auto val_features = data[2];
   auto val_targets = data[3]; 
   auto ground_truth = data[4];

   //std::cout << "Features Size: " << features.sizes() << "\n";

   // Training loop
   for (int64_t epoch{start_epoch > 0 ? start_epoch + 1 : start_epoch}; epoch < num_epochs; ++epoch) {
      model->train();
      
      auto params = model->forward(features, ground_truth, tfr, /*use_tf=*/true, /*dynamic_tf=*/true);   // output parameters
      auto loss = model->params_loss(params, features, targets);
      std::cout << "Training Params :\n" << params.slice(0, 0, 5) << "\n";

      optimizer.zero_grad();
      loss.backward();
      optimizer.step();

      auto current_loss = loss.template item<float>();
    
      {
         model->eval();
         torch::NoGradGuard no_grad;
         
         auto val_params = model->forward(val_features);            
         auto val_loss = model->params_loss(val_params, val_features, val_targets);
         std::cout << "Val Params :\n" << val_params.slice(0, 0, 5) << "\n";
         current_val_loss = val_loss.template item<float>();
         if (current_val_loss < best_val_loss) {
             best_val_loss = current_val_loss;
             patience_counter = 0; // Reset patience
         } else {
             patience_counter++;
             if (patience_counter >= patience) {
                 std::cout << "Breaking at epoch {}: " << epoch << "; " << "Loss: "<< current_loss << "\n";
                 break;
             }
         }

      }

      losses.push_back({current_loss, current_val_loss}); 
      to_csv(losses, temp_path);

      // Save checkpoint every 5 epochs
      if ((epoch+1) % 5 == 0) {
          save_checkpoint(checkpoint_path, *model, optimizer, epoch, current_loss);
          std::cout << "Checkpoint saved at epoch: " << epoch << "\n";
      }
      std::cout << "Done with epoch: " << epoch << "; " << "train loss: "<< current_loss<< "; "<< "val loss: "<< current_val_loss << "\n";
      //if (epoch == 10) break; 
      if (dynamic_tf && tfr > 0 && epoch > 10) tfr -= 0.01; // Gradually reduce reliance on ground truth
   }

   return model;
}

template <typename ModelType>
void final_loss(ModelType model, const torch::TensorList&){

//      auto [h1, z] = IVRMSE(res, opt, news, model);
      auto loss = torch::tensor(0.0);
      std::cout << "Final loss: " << loss << "\n";
}














