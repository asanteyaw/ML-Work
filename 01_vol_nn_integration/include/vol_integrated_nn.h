#pragma once

#include <cstdint>
#include <torch/torch.h>
#include <utility>
#include <vector>
#include <tuple>
#include "GARCH_Volatility.h"
#include "VIFunctional.h"
#include "c10/util/Exception.h"
#include "torch/csrc/autograd/generated/variable_factories.h"

// ============================================================================
// UTILITY FUNCTIONS FOR VOLATILITY-INTEGRATED RNNS
// ============================================================================

/**
 * @brief Create volatility-integrated RNN parameters
 */

// ============================================================================
// VOLATILITY-INTEGRATED PARAMETERS
// ============================================================================

/**
 * @brief Extended parameters for volatility-integrated RNNs
 */
namespace qmodels = garch_type_model;
namespace net_utils = vol_rnn_functional;

struct VolRNNModelImpl : torch::nn::Module {

   std::vector<net_utils::NetWeight> rnn_params;                  // Standard RNN parameters
   std::shared_ptr<qmodels::GARCHModel> garch_model;              // GARCH volatility model
   net_utils::NetConfig topol;                                    // Network Topology
   net_utils::Shelf series;
   torch::Tensor gamma_f;                                         // Volatility scaling parameter(s)
   int64_t gate_multiplier;                                       // 3 for GRU, 4 for LSTM
   torch::Tensor omega{}, alpha{}, phi{}, lamda{}, gamma{};       // HN parameters
   torch::Tensor gamma1{}, gamma2{}, vphi{}, rho{}, lb{}, ub{};               // CHN extra parameters
   
   VolRNNModelImpl() = default;
   
   VolRNNModelImpl(torch::Tensor garch_params,
                  std::vector<torch::Tensor>& aux,
                  const int64_t in_size,
                  const int64_t hid_size,
                  const int64_t n_layers,
                  double dropout = 0.0,
                  int64_t num_gates = 3,
                  bool bidirectional = false,
                  bool has_bias = true)
      :  gamma_f(aux[0]), lb(aux[1]), ub(aux[2]), gate_multiplier(num_gates) 
         {make_vi(in_size, hid_size, n_layers, dropout, has_bias, bidirectional, garch_params);}

   void make_vi (int input_size, int hidden_size, int num_layers, 
                        double dropout, bool use_bias, bool bidirectional,
                        torch::Tensor garch_params,
                        const std::string& garch_type = "hn",
                        torch::ScalarType dtype = torch::kFloat32);
   void VIParameters (std::vector<net_utils::NetWeight> rnn_params, 
                      std::shared_ptr<qmodels::GARCHModel> garch_model, 
                      torch::Tensor gamma_f);

   std::vector<torch::Tensor> get_vi_parameters(); 
   std::pair<torch::Tensor, torch::Tensor> forward(torch::TensorList exret);  
   torch::Tensor unscale_parameters(torch::Tensor sc_params);                   
};

TORCH_MODULE(VolRNNModel);

inline void VolRNNModelImpl::make_vi(
   int input_size, int hidden_size, int num_layers, 
   double dropout, bool use_bias, bool bidirectional,
   torch::Tensor garch_params,
   const std::string& garch_type, 
   torch::ScalarType dtype) {
   
   // Create Network Topology
   topol.dropout = dropout;
   topol.dtype = dtype;
   topol.use_bias = use_bias;
   topol.num_layers = num_layers;
   topol.bidirectional = bidirectional;
   topol.in_size = input_size;
   topol.hid_size = hidden_size;
   topol.n_gates = gate_multiplier;

   // Create standard RNN parameters
   auto rnn_params = net_utils::make_params(topol, gate_multiplier);
   
   if (garch_type == "hn") {
      garch_model = std::make_shared<qmodels::HNModel>(garch_params);
   } else if (garch_type == "chn") {
      garch_model = std::make_shared<qmodels::CHNModel>(garch_params);
   } else {
      throw std::invalid_argument("garch_type must be 'hn' or 'chn'");
   }
   
   VIParameters(rnn_params, garch_model, gamma_f);   // register parameters for training
}

static torch::Tensor slope_inverse_transform(const torch::Tensor& scaled_params, const torch::Tensor& lb, const torch::Tensor& ub, torch::Tensor slope) {
    return lb + (ub - lb) / (1 + torch::exp(-slope * scaled_params));
}

/**
 * @brief register parameter of Volatility integrated model
 */
inline void VolRNNModelImpl::VIParameters (std::vector<net_utils::NetWeight> rnn_params, 
                      std::shared_ptr<qmodels::GARCHModel> garch_model, 
                      torch::Tensor gamma_val)
{
   std::vector<torch::Tensor> vol_params = garch_model->get_params();

   omega = register_parameter("omega", vol_params.at(0));
   alpha = register_parameter("alpha", vol_params.at(1));
   phi = register_parameter("phi", vol_params.at(2));
   lamda = register_parameter("lamda", vol_params.at(3));
   gamma = register_parameter("gamma", vol_params.at(4));
   gamma_f = register_parameter("gammaf", gamma_val);

   for (int i = 0; i < rnn_params.size(); ++i) {

        std::string layer_idx = std::to_string(i);

        // Initialize GRU weights and biases
        auto weights = rnn_params.at(i);
        auto weight_ih = weights.weight_ih;
        auto weight_hh = weights.weight_hh;
        auto bias_ih = weights.bias_ih;
        auto bias_hh = weights.bias_hh;

        register_parameter("weight_ih_" + layer_idx, weight_ih);
        register_parameter("weight_hh_" + layer_idx, weight_hh);
        register_parameter("bias_ih_" + layer_idx, bias_ih);
        register_parameter("bias_hh_" + layer_idx, bias_hh);
    }
}

/**
 * @brief Collect all learnable parameters from volatility-integrated RNN
 * @param vi_params Volatility-integrated parameters
 * @return Vector of all learnable tensors
 */
inline std::vector<torch::Tensor> VolRNNModelImpl::get_vi_parameters() {
   auto vi_params = this->parameters();
   std::vector<torch::Tensor> parameters;
   
   for (const auto& param_set : vi_params) {
      parameters.push_back(param_set);
   }
   
   return parameters;
}

/**
 * @brief Set all parameters as learnable in volatility-integrated RNN
 * @param vol_params Volatility-integrated RNN parameters
 * @param learnable Whether parameters should require gradients
 */
inline std::pair<torch::Tensor, torch::Tensor> VolRNNModelImpl::forward(torch::TensorList data) {
   
   auto y = data[0];   // returns data
   auto x = data[1];   // rv data

   auto hl{torch::empty_like(y)}, zl{torch::empty_like(y)};
   auto batch_size = 1;

   auto scaled_params = torch::stack({omega,alpha,phi,lamda,gamma});
   torch::Tensor params = unscale_parameters(scaled_params);
   torch::Tensor om{params[0]}, al{params[1]}, ph{params[2]}, la{params[3]}, ga{params[4]};
   auto vi_params = get_vi_parameters();

   rnn_params = net_utils::make_net_weights(std::vector<torch::Tensor>(vi_params.begin() + 6, vi_params.end()), topol);
   garch_model->set_params({om, al, ph, la, ga});
   auto h_t = torch::var(y);
   auto eta_t = h_t;
   auto eta_out = torch::zeros_like(eta_t);
   auto z_t = garch_model->generate_shock(eta_t, y[0]);
   auto hx = torch::zeros({topol.num_layers, batch_size, topol.hid_size}).to(x.device());
   auto cx = torch::zeros_like(hx);
   std::tuple<torch::Tensor, torch::Tensor> hn;

   series.gamma_f = gamma_f;
   series.h_t = h_t;
   series.rnn_params = std::span<net_utils::NetWeight>(rnn_params.data(), rnn_params.size());
   // std::cout << "Weight ih: " << series.rnn_params[0].weight_ih << "\n";
   hl.index_put_({0}, h_t);
   zl.index_put_({0}, z_t);

   for(int64_t t{1}; t < y.size(0); ++t){

      auto seq = torch::stack({eta_t, y[t-1], x[t-1], z_t}).view({batch_size, -1, topol.in_size});

      // Forward pass
      std::tie(eta_out, hn) = net_utils::neural_step(seq, series, hx, cx, topol);
      eta_t = torch::sqrt(eta_out.squeeze().pow(2));
    
      h_t = garch_model->update_variance(h_t, z_t);
      z_t = garch_model->generate_shock(eta_t, y[t]);
    
      // TORCH_CHECK(h_t.item<float>() < 0.0, "Negative value for h_t");
      if(h_t.isnan().any().item<bool>()) std::cout << "h_t has nan values\n";
      if(z_t.isnan().any().item<bool>()) std::cout << "z_t has nan values\n";

      // updates
      std::tie(cx, hx) = hn;
      hl.index_put_({t}, h_t);
      zl.index_put_({t}, z_t);
   }

   return std::make_pair(zl, hl);
}

inline torch::Tensor VolRNNModelImpl::unscale_parameters(torch::Tensor sc_params) {
   // auto rate = (torch::tensor(10).pow(20)).to(sc_params.device());
   auto rate = torch::tensor(2.0).to(sc_params.device());
   auto unscaled_params = slope_inverse_transform(sc_params, lb, ub, rate);

   return unscaled_params;
}


