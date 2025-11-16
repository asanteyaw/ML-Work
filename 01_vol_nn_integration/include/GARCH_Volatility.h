#pragma once

#include <ATen/ops/stack.h>
#include <c10/util/Exception.h>
#include <torch/torch.h>
#include "RNN_Functionals.h"
#include <vector>
#include <tuple>
#include <memory>
#include <fstream>

/**
 * @brief GARCH-type Econometric Model Opertions
 * 
 * This is specifically built for GARCH-type econometric models
 * All the GARCH-type models will inherit from the GARCH base class
 * Currently serves the Heston and Nandi (2000) model it its componenent counterpart
 */
namespace garch_type_model {

// ============================================================================
// GARCH MODEL CLASSES
// ============================================================================

/**
 * @brief Base class for GARCH volatility models with unified parameter management
 */
class GARCHModel : public torch::nn::Module {
public:
    virtual ~GARCHModel() = default;
    
    // Core GARCH functionality
    virtual torch::Tensor update_variance(const torch::Tensor& h_t, const torch::Tensor& z_t) = 0;
    virtual torch::Tensor generate_returns(const torch::Tensor& h_t, const torch::Tensor& z_t, 
                                          double r = 0.0, double d = 0.0) = 0;
    virtual torch::Tensor generate_shock(const torch::Tensor& h_t, const torch::Tensor& y_t,
                                            double r = 0.0, double d = 0.0) = 0;
    
    // Unified parameter management interface
    virtual std::vector<torch::Tensor> get_params() = 0;
    virtual void set_params(std::vector<torch::Tensor> params, bool learnable = true) = 0;
    virtual std::string get_model_name() const = 0;
    virtual void initialize_from_config(const std::map<std::string, double>& params) = 0;
    
    // Model serialization interface
    virtual std::map<std::string, torch::Tensor> get_state_dict() const = 0;
    virtual void load_state_dict(const std::map<std::string, torch::Tensor>& state_dict) = 0;
};

/**
 * @brief Heston-Nandi GARCH(1,1) model
 */
class HNModel : public GARCHModel {
public:
    torch::Tensor omega, alpha, phi, lamda, gamma;
    
    HNModel(double omega_val, double alpha_val, double phi_val, 
                     double lambda_val, double gamma_val,
                     const torch::TensorOptions& options = torch::kFloat32)
        : omega(register_parameter("omega", torch::tensor(omega_val, options)))
        , alpha(register_parameter("alpha",torch::tensor(alpha_val, options)))
        , phi(register_parameter("phi",torch::tensor(phi_val, options)))
        , lamda(register_parameter("lamda",torch::tensor(lambda_val, options)))
        , gamma(register_parameter("gamma",torch::tensor(gamma_val, options))) {}
    
    torch::Tensor update_variance(const torch::Tensor& h_t, const torch::Tensor& z_t) override {
        // h_{t+1} = ω̄ + φ(h_t - ω̄) + α(z_t² - 1 - 2γ√h_t z_t)
               
        return omega + phi * (h_t - omega) + 
               alpha * (z_t.pow(2) - 1 - 2 * gamma * torch::sqrt(h_t) * z_t);
    }
    
    torch::Tensor generate_returns(const torch::Tensor& h_t, const torch::Tensor& z_t, 
                                  double r = 0.0, double d = 0.0) override {
        // R_{t+1} = r - d + λh_{t+1} + √h_{t+1} z_{t+1}, where λ=params
        return r - d + lamda * h_t + torch::sqrt(h_t) * z_t;
    }

    torch::Tensor generate_shock(const torch::Tensor& h_t, const torch::Tensor& y_t, 
                                  double r = 0.0, double d = 0.0) override {    
        return (y_t - r + d - lamda * h_t) / torch::sqrt(h_t);
    }
    
    // Unified parameter management interface implementation
    std::vector<torch::Tensor> get_params() override {
        return {omega, alpha, phi, lamda, gamma};
    }
    
    void set_params(std::vector<torch::Tensor> params, bool learnable = false) override {
        TORCH_CHECK(torch::stack(params).numel() == torch::stack(get_params()).numel(), "Number of parameters supplied inconsistent with current parameters!")

        omega.set_data(params.at(0));
        alpha.set_data(params.at(1));
        phi.set_data(params.at(2));
        lamda.set_data(params.at(3));
        gamma.set_data(params.at(4));

        if (learnable){
            omega.set_requires_grad(learnable);
            alpha.set_requires_grad(learnable);
            phi.set_requires_grad(learnable);
            lamda.set_requires_grad(learnable);
            gamma.set_requires_grad(learnable);
        }
    }
    
    std::string get_model_name() const override {
        return "HN";
    }
    
    void initialize_from_config(const std::map<std::string, double>& params) override {
        if (params.count("omega")) omega = torch::tensor(params.at("omega"));
        if (params.count("alpha")) alpha = torch::tensor(params.at("alpha"));
        if (params.count("phi")) phi = torch::tensor(params.at("phi"));
        if (params.count("lamda")) lamda = torch::tensor(params.at("lamda"));
        if (params.count("gamma")) gamma = torch::tensor(params.at("gamma"));
    }
    
    // Model serialization interface implementation
    std::map<std::string, torch::Tensor> get_state_dict() const override {
        return {
            {"omega", omega},
            {"alpha", alpha},
            {"phi", phi},
            {"lamda", lamda},
            {"gamma", gamma}
        };
    }
    
    void load_state_dict(const std::map<std::string, torch::Tensor>& state_dict) override {
        if (state_dict.count("omega")) omega = state_dict.at("omega");
        if (state_dict.count("alpha")) alpha = state_dict.at("alpha");
        if (state_dict.count("phi")) phi = state_dict.at("phi");
        if (state_dict.count("lamda")) lamda = state_dict.at("lamda");
        if (state_dict.count("gamma")) gamma = state_dict.at("gamma");
    }
};

/**
 * @brief Component Heston-Nandi GARCH model
 */
class CHNModel : public GARCHModel {
public:
    torch::Tensor omega, alpha, phi, lamda, gamma1, gamma2, vphi, rho;
    torch::Tensor q_t; // Long-run component state
    
    CHNModel(double omega_val, double alpha_val, double phi_val, 
                             double lamda_val, double gamma1_val, double gamma2_val,
                             double vphi_val, double rho_val,
                             const torch::TensorOptions& options = torch::kFloat32)
        : omega(register_parameter("omega",torch::tensor(omega_val, options)))
        , alpha(register_parameter("alpha",torch::tensor(alpha_val, options)))
        , phi(register_parameter("phi",torch::tensor(phi_val, options)))
        , lamda(register_parameter("lamda",torch::tensor(lamda_val, options)))
        , gamma1(register_parameter("gamma1",torch::tensor(gamma1_val, options)))
        , gamma2(register_parameter("gamma2",torch::tensor(gamma2_val, options)))
        , vphi(register_parameter("vphi",torch::tensor(vphi_val, options)))
        , rho(register_parameter("rho",torch::tensor(rho_val, options))) {}
    
    std::pair<torch::Tensor, torch::Tensor> update_variance_components(
        const torch::Tensor& h_t, const torch::Tensor& qt_1, const torch::Tensor& z_t) {
        // q_{t+1} = ω̄ + φ(h_t - ω̄) + α(z_t² - 1 - 2γ₁√h_t z_t)
        // h_{t+1} = q_t + φ(h_t - q_t) + α(z_t² - 1 - 2γ₂√h_t z_t)

        torch::Tensor qt = omega + rho * (qt_1 - omega) + vphi * (z_t.pow(2) - 2.0 * h_t.pow(0.5) * z_t * gamma2 - 1.0);
        torch::Tensor ht = qt + phi * (h_t - qt_1) + alpha * (z_t.pow(2) - 2.0 * h_t.pow(0.5) * gamma1 * z_t - 1.0);
        
        return std::make_pair(q_t, h_t);
    }
    
    torch::Tensor update_variance(const torch::Tensor& h_t, const torch::Tensor& z_t) override {

        auto [ht, qt] = update_variance_components(h_t, q_t, z_t);
        q_t = qt; // Update internal state

        return ht;
    }
    
    torch::Tensor generate_returns(const torch::Tensor& h_t, const torch::Tensor& z_t, 
                                  double r = 0.0, double d = 0.0) override {
        return r - d + lamda * h_t + torch::sqrt(h_t) * z_t;
    }

    torch::Tensor generate_shock(const torch::Tensor& h_t, const torch::Tensor& y_t, 
                                  double r = 0.0, double d = 0.0) override {    
        return (y_t - r + d - lamda * h_t) / torch::sqrt(h_t);
    }
    
    // Unified parameter management interface implementation
    std::vector<torch::Tensor> get_params() override {
        return {omega, alpha, phi, lamda, gamma1, gamma2, vphi, rho};
    }
    
    void set_params(std::vector<torch::Tensor> params, bool learnable = false) override {
        TORCH_CHECK(torch::stack(params).numel() == torch::stack(get_params()).numel(), "Number of parameters supplied inconsistent with current parameters!")

        omega.set_data(params.at(0));
        alpha.set_data(params.at(1));
        phi.set_data(params.at(2));
        lamda.set_data(params.at(3));
        gamma1.set_data(params.at(4));
        gamma2.set_data(params.at(5));
        vphi.set_data(params.at(6));
        rho.set_data(params.at(7));

        if (learnable) {
            omega.set_requires_grad(learnable);
            alpha.set_requires_grad(learnable);
            phi.set_requires_grad(learnable);
            lamda.set_requires_grad(learnable);
            gamma1.set_requires_grad(learnable);
            gamma2.set_requires_grad(learnable);
            vphi.set_requires_grad(learnable);
            rho.set_requires_grad(learnable);
        }
    }
    
    std::string get_model_name() const override {
        return "CHN";
    }
    
    void initialize_from_config(const std::map<std::string, double>& params) override {
        if (params.count("omega")) omega = torch::tensor(params.at("omega"));
        if (params.count("alpha")) alpha = torch::tensor(params.at("alpha"));
        if (params.count("phi")) phi = torch::tensor(params.at("phi"));
        if (params.count("lamda")) lamda = torch::tensor(params.at("lamda"));
        if (params.count("gamma1")) gamma1 = torch::tensor(params.at("gamma1"));
        if (params.count("gamma2")) gamma2 = torch::tensor(params.at("gamma2"));
        if (params.count("vphi")) vphi = torch::tensor(params.at("vphi"));
        if (params.count("rho")) rho = torch::tensor(params.at("rho"));
    }
    
    // Model serialization interface implementation
    std::map<std::string, torch::Tensor> get_state_dict() const override {
        return {
            {"omega", omega},
            {"alpha", alpha},
            {"phi", phi},
            {"lamda", lamda},
            {"gamma1", gamma1},
            {"gamma2", gamma2},
            {"vphi", phi},
            {"rho", rho},
        };
    }
    
    void load_state_dict(const std::map<std::string, torch::Tensor>& state_dict) override {
        if (state_dict.count("omega")) omega = state_dict.at("omega");
        if (state_dict.count("alpha")) alpha = state_dict.at("alpha");
        if (state_dict.count("phi")) phi = state_dict.at("phi");
        if (state_dict.count("lamda")) lamda = state_dict.at("lamda");
        if (state_dict.count("gamma1")) gamma1 = state_dict.at("gamma1");
        if (state_dict.count("gamma2")) gamma2 = state_dict.at("gamma2");
        if (state_dict.count("vphi")) vphi = state_dict.at("vphi");
        if (state_dict.count("rho")) rho = state_dict.at("rho");
        if (state_dict.count("q_t")) q_t = state_dict.at("q_t");
    }
};
}  // namespace garch_type_model

