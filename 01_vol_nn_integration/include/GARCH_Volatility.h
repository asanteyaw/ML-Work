#pragma once

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
class GARCHModel {
public:
    virtual ~GARCHModel() = default;
    
    // Core GARCH functionality
    virtual torch::Tensor update_variance(const torch::Tensor& h_t, const torch::Tensor& z_t, torch::Tensor params) = 0;
    virtual torch::Tensor generate_returns(const torch::Tensor& h_t, const torch::Tensor& z_t, torch::Tensor params, 
                                          double r = 0.0, double d = 0.0) = 0;
    virtual torch::Tensor generate_shock(const torch::Tensor& h_t, const torch::Tensor& y_t, torch::Tensor params,
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
    torch::Tensor omega_bar, alpha, phi, lamda, gamma;
    
    HNModel(double omega_bar_val, double alpha_val, double phi_val, 
                     double lambda_val, double gamma_val,
                     const torch::TensorOptions& options = torch::kFloat32)
        : omega_bar(torch::tensor(omega_bar_val, options))
        , alpha(torch::tensor(alpha_val, options))
        , phi(torch::tensor(phi_val, options))
        , lamda(torch::tensor(lambda_val, options))
        , gamma(torch::tensor(gamma_val, options)) {}
    
    torch::Tensor update_variance(const torch::Tensor& h_t, const torch::Tensor& z_t, torch::Tensor params) override {
        // h_{t+1} = ω̄ + φ(h_t - ω̄) + α(z_t² - 1 - 2γ√h_t z_t)
        auto omega_s = params[0];
        auto alpha_s = params[1];
        auto phi_s = params[2];
        auto lamda_s = params[3];
        auto gamma_s = params[4];
        
        return omega_s + phi_s * (h_t - omega_s) + 
               alpha_s * (z_t.pow(2) - 1 - 2 * gamma_s * torch::sqrt(h_t) * z_t);
    }
    
    torch::Tensor generate_returns(const torch::Tensor& h_t, const torch::Tensor& z_t, torch::Tensor params, 
                                  double r = 0.0, double d = 0.0) override {
        // R_{t+1} = r - d + λh_{t+1} + √h_{t+1} z_{t+1}, where λ=params
        return r - d + params * h_t + torch::sqrt(h_t) * z_t;
    }

    torch::Tensor generate_shock(const torch::Tensor& h_t, const torch::Tensor& y_t, torch::Tensor params, 
                                  double r = 0.0, double d = 0.0) override {    
        return (y_t - r + d - params * h_t) / torch::sqrt(h_t);
    }
    
    // Unified parameter management interface implementation
    std::vector<torch::Tensor> get_params() override {
        return {omega_bar, alpha, phi, lamda, gamma};
    }
    
    void set_params(std::vector<torch::Tensor> params, bool learnable = true) override {
        omega_bar.set_data(params[0]);
        alpha.set_data(params[1]);
        phi.set_data(params[2]);
        lamda.set_data(params[3]);
        gamma.set_data(params[4]);
        
        if (learnable){
            omega_bar.set_requires_grad(learnable);
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
        if (params.count("omega_bar")) omega_bar = torch::tensor(params.at("omega_bar"));
        if (params.count("alpha")) alpha = torch::tensor(params.at("alpha"));
        if (params.count("phi")) phi = torch::tensor(params.at("phi"));
        if (params.count("lamda")) lamda = torch::tensor(params.at("lamda"));
        if (params.count("gamma")) gamma = torch::tensor(params.at("gamma"));
    }
    
    // Model serialization interface implementation
    std::map<std::string, torch::Tensor> get_state_dict() const override {
        return {
            {"omega_bar", omega_bar},
            {"alpha", alpha},
            {"phi", phi},
            {"lamda", lamda},
            {"gamma", gamma}
        };
    }
    
    void load_state_dict(const std::map<std::string, torch::Tensor>& state_dict) override {
        if (state_dict.count("omega_bar")) omega_bar = state_dict.at("omega_bar");
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
    torch::Tensor omega_bar, alpha, phi, lamda, gamma1, gamma2, phi_q, rho;
    torch::Tensor q_t; // Long-run component state
    
    CHNModel(double omega_bar_val, double alpha_val, double phi_val, 
                             double lamda_val, double gamma1_val, double gamma2_val,
                             double phi_q_val, double rho_val,
                             const torch::TensorOptions& options = torch::kFloat32)
        : omega_bar(torch::tensor(omega_bar_val, options))
        , alpha(torch::tensor(alpha_val, options))
        , phi(torch::tensor(phi_val, options))
        , lamda(torch::tensor(lamda_val, options))
        , gamma1(torch::tensor(gamma1_val, options))
        , gamma2(torch::tensor(gamma2_val, options))
        , phi_q(torch::tensor(phi_q_val, options))
        , rho(torch::tensor(rho_val, options))
        , q_t(torch::tensor(omega_bar_val, options)) {}
    
    std::pair<torch::Tensor, torch::Tensor> update_variance_components(
        const torch::Tensor& h_t, const torch::Tensor& q_t_in, const torch::Tensor& z_t) {
        
        auto q_next = omega_bar + rho * (q_t_in - omega_bar) + 
                     phi_q * (z_t.pow(2) - 1 - 2 * gamma2 * torch::sqrt(h_t) * z_t);
        
        auto h_next = q_next + phi * (h_t - q_t_in) + 
                alpha * (z_t.pow(2) - 1 - 2 * gamma1 * torch::sqrt(h_t) * z_t);
        
        return std::make_pair(h_next, q_next);
    }
    
    torch::Tensor update_variance(const torch::Tensor& h_t, const torch::Tensor& z_t, torch::Tensor params) override {
        auto [h_next, q_next] = update_variance_components(h_t, q_t, z_t);
        q_t = q_next; // Update internal state
        return h_next;
    }
    
    torch::Tensor generate_returns(const torch::Tensor& h_t, const torch::Tensor& z_t, torch::Tensor params, 
                                  double r = 0.0, double d = 0.0) override {
        return r - d + lamda * h_t + torch::sqrt(h_t) * z_t;
    }

    torch::Tensor generate_shock(const torch::Tensor& h_t, const torch::Tensor& y_t, torch::Tensor params, 
                                  double r = 0.0, double d = 0.0) override {    
        return (y_t - r + d - lamda * h_t) / torch::sqrt(h_t);
    }
    
    // Unified parameter management interface implementation
    std::vector<torch::Tensor> get_params() override {
        return {omega_bar, alpha, phi, lamda, gamma1, gamma2, phi_q, rho};
    }
    
    void set_params(std::vector<torch::Tensor> params, bool learnable = true) override {
        omega_bar.set_requires_grad(learnable);
        alpha.set_requires_grad(learnable);
        phi.set_requires_grad(learnable);
        lamda.set_requires_grad(learnable);
        gamma1.set_requires_grad(learnable);
        gamma2.set_requires_grad(learnable);
        phi_q.set_requires_grad(learnable);
        rho.set_requires_grad(learnable);
    }
    
    std::string get_model_name() const override {
        return "CHN";
    }
    
    void initialize_from_config(const std::map<std::string, double>& params) override {
        if (params.count("omega_bar")) omega_bar = torch::tensor(params.at("omega_bar"));
        if (params.count("alpha")) alpha = torch::tensor(params.at("alpha"));
        if (params.count("phi")) phi = torch::tensor(params.at("phi"));
        if (params.count("lamda")) lamda = torch::tensor(params.at("lamda"));
        if (params.count("gamma1")) gamma1 = torch::tensor(params.at("gamma1"));
        if (params.count("gamma2")) gamma2 = torch::tensor(params.at("gamma2"));
        if (params.count("phi_q")) phi_q = torch::tensor(params.at("phi_q"));
        if (params.count("rho")) rho = torch::tensor(params.at("rho"));
    }
    
    // Model serialization interface implementation
    std::map<std::string, torch::Tensor> get_state_dict() const override {
        return {
            {"omega_bar", omega_bar},
            {"alpha", alpha},
            {"phi", phi},
            {"lamda", lamda},
            {"gamma1", gamma1},
            {"gamma2", gamma2},
            {"phi_q", phi_q},
            {"rho", rho},
            {"q_t", q_t}
        };
    }
    
    void load_state_dict(const std::map<std::string, torch::Tensor>& state_dict) override {
        if (state_dict.count("omega_bar")) omega_bar = state_dict.at("omega_bar");
        if (state_dict.count("alpha")) alpha = state_dict.at("alpha");
        if (state_dict.count("phi")) phi = state_dict.at("phi");
        if (state_dict.count("lamda")) lamda = state_dict.at("lamda");
        if (state_dict.count("gamma1")) gamma1 = state_dict.at("gamma1");
        if (state_dict.count("gamma2")) gamma2 = state_dict.at("gamma2");
        if (state_dict.count("phi_q")) phi_q = state_dict.at("phi_q");
        if (state_dict.count("rho")) rho = state_dict.at("rho");
        if (state_dict.count("q_t")) q_t = state_dict.at("q_t");
    }
};
}  // namespace garch_type_model

