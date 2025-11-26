#pragma once
#include <torch/torch.h>

struct SimResult {
    torch::Tensor R;   // [paths, steps]
    torch::Tensor h;   // [paths, steps+1]
    torch::Tensor S;   // [paths, steps+1]
};

//
// HESTON-NANDI SIMULATOR
//
inline SimResult simulate_HN(
    double S0,
    double r,
    double d,
    double omega_bar,
    double phi,
    double alpha,
    double gamma,
    double lambda,
    double h0,
    int64_t n_paths,
    int64_t n_steps,
    torch::Device device = torch::kCPU)
{
    using namespace torch::indexing;
    auto opts = torch::TensorOptions().dtype(torch::kDouble).device(device);

    auto R = torch::empty({n_paths, n_steps}, opts);
    auto h = torch::empty({n_paths, n_steps + 1}, opts);
    auto S = torch::empty({n_paths, n_steps + 1}, opts);

    h.index_put_({Slice(), 0}, torch::full({n_paths}, h0, opts));
    S.index_put_({Slice(), 0}, torch::full({n_paths}, S0, opts));

    torch::Tensor h_t = h.index({Slice(), 0});
    torch::Tensor S_t = S.index({Slice(), 0});

    const double mu = r - d;

    for (int64_t t = 0; t < n_steps; ++t) {
        auto z = torch::randn({n_paths}, opts);
        auto sqrt_h_t = h_t.sqrt();

        auto h_next =
            omega_bar
          + phi * (h_t - omega_bar)
          + alpha * (z.pow(2) - 1.0 - 2.0 * gamma * sqrt_h_t * z);

        h_next = torch::clamp(h_next, 1e-12);

        auto sqrt_h_next = h_next.sqrt();

        auto R_next = mu + lambda * h_next + sqrt_h_next * z;
        auto S_next = S_t * torch::exp(R_next);

        R.index_put_({Slice(), t}, R_next);
        h.index_put_({Slice(), t+1}, h_next);
        S.index_put_({Slice(), t+1}, S_next);

        h_t = h_next;
        S_t = S_next;
    }

    return {R, h, S};
}

//
// COMPONENT HESTON-NANDI SIMULATOR
//
inline SimResult simulate_CHN(
    double S0,
    double r,
    double d,
    double omega_bar,
    double phi,
    double alpha,
    double gamma1,
    double rho,
    double varphi,
    double gamma2,
    double lambda,
    double h0,
    double q0,
    int64_t n_paths,
    int64_t n_steps,
    torch::Device device = torch::kCPU)
{
    using namespace torch::indexing;
    auto opts = torch::TensorOptions().dtype(torch::kDouble).device(device);

    auto R = torch::empty({n_paths, n_steps}, opts);
    auto h = torch::empty({n_paths, n_steps + 1}, opts);
    auto S = torch::empty({n_paths, n_steps + 1}, opts);
    auto q = torch::empty({n_paths, n_steps + 1}, opts);

    h.index_put_({Slice(), 0}, torch::full({n_paths}, h0, opts));
    q.index_put_({Slice(), 0}, torch::full({n_paths}, q0, opts));
    S.index_put_({Slice(), 0}, torch::full({n_paths}, S0, opts));

    torch::Tensor h_t = h.index({Slice(), 0});
    torch::Tensor q_t = q.index({Slice(), 0});
    torch::Tensor S_t = S.index({Slice(), 0});

    const double mu = r - d;

    for (int64_t t = 0; t < n_steps; ++t) {
        auto z = torch::randn({n_paths}, opts);
        auto sqrt_h_t = h_t.sqrt();

        auto q_next =
            omega_bar +
            rho * (q_t - omega_bar) +
            varphi * (z.pow(2) - 1.0 - 2.0 * gamma2 * sqrt_h_t * z);

        q_next = torch::clamp(q_next, 1e-12);

        auto h_next =
            q_next +
            phi * (h_t - q_t) +
            alpha * (z.pow(2) - 1.0 - 2.0 * gamma1 * sqrt_h_t * z);

        h_next = torch::clamp(h_next, 1e-12);

        auto sqrt_h_next = h_next.sqrt();
        auto R_next = mu + lambda * h_next + sqrt_h_next * z;
        auto S_next = S_t * torch::exp(R_next);

        R.index_put_({Slice(), t}, R_next);
        h.index_put_({Slice(), t+1}, h_next);
        q.index_put_({Slice(), t+1}, q_next);
        S.index_put_({Slice(), t+1}, S_next);

        h_t = h_next;
        q_t = q_next;
        S_t = S_next;
    }

    return {R, h, S};
}

//
// DISPATCHER
//
enum class VolModel { HN, CHN };

inline SimResult simulate_vol_model(
    VolModel type,
    const torch::Tensor& params,
    double S0,
    double r, double d,
    int64_t n_paths,
    int64_t n_steps,
    torch::Device device = torch::kCPU)
{
    auto p = params.to(torch::kDouble).cpu();

    if (type == VolModel::HN) {
        TORCH_CHECK(p.size(0) == 6,
            "HN expects [omega_bar, phi, alpha, gamma, lambda, h0]");

        return simulate_HN(
            S0, r, d,
            p[0].item<double>(),
            p[1].item<double>(),
            p[2].item<double>(),
            p[3].item<double>(),
            p[4].item<double>(),
            p[5].item<double>(),
            n_paths, n_steps, device);
    }

    if (type == VolModel::CHN) {
        TORCH_CHECK(p.size(0) == 10,
            "CHN expects [omega_bar, phi, alpha, gamma1, rho, varphi, gamma2, lambda, h0, q0]");

        return simulate_CHN(
            S0, r, d,
            p[0].item<double>(),  // omega_bar
            p[1].item<double>(),  // phi
            p[2].item<double>(),  // alpha
            p[3].item<double>(),  // gamma1
            p[4].item<double>(),  // rho
            p[5].item<double>(),  // varphi
            p[6].item<double>(),  // gamma2
            p[7].item<double>(),  // lambda
            p[8].item<double>(),  // h0
            p[9].item<double>(),  // q0
            n_paths, n_steps, device);
    }

    TORCH_CHECK(false, "Unknown model type");
}