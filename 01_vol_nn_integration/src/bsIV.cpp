#include <cmath>
#include <stdexcept>
#include <vector>
#include "bsIV.h"


// constructor
ImpliedVolatility::ImpliedVolatility(std::shared_ptr<DataFrame> options, torch::Tensor limit){
    setData(options, limit);
    m_ImpVol = getImpliedVolatility();    // final output
}

torch::Tensor ImpliedVolatility::getData() const{
    return m_ImpVol;
}

void ImpliedVolatility::setData(std::shared_ptr<DataFrame> options, torch::Tensor limit){

    m_S = options->get_col("StockPrice");
    m_K = options->get_col("Strike");
    m_RATE = options->get_col("RiskFree");
    m_T = (options->get_col("bDTM"))/252.0;
    m_YIELD = options->get_col("Dividends");
    m_CLASS = options->get_col("isCall");
    m_INITGUESS = options->get_col("ImpliedVolOM");
    m_VALUE = options->get_col("ModelPrice");
    m_LIMIT = limit;

}

torch::Tensor ImpliedVolatility::min(torch::Tensor a, torch::Tensor b) {
    return torch::min(a, b);
}

torch::Tensor ImpliedVolatility::max(torch::Tensor a, torch::Tensor b) {
    return torch::max(a, b);
}

torch::Tensor ImpliedVolatility::normpdf(torch::Tensor X) {
    return 1 / torch::sqrt(2 * m_PI) * torch::exp(-X * X / 2);
}

torch::Tensor ImpliedVolatility::normcdf(const torch::Tensor& X) {
    auto device = torch::kMPS; // Adjust the device as necessary for your configuration

    // Handling edge cases in a vectorized manner
    auto lower_bound_mask = X < -8.0;
    auto upper_bound_mask = X > 8.0;
    auto central_mask = ~(lower_bound_mask | upper_bound_mask);

    // Initialize output tensor
    torch::Tensor result = torch::empty_like(X).to(device);

    // Apply zeros and ones for the edge cases
    result = torch::where(lower_bound_mask, torch::zeros_like(X), result);
    result = torch::where(upper_bound_mask, torch::ones_like(X), result);

    // Handle central values
    if (central_mask.any().item<bool>()) {
        torch::Tensor s = torch::zeros_like(X);
        s = torch::where(central_mask, X, s);
        torch::Tensor t = torch::zeros_like(s);
        torch::Tensor b = s.clone();
        torch::Tensor q = s * s;
        torch::Tensor i = torch::ones_like(s);

        while (torch::any(s != t).item<bool>()) {
            t = s;
            b = b * (q / (i += 2));
            s = t + b;
        }

	    torch::Tensor central_result = 0.5 + s * torch::exp(-0.5 * q - 0.91893853320467274178);
        result = torch::where(central_mask, central_result, result);
    }

    return result;
}

torch::Tensor ImpliedVolatility::bsmPrice(const torch::Tensor& S,
                                          const torch::Tensor& K,
                                          const torch::Tensor& SIGMA,
                                          const torch::Tensor& T,
                                          const torch::Tensor& ISCALL) {

    // Calculating d1 and d2 for the Black-Scholes formula
    torch::Tensor d1 = (torch::log(S / K) + (SIGMA * SIGMA / 2.0) * T) / (SIGMA * torch::sqrt(T));
    torch::Tensor d2 = d1 - SIGMA * torch::sqrt(T);

    // Calculate call and put prices using normcdf
    torch::Tensor call_prices = S * normcdf(d1) - K * normcdf(d2);
    torch::Tensor put_prices = K * normcdf(-d2) - S * normcdf(-d1);

    // Select call or put prices based on ISCALL
    return torch::where(ISCALL.toType(torch::kBool), call_prices, put_prices);
}

torch::Tensor ImpliedVolatility::getImpliedVolatility() {
    int64_t Nb = m_S.size(0);  // Number of elements
    int max_iterations = 500;

    // Expand initial guesses and limits to match the batch size
    torch::Tensor ImpVol = m_INITGUESS.clone();
    torch::Tensor min_b = torch::zeros({Nb}, m_S.options());
    torch::Tensor max_b = m_LIMIT.expand_as(min_b);

    for (int j = 0; j < max_iterations; ++j) {
        
        auto Value_tmp = bsmPrice(m_S * torch::exp(-m_T * m_YIELD),
                                           m_K * torch::exp(-m_T * m_RATE),
                                           ImpVol, m_T, m_CLASS);

        auto price_diff = Value_tmp - m_VALUE;
        auto is_positive = price_diff > 0;

        // Update bounds without in-place operations
        max_b = torch::where(is_positive, ImpVol, max_b);
        min_b = torch::where(is_positive, min_b, ImpVol);

        // Update current guess
        ImpVol = (min_b + max_b) * 0.5;

        // Check for convergence
        if (torch::all(torch::abs(price_diff) <= m_TOLERANCE).item<bool>()) {
            break;
        }

    }

    // Handle cases where the implied volatility exceeds limits without in-place operations
    ImpVol = torch::where(ImpVol > m_LIMIT, m_LIMIT, ImpVol);

    return ImpVol;
}

