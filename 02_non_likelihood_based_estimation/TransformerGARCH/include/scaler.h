#include <torch/torch.h>
#include <iostream>

class DataScaler {
private:
    torch::Tensor data_min;
    torch::Tensor data_max;
    torch::Tensor data_mean;
    torch::Tensor data_std;
    torch::Tensor data_median;
    torch::Tensor data_iqr;
    float min_range;
    float max_range;

public:
    DataScaler(float min_range = 0.0, float max_range = 1.0) : min_range(min_range), max_range(max_range) {}

    // Min-Max Scaling
    torch::Tensor min_max_scale(const torch::Tensor& data) {
        data_min = data.min(0, true).values;
        data_max = data.max(0, true).values;
        return (data - data_min) / (data_max - data_min) * (max_range - min_range) + min_range;
    }

    // Inverse Min-Max Scaling
    torch::Tensor inverse_min_max_scale(const torch::Tensor& scaled_data) {
        return (scaled_data - min_range) / (max_range - min_range) * (data_max - data_min) + data_min;
    }

    // Standard Scaling
    torch::Tensor standard_scale(const torch::Tensor& data) {
        data_mean = data.mean(0, true);
        data_std = data.std(0, true);
        return (data - data_mean) / data_std;
    }

    // Inverse Standard Scaling
    torch::Tensor inverse_standard_scale(const torch::Tensor& scaled_data) {
        return scaled_data * data_std + data_mean;
    }

    // Robust Scaling
    torch::Tensor robust_scale(const torch::Tensor& data) {
        data_median = torch::median(data, 0, true).values;
        torch::Tensor q1 = torch::quantile(data, 0.25, 0, true);
        torch::Tensor q3 = torch::quantile(data, 0.75, 0, true);
        data_iqr = q3 - q1;
        return (data - data_median) / data_iqr;
    }

    // Inverse Robust Scaling
    torch::Tensor inverse_robust_scale(const torch::Tensor& scaled_data) {
        return scaled_data * data_iqr + data_median;
    }
};
