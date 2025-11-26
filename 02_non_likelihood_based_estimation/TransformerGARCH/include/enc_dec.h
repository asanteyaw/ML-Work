#pragma once

#include <torch/torch.h>
#include <iostream>
#include <vector>

// GRU Encoder
struct EncoderImpl : torch::nn::Module {
    // Member Functions
    EncoderImpl(int64_t input_size, 
                   int64_t hidden_size, 
                   int64_t num_layers = 1, 
                   bool batch_first = true) {
        gru = register_module(
            "gru", 
            torch::nn::GRU(torch::nn::GRUOptions(input_size, hidden_size)
                           .num_layers(num_layers)
                           .batch_first(batch_first))
        );
    }

    std::pair<torch::Tensor,torch::Tensor> forward(torch::Tensor x, torch::Tensor hx) {
        // x: (batch_size, seq_len, input_size)
        auto [output, h] = gru->forward(x, hx);
        return {output, h}; // Return the final hidden state
    }

    // Data Members
    torch::nn::GRU gru{nullptr};
};
TORCH_MODULE(Encoder);

struct AttentionImpl : torch::nn::Module {

    AttentionImpl(int64_t hidden_dim)
        : attn(torch::nn::Linear(hidden_dim * 2, hidden_dim)),
          v(torch::randn({hidden_dim}, torch::requires_grad())) {
        register_module("attn", attn);
        register_parameter("v", v);   // v a trainable param
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor hidden, torch::Tensor encoder_outputs) {
        // hidden: [batch_size, hidden_dim]
        // encoder_outputs: [batch_size, seq_len, hidden_dim]
        int64_t seq_len = encoder_outputs.size(1);

        // Expand hidden state to match encoder_outputs' seq_len: [batch_size, seq_len, hidden_dim]
        auto repeated_hidden = hidden.unsqueeze(1).repeat({1, seq_len, 1});

        // Concatenate hidden and encoder_outputs: [batch_size, seq_len, hidden_dim * 2]
        auto combined = torch::cat({repeated_hidden, encoder_outputs}, 2);

        // Transform and compute energy: [batch_size, seq_len, hidden_dim]
        auto energy = torch::tanh(attn(combined));

        // Compute scores: [batch_size, seq_len]
        auto scores = torch::matmul(energy, v);
      
        // Softmax over seq_len: [batch_size, seq_len]
        auto attention_weights = torch::softmax(scores, 1);

        // Compute context vector: [batch_size, hidden_dim]
        auto context = torch::bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1);

        return std::make_tuple(context, attention_weights);
    }

    // Data members
    torch::nn::Linear attn{nullptr};
    torch::Tensor v;  // Learnable parameter
};

TORCH_MODULE(Attention);

// LSTM Decoder
struct DecoderImpl : torch::nn::Module {
    // Member Functions
    DecoderImpl(int64_t input_size, 
                    int64_t hidden_size, 
                    int64_t num_layers = 1, 
                    int64_t output_size = 1,
                    bool batch_first = true) {
        lstm = register_module(
            "lstm", 
            torch::nn::LSTM(torch::nn::LSTMOptions(input_size, hidden_size)
                            .num_layers(num_layers)
                            .batch_first(batch_first))
        );
        fc = register_module("fc", torch::nn::Linear(hidden_size, output_size)); // Use output_size dynamically
    }

    std::pair<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> forward(torch::Tensor y, std::tuple<torch::Tensor, torch::Tensor> h) {
       
       auto [lstm_output, lstm_hidden] = lstm->forward(y, h);
       auto fc_out = fc(lstm_output.squeeze(1));
      
       return std::make_pair(fc_out, lstm_hidden); // (batch_size, seq_len, 1)
    }

    // Data Members
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear fc{nullptr};
};
TORCH_MODULE(Decoder);
