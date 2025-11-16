#pragma once // prevent multiple inclusions of header

#include <torch/torch.h>
#include "dataframe.h"

using namespace pluss::table;

class ImpliedVolatility{
   public:
      
      ImpliedVolatility(std::shared_ptr<DataFrame> options, torch::Tensor limit);
      torch::Tensor getData() const;
      void setData(std::shared_ptr<DataFrame> options, torch::Tensor limit);
      torch::Tensor getImpliedVolatility();

   private:
      // data members
      torch::Tensor m_PI{torch::tensor(3.14159265358979323846264338327).to(torch::kMPS)};
      torch::Tensor m_TOLERANCE{torch::tensor(0.0000001).to(torch::kMPS)};
      torch::Tensor m_LIMIT{};
      torch::Tensor m_S{}, m_K{}, m_RATE{}, m_T{}, m_VALUE{}, m_YIELD{}, m_CLASS{}, m_INITGUESS{}, m_ImpVol{}; 

      // helper member functions
      torch::Tensor bsmPrice(const torch::Tensor& S, 
                             const torch::Tensor& K, 
                             const torch::Tensor& SIGMA, 
                             const torch::Tensor& T, 
                             const torch::Tensor& ISCALL);
      torch::Tensor normcdf(const torch::Tensor& X);
      torch::Tensor normpdf(torch::Tensor X);
      torch::Tensor max(torch::Tensor a, torch::Tensor b);
      torch::Tensor min(torch::Tensor a, torch::Tensor b);
};

