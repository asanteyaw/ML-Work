#include "dataframe.h"
#include <torch/torch.h>
#include <iostream>

using namespace pluss::table;

int main(){

  auto mps_available = torch::mps::is_available();
  torch::Device device(mps_available ? torch::kMPS : torch::kCPU);
  std::cout << (mps_available ? "MPS available. Training on GPU." : "Training on CPU.") << '\n';

  
  std::cout << "Done! \n";

  return 0;
}