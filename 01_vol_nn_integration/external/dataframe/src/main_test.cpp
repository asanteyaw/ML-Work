#include "dataframe.h"
#include <torch/torch.h>
#include <iostream>

using namespace pluss::table;

int main(){
   auto df = DataFrame::load("csv", "../data/demoSPX.csv");  //out_ops_data19/demoSPX
   auto op_df = DataFrame::read_csv("../data/out_ops_data19.csv");

   df->head(10);
   std::cout << "Displaying options data\n";
   op_df->head(10);
   std::cout << "Displaying info\n";
   df->info();
   std::cout << "Describing the data\n";
   df->describe();

   return 0;
}