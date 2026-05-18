#!/bin/bash

# ./run_ft_transformer_sweep.sh --presim presim
# ./run_gbdt_sweep.sh --presim presim
# ./run_ridge_sweep.sh --presim presim
# ./run_mlp_sweep.sh --presim presim
# ./run_rulefit_sweep.sh --presim presim
# mv ../../output ../../output_presim

# ./run_ft_transformer_sweep.sh --presim presim_no_addr_data
# ./run_gbdt_sweep.sh --presim presim_no_addr_data
# ./run_ridge_sweep.sh --presim presim_no_addr_data
# ./run_mlp_sweep.sh --presim presim_no_addr_data
# ./run_rulefit_sweep.sh --presim presim_no_addr_data
# mv ../../output ../../output_presim_no_addr_data

# ./run_ft_transformer_sweep.sh --presim presim_large
# ./run_gbdt_sweep.sh --presim presim_large
# ./run_ridge_sweep.sh --presim presim_large
# ./run_mlp_sweep.sh --presim presim_large
# ./run_rulefit_sweep.sh --presim presim_large
# mv ../../output ../../output_presim_large

./run_ft_transformer_sweep.sh --presim presim_large_no_addr_data
./run_gbdt_sweep.sh --presim presim_large_no_addr_data
./run_ridge_sweep.sh --presim presim_large_no_addr_data
./run_mlp_sweep.sh --presim presim_large_no_addr_data
./run_rulefit_sweep.sh --presim presim_large_no_addr_data
mv ../../output ../../output_presim_large_no_addr_data
