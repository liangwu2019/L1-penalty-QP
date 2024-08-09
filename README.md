# L1-penatly-soft-constrained-QP-solver
L1-penalty soft-constrained QP solver with global feasibility and execution time certificate for real-time MPC applications

# Please Cite
Wu, Liang, and Richard D. Braatz. "A Parallel Vector-form $LDL^\top$ Decomposition for Accelerating Execution-time-certified $\ell_1$-penalty Soft-constrained MPC." arXiv preprint arXiv:2403.18235 (2024).

And test_XXX.ipynb files correspond to three numerical examples of the paper.

# Compile commands:
gcc chol.c -o chol.so -O3 -shared -fPIC
gcc prodLDL.c -o prodLDL.so -O3 -shared -fPIC
gcc vecLDL_AVX_OpenMP.c -o vecLDL_AVX_OpenMP.so -O3 -mavx -fopenmp -shared -fPIC
gcc vecLDL_AVX_Unroll.c -o vecLDL_AVX_Unroll.so -O3 -mavx -shared -fPIC
gcc ETC_L1_QP.c -o ETC_L1_QP.so -O3 -mavx -shard -fPIC


