# L1-penatly-soft-constrained-QP-solver
L1-penalty soft-constrained QP solver with global feasibility and execution time certificate for real-time MPC applications

# Compile commands:
gcc chol.c -o chol.so -O3 -shared -fPIC

gcc prodLDL.c -o prodLDL.so -O3 -shared -fPIC

gcc vecLDL_AVX_OpenMP.c -o vecLDL_AVX_OpenMP.so -O3 -mavx -fopenmp -shared -fPIC

gcc vecLDL_AVX_Unroll.c -o vecLDL_AVX_Unroll.so -O3 -mavx -shared -fPIC

gcc ETC_L1_QP.c -o ETC_L1_QP.so -O3 -mavx -shard -fPIC
