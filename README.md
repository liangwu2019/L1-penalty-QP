# "A Parallel Vector-form $LDL^\top$ Decomposition for Accelerating Execution-time-certified $\ell_1$-penalty Soft-constrained MPC." arXiv preprint arXiv:2403.18235 (2024).
Consider the linear system: $(VV^\top + D)x=b$, where $V\in\mathbb{R}^{n\times m} (n\geqm)$, the proposed vector-form $LDL^\top$ decomposition outperforms the standard Cholesky decomposition only when $n\geq8m$ (both of their implementations are based on the same OpenBLAS v.0.3.30 library for a fair comparison), see the following figures:
![pipeline]vecLDL_vs_Chol.pdf







