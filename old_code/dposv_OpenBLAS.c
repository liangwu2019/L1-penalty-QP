#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cblas.h>
#include <lapacke.h>

const int ione = 1;
const double fone = 1.0;
const double fzero = 0.0;

void dposv_OpenBLAS(double *D, double *V, double *w, int m, int n, double *u)
{
    int i, info;
    double *M = (double*)aligned_alloc(64,n*n*sizeof(double));
    // calculate M
    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,n,n,m,fone,V,n,V,n,fzero,M,n);
    for(i=0;i<n;i++)
        M[i*n+i] += D[i];
    memcpy(u,w,sizeof(double)*n);
    info = LAPACKE_dposv(LAPACK_COL_MAJOR,'L',n,ione,M,n,u,n);
    free(M);
    return;
}