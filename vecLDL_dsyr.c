#include <stdlib.h>
#include <string.h>
#include <cblas.h>

const int ione = 1;
const double fone = 1.0;
const double fzero = 0.0;

void vecLDL_dsyr(double *D, double *V, double *w, int m, int n, double *u)
{
    int i, j, info;
    double *B = (double*)aligned_alloc(64,n*m*sizeof(double));
    double *M = (double*)aligned_alloc(64,m*m*sizeof(double));
    double *lambda = (double*)aligned_alloc(64,n*sizeof(double));
    double *lambda_inverse = (double*)aligned_alloc(64,n*sizeof(double));
    double *q = (double*)aligned_alloc(64,m*sizeof(double));
    double *V_row = (double*)aligned_alloc(64,m*n*sizeof(double));
    double temp;

    for(i=0;i<n;i++)
        for(j=0;j<m;j++)
        V_row[i*m+j] = V[j*n+i];
    memcpy(lambda,D,sizeof(double)*n);
    memset(B,0,sizeof(double)*n*m);
    memset(M,0,sizeof(double)*m*m);
    for(i=0;i<m;i++)
        M[i*m+i] = 1.0;
    /*======== Begin: vector-from LDL factorization =======*/ 
    for(i=0;i<n;i++)
    {
        // compute q
        cblas_dsymv(CblasColMajor,CblasLower,m,fone,M,m,V_row+i*m,ione,fzero,q,ione);
        // update lambda
        temp = cblas_ddot(m,V_row+i*m,ione,q,ione);
        lambda[i] += temp;
        // compute lambda_inverse
        lambda_inverse[i] = 1.0/lambda[i];
        // update B_i
        cblas_dcopy(m,q,ione,B+i*m,ione);
        cblas_dscal(m,lambda_inverse[i],B+i*m,ione);
        // update M
        cblas_dsyr(CblasColMajor,CblasLower,m,-lambda[i],B+i*m,ione,M,m);
    }

    /*======== Begin: froward substitution and diagonal substitution =======*/
    memset(q,0,sizeof(double)*m);
    for(i=0;i<n;i++)
    {
        // compute u[i]
        temp = cblas_ddot(m,V_row+i*m,ione,q,ione);
        u[i] = w[i] - temp;
        // update q
        cblas_daxpy(m,u[i],B+i*m,ione,q,ione);
        // diagonal substitution
        u[i] *= lambda_inverse[i];
    } 

    /*======== Begin: backward substitution =======*/ 
    memset(q,0,sizeof(double)*m);
    for(i=n-1;i>=0;i--)
    {
        // compute u[i]
        temp = cblas_ddot(m,B+i*m,ione,q,ione);
        u[i] = u[i] - temp;
        // update q
        cblas_daxpy(m,u[i],V_row+i*m,ione,q,ione);
    }
    free(B); free(M); free(lambda); free(lambda_inverse); free(q); free(V_row); 
    return;
}