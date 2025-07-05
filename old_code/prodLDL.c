#include <time.h>
#include <stdlib.h>
void prodLDL(double *D, double *V, double *w, int m, int n, double *u, double *run_time)
{
    int i, j, k;
    struct timespec t_start, t_end;
    double *P=(double*)aligned_alloc(64, n*m*sizeof(double));
    double *Beta=(double*)aligned_alloc(64,n*m*sizeof(double));
    double *lambda=(double*)aligned_alloc(64,n*sizeof(double));
    double sigma;
    double kappa;

    clock_gettime(CLOCK_MONOTONIC, &t_start);
    for(i=0;i<n;i++)
    {
        lambda[i] = D[i*n+i];
        u[i] = w[i];
    }
    // start product-form LDL factorization
    for(i=0;i<m;i++)
    {
        for(j=0;j<n;j++)
            P[i*n+j] = V[i*n+j];
        for(j=0;j<i;j++)
        {
            sigma = 0;
            for(k=0;k<n;k++)
            {
                P[i*n+k] -= P[j*n+k]*sigma;
                sigma += P[i*n+k]*Beta[j*n+k];
            }
        }
        kappa = 1.0;
        for(j=0;j<n;j++)
        {
            lambda[j] = lambda[j] + P[i*n+j]*kappa*P[i*n+j];
            Beta[i*n+j] = kappa*P[i*n+j]/lambda[j];
            kappa = kappa - lambda[j]*Beta[i*n+j]*Beta[i*n+j];
        }
    }
    // start substitution
    for(i=0;i<m;i++)
    {
        sigma = 0;
        for(j=0;j<n;j++)
        {
            u[j] -= P[i*n+j] * sigma;
            sigma += u[j]*Beta[i*n+j];
        }
    }
    for(i=0;i<n;i++)
        u[i] /= lambda[i];
    for(i=m-1;i>=0;i--)
    {
        sigma = 0;
        for(j=n-1;j>=0;j--)
        {
            u[j] -= Beta[i*n+j]*sigma;
            sigma += u[j]*P[i*n+j];
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    *run_time = (t_end.tv_sec  - t_start.tv_sec) + (t_end.tv_nsec-t_start.tv_nsec)/1.0e9;
    free(P); free(Beta); free(lambda);
}
