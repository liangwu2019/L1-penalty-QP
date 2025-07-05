#include <time.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>
void vecLDL_AVX_OpenMP(double *D, double *V, double *w, int m, int n, double *u, double *run_time)
{
    int i, j, k;
    struct timespec t_start, t_end;
    double *B=(double*)aligned_alloc(64,n*m*sizeof(double));
    double *M=(double*)aligned_alloc(64,m*m*sizeof(double));
    double *lambda=(double*)aligned_alloc(64,n*sizeof(double));
    double *lambda_inverse=(double*)aligned_alloc(64,n*sizeof(double));
    double *q=(double*)aligned_alloc(64,m*sizeof(double));
    double *V_row=(double*)aligned_alloc(64,m*n*sizeof(double));

    __m256d regi_res, regi_v1, regi_v2, regi_v3, regi_v4;
    double host_res[4];

    clock_gettime(CLOCK_MONOTONIC, &t_start);
    for(i=0;i<n;i++)
        for(j=0;j<m;j++)
            V_row[i*m+j] = V[j*n+i];
    for(i=0;i<n;i++)
        lambda[i] = D[i*n+i];
    
    for(i=0;i<m;i++)
    {
        for(j=0;j<m;j++)
        {
            if(i==j)
                M[i*m+j] = 1.0;
            else
                M[i*m+j] = 0.0;
        }           
    }
    // start vector-form LDL factorization
    for(i=0;i<n;i++)
    {
         // update q
        #pragma omp parallel for num_threads(10)
        for(j=0;j<m;j++)
        {
            regi_res = _mm256_setzero_pd();
            for(k=0;k<=m-4;k+=4)
            {
                regi_v1 = _mm256_loadu_pd(&M[j*m+k]);
                regi_v2 = _mm256_loadu_pd(&V_row[i*m+k]);
                regi_res= _mm256_add_pd(regi_res, _mm256_mul_pd(regi_v1,regi_v2));
            }
            _mm256_storeu_pd(host_res, regi_res);
            q[j] = host_res[0] + host_res[1] + host_res[2] + host_res[3];
            for(;k<m;k++)
                q[j] += M[j*m+k]*V_row[i*m+k];
        }
        
        // update lambda
        regi_res = _mm256_setzero_pd();
        for(j=0;j<=m-4;j+=4)
        {
            regi_v1 = _mm256_loadu_pd(&V_row[i*m+j]);   
            regi_v2 = _mm256_loadu_pd(&q[j]);
            regi_res = _mm256_add_pd(regi_res, _mm256_mul_pd(regi_v1,regi_v2));
        }
        _mm256_storeu_pd(host_res, regi_res);
        lambda[i] += (host_res[0] + host_res[1] + host_res[2] + host_res[3]);
        for(;j<m;j++)
            lambda[i] += V_row[i*m+j]*q[j];
        
        // update B
        lambda_inverse[i] = 1.0/lambda[i];
         for(j=0;j<m;j++)
            B[i*m+j] = q[j]*lambda_inverse[i];
        
        // update M
        regi_v1 = _mm256_set1_pd(lambda[i]);
        #pragma omp parallel for num_threads(10)
        for(j=0;j<m;j++)
        {
            regi_v2 = _mm256_set1_pd(B[i*m+j]);
            for(k=0;k<=m-4;k+=4)
            {
                regi_v3 = _mm256_loadu_pd(&B[i*m+k]);
                regi_v4 = _mm256_loadu_pd(&M[j*m+k]);
                regi_res = _mm256_sub_pd(regi_v4, _mm256_mul_pd(_mm256_mul_pd(regi_v3, regi_v2), regi_v1));
                _mm256_storeu_pd(&M[j*m+k],regi_res);
            }
            for(;k<m;k++)
                M[j*m+k] -= lambda[i] * B[i*m+j] * B[i*m+k];
        }
    }
    
    // start forward substitution and diagonal substitution
    for(i=0;i<m;i++)
        q[i] = 0.0;
    for(i=0;i<n;i++)
    {
        regi_res = _mm256_setzero_pd();
        for(j=0;j<=m-4;j+=4)
        {
            regi_v1 = _mm256_loadu_pd(&V_row[i*m+j]);
            regi_v2 = _mm256_loadu_pd(&q[j]);
            regi_res = _mm256_add_pd(regi_res, _mm256_mul_pd(regi_v1, regi_v2));
        }
        _mm256_storeu_pd(host_res, regi_res);
        u[i] = w[i] - (host_res[0] + host_res[1] + host_res[2] + host_res[3]);
        for(;j<m;j++)
            u[i] -= V_row[i*m+j]*q[j];
        for(j=0;j<m;j++)
            q[j] += u[i] * B[i*m+j];
        u[i] *= lambda_inverse[i]; // diagonal substitution
    }        
    
    // start backward substitution
    for(i=0;i<m;i++)
        q[i] = 0.0;
    for(i=n-1;i>=0;i--)
    {
        regi_res = _mm256_setzero_pd();
        for(j=0;j<=m-4;j+=4)
        {
            regi_v1 = _mm256_loadu_pd(&B[i*m+j]);
            regi_v2 = _mm256_loadu_pd(&q[j]);
            regi_res = _mm256_add_pd(regi_res, _mm256_mul_pd(regi_v1, regi_v2));
        }
        _mm256_storeu_pd(host_res, regi_res);
        u[i] = u[i] - (host_res[0] + host_res[1] + host_res[2] + host_res[3]);
        for(;j<m;j++)
            u[i] -= B[i*m+j]*q[j];
        for(j=0;j<m;j++)
            q[j] += u[i] * V_row[i*m+j];
    }
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    *run_time = (t_end.tv_sec  - t_start.tv_sec) + (t_end.tv_nsec-t_start.tv_nsec)/1.0e9;
    free(B); free(M); free(lambda); free(q); free(V_row); free(lambda_inverse);
}
