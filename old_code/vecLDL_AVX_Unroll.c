#include <time.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>
void vecLDL_AVX_Unroll(double *D, double *V, double *w, int m, int n, double *u, double *run_time)
{
    int i, j, k;
    struct timespec t_start, t_end;
    double *B=(double*)aligned_alloc(64,n*m*sizeof(double));
    double *M=(double*)aligned_alloc(64,m*m*sizeof(double));
    double *lambda=(double*)aligned_alloc(64,n*sizeof(double));
    double *lambda_inverse=(double*)aligned_alloc(64,n*sizeof(double));
    double *q=(double*)aligned_alloc(64,m*sizeof(double));
    double *V_row=(double*)aligned_alloc(64,m*n*sizeof(double));    

    __m256d regi_res1, regi_res2, regi_v1, regi_v2, regi_v3, regi_v4, regi_v5, regi_v6;
    double host_res1[4], host_res2[4];
    
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
        for(j=0;j<=m-2;j+=2) // unroll the row with the size 2
        {
            regi_res1 = _mm256_setzero_pd();
            regi_res2 = _mm256_setzero_pd();
            for(k=0;k<=m-4;k+=4)
            {
                regi_v1 = _mm256_loadu_pd(&M[j*m+k]);
                regi_v2 = _mm256_loadu_pd(&M[j*m+m+k]);
                regi_v3 = _mm256_loadu_pd(&V_row[i*m+k]);
                regi_res1= _mm256_add_pd(regi_res1, _mm256_mul_pd(regi_v1,regi_v3));
                regi_res2= _mm256_add_pd(regi_res2, _mm256_mul_pd(regi_v2,regi_v3));
            }
            _mm256_storeu_pd(host_res1, regi_res1);
            _mm256_storeu_pd(host_res2, regi_res2);
            q[j] = host_res1[0] + host_res1[1] + host_res1[2] + host_res1[3];
            q[j+1] = host_res2[0] + host_res2[1] + host_res2[2] + host_res2[3];
            for(;k<m;k++)
            {
                q[j] += M[j*m+k]*V_row[i*m+k];
                q[j+1] += M[j*m+m+k]*V_row[i*m+k];
            }
        }
        for(;j<m;j++)
        {
            regi_res1 = _mm256_setzero_pd();
            for(k=0;k<=m-4;k+=4)
            {
                regi_v1 = _mm256_loadu_pd(&M[j*m+k]);
                regi_v3 = _mm256_loadu_pd(&V_row[i*m+k]);
                regi_res1= _mm256_add_pd(regi_res1, _mm256_mul_pd(regi_v1,regi_v3));
            }
            _mm256_storeu_pd(host_res1, regi_res1);
            q[j] = host_res1[0] + host_res1[1] + host_res1[2] + host_res1[3];
            for(;k<m;k++)
                q[j] += M[j*m+k]*V_row[i*m+k];
        }

        // update lambda
        regi_res1 = _mm256_setzero_pd();
        for(j=0;j<=m-4;j+=4)
        {
            regi_v1 = _mm256_loadu_pd(&V_row[i*m+j]);   
            regi_v2 = _mm256_loadu_pd(&q[j]);
            regi_res1 = _mm256_add_pd(regi_res1, _mm256_mul_pd(regi_v1,regi_v2));
        }
        _mm256_storeu_pd(host_res1, regi_res1);
        lambda[i] += (host_res1[0] + host_res1[1] + host_res1[2] + host_res1[3]);
        for(;j<m;j++)
            lambda[i] += V_row[i*m+j]*q[j];
        
        // update B
        lambda_inverse[i] = 1.0/lambda[i];
        regi_v1 = _mm256_set1_pd(lambda_inverse[i]);
        for(j=0;j<=m-4;j+=4)
            _mm256_storeu_pd(&B[i*m+j], _mm256_mul_pd(regi_v1, _mm256_loadu_pd(&q[j])));
        for(;j<m;j++)
            B[i*m+j] = q[j]*lambda_inverse[i];

        // update M
        regi_v1 = _mm256_set1_pd(lambda[i]);
        for(j=0;j<=m-2;j+=2)
        {
            regi_v2 = _mm256_set1_pd(B[i*m+j]);
            regi_v3 = _mm256_set1_pd(B[i*m+j+1]);
            for(k=0;k<=m-4;k+=4)
            {
                regi_v4 = _mm256_loadu_pd(&B[i*m+k]);
                regi_v5 = _mm256_loadu_pd(&M[j*m+k]);
                regi_v6 = _mm256_loadu_pd(&M[j*m+m+k]);
                regi_res1 = _mm256_sub_pd(regi_v5, _mm256_mul_pd(_mm256_mul_pd(regi_v4, regi_v2), regi_v1));
                regi_res2 = _mm256_sub_pd(regi_v6, _mm256_mul_pd(_mm256_mul_pd(regi_v4, regi_v3), regi_v1));
                _mm256_storeu_pd(&M[j*m+k],regi_res1);
                _mm256_storeu_pd(&M[j*m+m+k],regi_res2);
            }
            for(;k<m;k++)
            {
                M[j*m+k] -= lambda[i] * B[i*m+j] * B[i*m+k];
                M[j*m+m+k] -= lambda[i] * B[i*m+j+1] * B[i*m+k];
            }
        }
        for(;j<m;j++)
        {
            regi_v2 = _mm256_set1_pd(B[i*m+j]);
            for(k=0;k<=m-4;k+=4)
            {
                regi_v3 = _mm256_loadu_pd(&B[i*m+k]);
                regi_v4 = _mm256_loadu_pd(&M[j*m+k]);
                regi_res1 = _mm256_sub_pd(regi_v4, _mm256_mul_pd(_mm256_mul_pd(regi_v3, regi_v2), regi_v1));
                _mm256_storeu_pd(&M[j*m+k],regi_res1);
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
        regi_res1 = _mm256_setzero_pd();
        for(j=0;j<=m-4;j+=4)
        {
            regi_v1 = _mm256_loadu_pd(&V_row[i*m+j]);
            regi_v2 = _mm256_loadu_pd(&q[j]);
            regi_res1 = _mm256_add_pd(regi_res1, _mm256_mul_pd(regi_v1, regi_v2));
        }
        _mm256_storeu_pd(host_res1, regi_res1);
        u[i] = w[i] - (host_res1[0] + host_res1[1] + host_res1[2] + host_res1[3]);
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
        regi_res1 = _mm256_setzero_pd();
        for(j=0;j<=m-4;j+=4)
        {
            regi_v1 = _mm256_loadu_pd(&B[i*m+j]);
            regi_v2 = _mm256_loadu_pd(&q[j]);
            regi_res1 = _mm256_add_pd(regi_res1, _mm256_mul_pd(regi_v1, regi_v2));
        }
        _mm256_storeu_pd(host_res1, regi_res1);
        u[i] = u[i] - (host_res1[0] + host_res1[1] + host_res1[2] + host_res1[3]);
        for(;j<m;j++)
            u[i] -= B[i*m+j]*q[j];
        for(j=0;j<m;j++)
            q[j] += u[i] * V_row[i*m+j];
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    *run_time = (t_end.tv_sec  - t_start.tv_sec) + (t_end.tv_nsec-t_start.tv_nsec)/1.0e9;
    free(B); free(M); free(lambda); free(lambda_inverse); free(q); free(V_row);
}
