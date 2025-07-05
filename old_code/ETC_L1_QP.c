#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>

void vecLDL(double *V, double *D, double *p, int m, int n, double *delta_z) // This is vecLDL_AVX_Unroll
{
    int i, j, k;
    double *B=(double*)aligned_alloc(64,n*m*sizeof(double));
    double *M=(double*)aligned_alloc(64,m*m*sizeof(double));
    double *lambda=(double*)aligned_alloc(64,n*sizeof(double));
    double *lambda_inverse=(double*)aligned_alloc(64,n*sizeof(double));
    double *q=(double*)aligned_alloc(64,m*sizeof(double));
    double *V_row=(double*)aligned_alloc(64,m*n*sizeof(double));   
    __m256d regi_res1, regi_res2, regi_v1, regi_v2, regi_v3, regi_v4, regi_v5, regi_v6;
    double host_res1[4], host_res2[4];
    for(i=0;i<n;i++)
        for(j=0;j<m;j++)
            V_row[i*m+j] = V[j*n+i];
    for(i=0;i<n;i++)
        lambda[i] = D[i];
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
        delta_z[i] = p[i] - (host_res1[0] + host_res1[1] + host_res1[2] + host_res1[3]);
        for(;j<m;j++)
            delta_z[i] -= V_row[i*m+j]*q[j];
        for(j=0;j<m;j++)
            q[j] += delta_z[i] * B[i*m+j];
        delta_z[i] *= lambda_inverse[i]; // diagonal substitution
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
        delta_z[i] = delta_z[i] - (host_res1[0] + host_res1[1] + host_res1[2] + host_res1[3]);
        for(;j<m;j++)
            delta_z[i] -= B[i*m+j]*q[j];
        for(j=0;j<m;j++)
            q[j] += delta_z[i] * V_row[i*m+j];
    }       
    free(B); free(M); free(lambda); free(lambda_inverse); free(q); free(V_row);
}

void ETC_L1_QP(double *L_Q_inv, double *V, double *c, double *G, double *g, double *rho, double epsilon, int m, int n, double *y, double *run_time)
{
    int i, j, k, iter;
    struct timespec t_start, t_end;
    double h_norm = 0.0;
    double eta = 1.0/((sqrt(2.0)+1.0)*sqrt(2.0*n) + 1.0);
    double tau = 1.0/(1.0-eta);
    double lambda = 1.0/sqrt(n+1);
    int    max_iter = ceil(-0.5*log(2.0*n/epsilon)/log(1-eta)) + 1;
    double *V_hat   = (double*)aligned_alloc(64,n*n*sizeof(double));
    double *h       = (double*)aligned_alloc(64,n*sizeof(double));
    double *z       = (double*)aligned_alloc(64,n*sizeof(double));
    double *gamma   = (double*)aligned_alloc(64,n*sizeof(double));
    double *theta   = (double*)aligned_alloc(64,n*sizeof(double));
    double *phi     = (double*)aligned_alloc(64,n*sizeof(double));
    double *psi     = (double*)aligned_alloc(64,n*sizeof(double));
    double *D       = (double*)aligned_alloc(64,n*sizeof(double));
    double *p       = (double*)aligned_alloc(64,n*sizeof(double));
    double *temp1   = (double*)aligned_alloc(64,n*sizeof(double));
    double *temp2   = (double*)aligned_alloc(64,n*sizeof(double));
    double *temp_r1 = (double*)aligned_alloc(64,n*sizeof(double));
    double *temp_r2 = (double*)aligned_alloc(64,n*sizeof(double));
    double *delta_z = (double*)aligned_alloc(64,n*sizeof(double));
    double *d_tmp   = (double*)aligned_alloc(64,m*sizeof(double));
    double *d       = (double*)aligned_alloc(64,n*sizeof(double));
    double *y_tmp   = (double*)aligned_alloc(64,m*sizeof(double));

    clock_gettime(CLOCK_MONOTONIC, &t_start);

    // compute d
    for(i=0;i<m;i++)
    {   
        d_tmp[i] = 0.0;
        for(j=0;j<m;j++)
            d_tmp[i] += L_Q_inv[i+j*m]*c[j];
    }
    for(i=0;i<n;i++)
    {
        d[i] = g[i];
        for(j=0;j<m;j++)
            d[i] += V[i+j*n]*d_tmp[j];
    }
    // compute h
    for(i=0;i<m;i++)
    {
        d_tmp[i] = 0.0;
        for(j=0;j<n;j++)
            d_tmp[i] += V[j+i*n]*rho[j];
    }
    for(i=0;i<n;i++)
    {
        h[i]= 2*d[i];
        for(j=0;j<m;j++)
            h[i] += V[i+j*n]*d_tmp[j];
        h[i] *= rho[i];
    }

    for(i=0;i<n;i++)
    {
        z[i] = 0.0;
        h_norm = h_norm > fabs(h[i]) ? h_norm : fabs(h[i]);
    }
    if(h_norm==0)
    {
        // recover y from z
        for(i=0;i<m;i++)
        {
            d_tmp[i] = c[i];
            for(j=0;j<n;j++)
                d_tmp[i] += 0.5*G[i*n+j]*rho[j];
        }
        for(i=0;i<m;i++)
        {   
            y_tmp[i] = 0.0;
            for(j=0;j<m;j++)
                y_tmp[i] += L_Q_inv[i+j*m] * d_tmp[j];
        }
        for(i=0;i<m;i++)
        {
            y[i] = 0.0;
            for(j=0;j<m;j++)
                y[i] -= L_Q_inv[i*m+j]*y_tmp[j];
        }
        clock_gettime(CLOCK_MONOTONIC, &t_end);
        *run_time = (t_end.tv_sec  - t_start.tv_sec) + (t_end.tv_nsec-t_start.tv_nsec)/1.0e9;
        return;
    }
    else
    {
        // scale V
        for(i=0;i<m;i++)
        {
            for(j=0;j<n;j++)
                V_hat[i*n+j] = sqrt(2*lambda/h_norm) * rho[j] * V[i*n+j];
        }
        // initialize gamma, theta, phi, psi
        for(i=0;i<n;i++)
        {
            gamma[i] = 1.0 - lambda/h_norm * h[i];
            theta[i] = 1.0 + lambda/h_norm * h[i];
            phi[i] = 1.0;
            psi[i] = 1.0;
        }
        // start the iterations
        for(iter=0;iter<max_iter;iter++)
        {
            tau = (1.0-eta)*tau;
            for(i=0;i<=n-4;i+=4)
            {
                temp1[i] = gamma[i]/phi[i]; temp1[i+1] = gamma[i+1]/phi[i+1]; temp1[i+2] = gamma[i+2]/phi[i+2]; temp1[i+3] = gamma[i+3]/phi[i+3];
                temp2[i] = theta[i]/psi[i]; temp2[i+1] = theta[i+1]/psi[i+1]; temp2[i+2] = theta[i+2]/psi[i+2]; temp2[i+3] = theta[i+3]/psi[i+3]; 
                temp_r1[i] = 2.0*(sqrt(temp1[i])*tau-gamma[i]); temp_r1[i+1] = 2.0*(sqrt(temp1[i+1])*tau-gamma[i+1]); temp_r1[i+2] = 2.0*(sqrt(temp1[i+2])*tau-gamma[i+2]); temp_r1[i+3] = 2.0*(sqrt(temp1[i+3])*tau-gamma[i+3]);
                temp_r2[i] = 2.0*(sqrt(temp2[i])*tau-theta[i]); temp_r2[i+1] = 2.0*(sqrt(temp2[i+1])*tau-theta[i+1]); temp_r2[i+2] = 2.0*(sqrt(temp2[i+2])*tau-theta[i+2]); temp_r2[i+3] = 2.0*(sqrt(temp2[i+3])*tau-theta[i+3]); 
                D[i] = temp1[i] + temp2[i]; D[i+1] = temp1[i+1] + temp2[i+1]; D[i+2] = temp1[i+2] + temp2[i+2]; D[i+3] = temp1[i+3] + temp2[i+3];
                p[i] = temp_r2[i] - temp_r1[i]; p[i+1] = temp_r2[i+1] - temp_r1[i+1]; p[i+2] = temp_r2[i+2] - temp_r1[i+2]; p[i+3] = temp_r2[i+3] - temp_r1[i+3];
            }
            for(;i<n;i++)
            {
                temp1[i] = gamma[i]/phi[i];
                temp2[i] = theta[i]/psi[i];
                temp_r1[i] = 2.0*(sqrt(temp1[i])*tau-gamma[i]);
                temp_r2[i] = 2.0*(sqrt(temp2[i])*tau-theta[i]);
                D[i] = temp1[i] + temp2[i];
                p[i] = temp_r2[i] - temp_r1[i];
            }
            
            vecLDL(V_hat, D, p, m, n, delta_z); // solving (VV^T + D) dz = p

            // update z, gamma, theta, phi, psi
            for(i=0;i<=n-4;i+=4)
            {

                z[i] += delta_z[i]; z[i+1] += delta_z[i+1]; z[i+2] += delta_z[i+2]; z[i+3] += delta_z[i+3];
                gamma[i] += (temp_r1[i] + temp1[i]*delta_z[i]); gamma[i+1] += (temp_r1[i+1] + temp1[i+1]*delta_z[i+1]); gamma[i+2] += (temp_r1[i+2] + temp1[i+2]*delta_z[i+2]); gamma[i+3] += (temp_r1[i+3] + temp1[i+3]*delta_z[i+3]);
                theta[i] += (temp_r2[i] - temp2[i]*delta_z[i]); theta[i+1] += (temp_r2[i+1] - temp2[i+1]*delta_z[i+1]); theta[i+2] += (temp_r2[i+2] - temp2[i+2]*delta_z[i+2]); theta[i+3] += (temp_r2[i+3] - temp2[i+3]*delta_z[i+3]);
                phi[i] -= delta_z[i]; phi[i+1] -= delta_z[i+1]; phi[i+2] -= delta_z[i+2]; phi[i+3] -= delta_z[i+3]; 
                psi[i] += delta_z[i]; psi[i+1] += delta_z[i+1]; psi[i+2] += delta_z[i+2]; psi[i+3] += delta_z[i+3];
            }
            for(;i<n;i++)
            {
                z[i] += delta_z[i];
                gamma[i] += (temp_r1[i] + temp1[i]*delta_z[i]);
                theta[i] += (temp_r2[i] - temp2[i]*delta_z[i]);
                phi[i] -= delta_z[i];
                psi[i] += delta_z[i];
            }
        }
        // recover y from z
        for(i=0;i<m;i++)
        {
            d_tmp[i] = c[i];
            for(j=0;j<n;j++)
                d_tmp[i] += 0.5*G[i*n+j]*(rho[j]*z[j]+rho[j]);
        }
        for(i=0;i<m;i++)
        {   
            y_tmp[i] = 0.0;
            for(j=0;j<m;j++)
                y_tmp[i] += L_Q_inv[i+j*m] * d_tmp[j];
        }
        for(i=0;i<m;i++)
        {
            y[i] = 0.0;
            for(j=0;j<m;j++)
                y[i] -= L_Q_inv[i*m+j]*y_tmp[j];
        }
        clock_gettime(CLOCK_MONOTONIC, &t_end);
        *run_time = (t_end.tv_sec  - t_start.tv_sec) + (t_end.tv_nsec-t_start.tv_nsec)/1.0e9;
        return;
    }   
}
