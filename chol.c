#include <time.h>
#include <math.h>
#include <stdlib.h>
void chol(double *D, double *V, double *w, int m, int n, double *u, double *run_time)
{
    int i, j, k;
    struct timespec t_start, t_end;
    double *M=(double*)aligned_alloc(64, n*n*sizeof(double));
    double sum;

    clock_gettime(CLOCK_MONOTONIC, &t_start);
    // calculate M
    for(i=0;i<n;i++)
	{
		for(j=0;j<n;j++)
		{
			M[i*n+j] = 0.0;
			if(i==j)
				M[i*n+i] += D[i*n+i];
			for(k=0;k<m;k++)
				M[i*n+j] += V[k*n+i] * V[k*n+j];
		}
	}
    // start cholesky decomposition
    for(i=0;i<n;i++)
    {
        M[i+i*n]=sqrt(M[i+i*n]);
        for(j=i+1;j<n;j++)
            M[j+i*n]=M[j+i*n]/M[i+i*n];
        for(k=i+1;k<n;k++)
            for(j=k;j<n;j++)
                M[j+k*n]=M[j+k*n]-M[j+i*n]*M[k+i*n];
    }
    // start substitition
    for(i=0;i<n;i++)
    	u[i] = w[i];
    for(i=0;i<n;i++)
    {
    	sum = 0.0;
    	for(j=0;j<i;j++)
    		sum += M[i+j*n]*u[j];
    	u[i] = (u[i]-sum)/M[i+i*n];
    }
    for(i=n-1;i>=0;i--)
    {
    	sum = 0.0;
    	for(j=i+1;j<n;j++)
    		sum += M[j+i*n]*u[j];
    	u[i] = (u[i]-sum)/M[i+i*n];
    }
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    *run_time = (t_end.tv_sec  - t_start.tv_sec) + (t_end.tv_nsec-t_start.tv_nsec)/1.0e9;
    free(M);
}