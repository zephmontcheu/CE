// Kompilieren mit: nvcc -ccbin clang-14 generate_image.c jacobi.cu -o jacobi_cuda -lm
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

extern "C" void generate_image(double *, size_t);

void print_grid(double* u, size_t n) {
    for(size_t i = 0; i < n; ++i) {
        for(size_t j = 0; j < n; ++j) {
            printf("%f ", u[i*n+j]);
        }
        printf("\n");
    }
    printf("\n\n");
}

__global__ void sum_residual_gpu(double* r, size_t const n) {
    extern __shared__ double sum[];

    size_t const m = n+2;
    int posX = threadIdx.x+1;
    int posY = blockIdx.x+1;
    if (posX < n+1 && posY < n+1) {
	if (!(posX == m / 2 && posY == m / 2)) {
	    sum[threadIdx.x] = r[posY*m+posX];
	}else{
	    sum[threadIdx.x] = 0;
 	}
        __syncthreads();
        
	//TODO Baumartiges aufsummieren der Werte in sum, das Endergebnis soll in sum[0] stehen
	for ( int stride = blockDim.x/2 ; stride >0 ; stride /= 2){
		if(threadIdx.x < stride){
			sum[threadIdx.x] += sum [threadIdx.x + stride]
		}
		__synthreads();
	}

        if(threadIdx.x == 0) {
            r[blockIdx.x+m+1] = sum[0];
        }
    }
}

/* Die Funktion bekommt Pointer auf das neue (u_new) sowie das alte Gitter u, einen Pointer auf das Residuumsgitter r und die Anzahl der inneren Gitterpunkte n in einer Raumdimension übergeben, 
berechnet für jeden Gitterpunkt den neuen Näherungswert und schreibt diesen in u_new. 
Danach wird das Residuum für jeden Gitterpunkt, basierend auf dem alten Gitter u, berechnet, quadriert und das Ergebnis in r geschrieben. */
__global__ void jacobi_gpu(double* u, double* u_new, double* r, size_t const n) {
    size_t const m = n + 2;
    int posX = threadIdx.x+1;
    int posY = blockIdx.x+1;
    
    //TODO
}

size_t jacobi_helper(double *u, double *u_new, size_t const n, size_t const steps) {
    dim3 blockDim(n);
    dim3 gridDim(n);
    double* r = (double*)calloc(1, sizeof(double));
    double* d_r;
    cudaMalloc((double**)&d_r, (n+2)*(n+2)*sizeof(double));
    
    for (size_t k = 0; k < steps; ++k) {
        jacobi_gpu<<<gridDim, blockDim>>>(u, u_new, d_r, n); 
        cudaDeviceSynchronize();
	double* tmp = u;
	u = u_new;
	u_new = tmp;

	sum_residual_gpu<<<n, n, n*sizeof(double)>>>(d_r, n);
        sum_residual_gpu<<<1, n, n*sizeof(double)>>>(d_r, n);
        cudaDeviceSynchronize();
        cudaMemcpy(r, &d_r[n+2+1], sizeof(double), cudaMemcpyDeviceToHost);
        r[0] = sqrt(r[0]);
        if(k%1000 == 0) {
            printf("Residual norm in step %zu: %f\n", k, r[0]);
        }
        if(*r < 1e-2) { 
            printf("Required %zu steps for solving\n", k);
            break;
        }

    }
    
    cudaFree(d_r);
    free(r);
    return k;
}

double get_time(void) {
    struct timespec a;
    clock_gettime(CLOCK_MONOTONIC, &a);
    double t = (double) a.tv_nsec * 1e-6 + (double) a.tv_sec*1e3;
    return t;
}

int main(int argc, char **argv) {
    if(argc != 2) {
        printf("Usage: %s size\n", argv[0]);
        return EXIT_FAILURE;
    }
    size_t m = atol(argv[1]);
    size_t const n_ges = 2*m + 1;
    size_t const n = n_ges - 2;

    size_t data_points = n_ges * n_ges;
    double* u = (double*) calloc(data_points, sizeof(double));
    size_t midpoint = m * n_ges + m;

    u[midpoint] = 1.0;
    print_grid(u, n_ges);
    double* d_u, *d_u_new;
    cudaMalloc((double**)&d_u, data_points * sizeof(double));
    cudaMalloc((double**)&d_u_new, data_points * sizeof(double));

    cudaMemcpy(d_u, u, data_points * sizeof(double), cudaMemcpyHostToDevice);
    cudaError_t status = cudaMemcpy(d_u_new, u, data_points * sizeof(double), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        printf("Error at memcpy: %d\n", status);
        return 1;
    }

    size_t k = jacobi_helper(d_u, d_u_new, n, 10000);
    if (k % 2 == 0) { 
        double * tmp = d_u;
        d_u = d_u_new;
        d_u_new = tmp;
    }
    cudaMemcpy(u, d_u_new, data_points * sizeof(double), cudaMemcpyDeviceToHost);

    generate_image(u, n_ges);
    print_grid(u, n_ges);

    cudaFree(d_u);
    cudaFree(d_u_new);
    free(u);
}

