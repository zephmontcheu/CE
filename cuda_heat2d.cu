// übersetzen mit: nvcc -ccbin clang-14 cuda_heat2d.cu -o cuda_heat2D -lm
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

void initEdges(double* value, int N, double h_x) {
    int nt = N+2; // anzahl der Gitterpunkten inklusive die Randpunkten
     const double pi = acos(-1);
     for( int i = 0; i< nt ; ++i){
	for(int j=0; j< nt  ; ++j){
		if( i == 0)  // randpunkten links 
			value[i * nt+ j] =  1 - sin((pi * j * h_x));
	   	 else if  (i == nt-1) // randpunkten rechts
	   		value [ i * nt + j] =   sin((pi * j * h_x )) - 1;
	   	else if ( j ==  0 || j == nt-1) // randpunkten oben und unten
	   		value[ i * nt  + j ] =  cos( (pi *  i * h_x));
	   	else 	
	   		value [ i * nt + j] = 0;	
     }
    
}
}

__global__ void heat2d(double* __restrict__ plane_old, double* __restrict__ plane_new, double h_x, double h_t) {
    //calculate position of current thread with padding
    int pos = (blockDim.x + 2)*(blockIdx.x+1) + threadIdx.x + 1;
    double zwischen_x = (plane_old[pos + (blockDim.x +2)] + plane_old[pos - (blockDim.x +2)] - 2 * plane_old[pos]) / (h_x *h_x);	
     double zwischen_y = (plane_old[pos +1] + plane_old[pos - 1] - 2 * plane_old[pos]) / (h_x *h_x);	
     plane_new[pos] = h_t * (zwischen_x + zwischen_y) + plane_old[pos];	
    //TODO
}

void checkError() {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("ERROR: %s\n", cudaGetErrorString(error));
        exit(1);
    }
}

void output(double *plane, int N){
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            printf("%7.2f", plane[i*N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char** argv) {
    long n_x, n_t;
    if (argc == 3) {
        n_x = strtol(argv[1], NULL, 10);
        n_t = strtol(argv[2], NULL, 10);
    }
    else{
        printf("Usage %s <n_x> <n_t>\n", argv[0]);
        return 0;
    }
    double *plane_old = (double *) malloc ( sizeof(double) * (n_x +2) *(n_x + 2));
    double *plane_new = (double *) malloc ( sizeof(double) * (n_x +2) *(n_x + 2));
    double h_x =   1.000 / (n_x +1) ;
    double h_t = 1.000 /( n_t +1) ;
    size_t size = sizeof(double) * (n_x +2) *(n_x + 2);
    // initialise the Edges from new and old plate
    initEdges( plane_old, n_x , h_x);
    initEdges(plane_new, n_x, h_x);
    
    output (plane_old, n_x +2); // 
    
    //Allocate the device input for plates
    double *cuda_old= nullptr;
    cudaMalloc((void **)&cuda_old,size);
    double *cuda_new = nullptr;
    cudaMalloc((void **)&cuda_new,size);
    
    // copy
    cudaMemcpy(cuda_old,plane_old,size,cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_new,plane_new,size,cudaMemcpyHostToDevice);
    //int threadsPerBlock = n_x; 
    //int blocksperGrid = n_x; 
    for (long k = 0; k < n_t ; ++k){
    	heat2d<<<n_x,n_x>>>(cuda_old,cuda_new,h_x,h_t);
    	cuda_old = cuda_new;
    }
    cudaMemcpy(plane_new, cuda_new, size,cudaMemcpyDeviceToHost);
    output (plane_new,n_x+2);
    
    // TODO

    return 0;
}
