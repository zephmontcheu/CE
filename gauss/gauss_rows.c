#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <math.h>

double square(double x){
    return x * x;
}

//TODO Funktion init_matrix
void init_matrix (double *matrix, double *vector , int *loc,   int n ){
	srand(time(NULL));
	for ( int i = 0 ; i < n ; i++){
	     double random2 = rand() % 10; 
	     loc[i] = i;
	     vector[i] = random2;
	     for ( int j = 0; j < n ; j++){
	        double random1 = rand() % 10;  
	        matrix[i * n + j] = random1 ;
	     }
	}
}
void kopieren(double *A , double *b , double *U , double *c , int n, int threads){
// copy the matrix A und the vector b 
	#pragma omp parallel for schedule(static,threads)
	for ( int i = 0 ; i < n ; i++){
		c[i] = b[i];
	    for (int j = 0 ; j < n ; j++ ){
	    	U[i * n +j] = A[ i * n + j];
	    }
	}
}

void output(double *A, double *b, int *loc, double *x ,int n){
    for(int i = 0; i < n; i++){
      printf("%d |\t", loc[i]); // output for the location of the rows
        for(int j = 0; j < n; j++){
            printf("%f\t", A[loc[i] * n + j ]); 
        }
        printf("|\t%f\t", b[loc[i]]); 
        printf("|\t%f\n", x[loc[i]]); 
    }
    printf("\n");
}

 
 void gauss_pivot(double *A , double *b ,int *loc, int n,int threads){
 // do the gauss algorithms
 	for ( int i = 0 ; i < n ; i++ ){
 	     int magnitude = 0;
 	     int picked  = i;
 	     for ( int j = i;  j < n ; j++){
 	     // search the pivot row 
 	     	if (A [loc[j] * n + i ] > magnitude){
		      magnitude =  A [loc[j] * n + i] ;
		      picked = j;		
 	     	}
 	     }
 	     //save the pivot row und switch it
 	     int  tmp = loc[i];
 	     loc[i] = loc[picked] ;
 	     loc[picked] = tmp ;
 	     for ( int j = i+1; j < n ; j++ ){
 	     	double temp = A[loc[j] * n + i ] / A [loc[i] * n + i];
 	     	#pragma omp parallel for schedule(static , threads)
 	     	for (int k = i ; k < n ; k++){
	    	     A[loc[j] * n + k]  =  A[loc[j] * n +k]  - A [loc[i] * n + k] * temp ;
	    	     b [loc[k]] = b[loc[k]] -b[loc[i]] * temp ;
	    	}
 	     }
 	     
 	     
 	}
 }
/
void pivot_back_substitution(  double *U, double *c, double *x, int *loc,int n, int threads ){
	for (int i = n-1 ; i >= 0 ; i--){
		x [loc[i]]= c[loc[i]]/ U[loc[i] * n  + i];
	   #pragma omp parallel for schedule (static,threads)
	    for ( int j = 0 ; j < i ; j++){
	 	c[loc[j]] -= x[loc[i]] * U[ loc[j] * n + i];
	    }
	}
}
//TODO Funktion residual_norm
double residual_norm( double *A , double *b , double  *x, int *loc, int n ){
	double *end = malloc ( n * sizeof(double));
	double norm = 0;
	for ( int i = 0 ; i < n ; i++ ){
	#pragma omp parallel for reduction (+:norm)
		double zwischen = 0;
		for (int j = 0 ; j < n ; j++){
		 zwischen += A [ loc[i] * n +j] * x[loc[j]];
		}
		 end [loc[i]] = zwischen;
		double  sub =  b[loc[i]] - end[loc[i]];
		norm += pow(sub,2);
	}
	norm = sqrt(norm);
	//printf( " %f\n" , norm);
	return norm;
}

int main(int argc, char** argv){
    if(argc != 3){
        printf("Usage: %s <n> <nthreads>\n", argv[0]);
        return 0;
    }
    int n = (int) strtol(argv[1], NULL, 10);
    int nthreads = (int) strtol(argv[2], NULL, 10);
    double *A = malloc ( n * n * sizeof(double));
    double *U = malloc ( n * n * sizeof(double));// matrix für die zwischenspeicherung
    double *b = malloc ( n *sizeof(double));
    double *c = malloc ( n *sizeof(double));// vector für die zwischenspeicherung
    double *x = calloc ( n, sizeof(double));
    int *loc = malloc ( n*sizeof(int));
    init_matrix (A , b ,loc , n);
    kopieren(A , b , U , c,n);
    output(A , b , loc ,x ,n );
    printf( "\n");
    gauss_pivot( U , c,loc,n);
    //output(U, c, loc ,x ,n );
    pivot_back_substitution ( U, c, x, loc, n );
    residual_norm(A , b ,x ,loc, n);
    output ( A ,b , loc , x , n);
    printf( "\n");
    output ( U ,c , loc , x , n);
    free(A);
    free(U);
    free(b);
    free(c);
    free(x);
    // TODO
}
