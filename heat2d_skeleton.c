//Compile with gcc heat2d.c -o heat2d -Wall -Werror -O3 -lm
//Run with ./heat2d (plus Arguments)

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

 

/* Die Funktion bekommt einen Pointer auf das Gitter plate, die Anzahl der inneren Gitterpunkte N sowie die Schrittweite h_x übergeben
und initialisiert das Gitter mit den Randbedingungen der Differentialgleichung. */
void initEdges(double *plate, int N, double h_x){
     
     int nt = N+2; // anzahl der Gitterpunkten inklusive die Randpunkten
     const double pi = acos(-1);
     for( int i = 0; i< nt ; ++i){
	for(int j=0; j< nt  ; ++j){
		if( i == 0)  // randpunkten links 
			plate[i * nt+ j] =  1 - sin((pi * j * h_x));
	   	 else if  (i == nt-1) // randpunkten rechts
	   		plate [ i * nt + j] =   sin((pi * j * h_x )) - 1;
	   	else if ( j ==  0 || j == nt-1) // randpunkten oben und unten
	   		plate[ i * nt  + j ] =  cos( (pi *  i * h_x));
	   	else 	
	   		plate [ i * nt + j] = 0;	
     }
    
    // TODO
}
}

/* Die Funktion bekommt Pointer auf das neue (plate_new) sowie das alte (plate_old) Gitter, die Anzahl der inneren Gitterpunkte N und
die Schrittweiten h_x, h_y und h_t übergeben, berechnet einen Zeitschritt mithilfe der Berechnungsformel aus Aufgabenteil a) 
und trägt die resultierenden Werte in das neue Gitter ein. */
void timestep(double *plate_new, double *plate_old, int N, double h_x, double h_y, double h_t){
   
  	int nt = N+2;
    	for ( int i = 1; i <= N  ; i++){
    	
    		for (int j =1; j <= N ; j++){
    			double zwischen_x = ( plate_old[(i+1) * nt +j] + plate_old[(i-1) * nt + j] - 2 * plate_old[ i * nt +j])/ (pow(h_x,2));
    			double zwischen_y =  ( plate_old [ i * nt + j+1] + plate_old [ i * nt + j-1] - 2 * plate_old[ i * nt + j]) / (pow (h_y,2));
    			plate_new [ i * nt + j] =  h_t * (zwischen_x + zwischen_y) + plate_old[ i * nt + j];	
    		}
    	}
    
    // TODO
}
void par_timestep (double *plate_new, double *plate_old, int N, double h_x, double h_y, double h_t, int threads){
		int nt = N+2;
		#pragma omp parallel for   schedule (static , threads)
    	for ( int i = 1; i <= N  ; i++){
    		for (int j =1; j <= N ; j++){
    			double zwischen_x = ( plate_old[(i+1) * nt +j] + plate_old[(i-1) * nt + j] - 2 * plate_old[ i * nt +j])/ (pow(h_x,2));
    			double zwischen_y =  ( plate_old [ i * nt + j+1] + plate_old [ i * nt + j-1] - 2 * plate_old[ i * nt + j]) / (pow (h_y,2));
    			plate_new [ i * nt + j] =  h_t * (zwischen_x + zwischen_y) + plate_old[ i * nt + j];	
    		}
    	}
    
}

void output(double *plate, int N){
    printf("\n");			
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            printf("%f\t", plate[i * N + j]);
        }
        printf("\n");
    printf("\n");
	}
}

int main(int argc, char** argv){
     		
    long N;
    long n_t;
    int num_threads;

    if(argc == 4){
        N = strtol(argv[1], NULL, 10);
        n_t = strtol(argv[2], NULL, 10);
        num_threads = strtol(argv[3] , NULL , 10);
    }
    else{
        printf("Usage: %s <n_xy> <n_t>\n", argv[0]);
        return 0;
    }
	double *plate_old  = malloc ( sizeof ( double ) * (N+2) * (N+2));
	double *plate_new = malloc ( sizeof ( double ) * (N+2) * (N+2));
	double h_x =   1.000 / (N +1) ;
	double h_t = 1.000 /( n_t +1) ;
	printf( "%f\n %f\n", h_x , h_t);
	initEdges( plate_old , N , h_x);
	initEdges ( plate_new , N , h_x);
	output ( plate_old , N+2);
	for ( long k = 0 ; k < n_t ; ++k){
		// double  *swapper = NULL; 
		par_timestep( plate_new , plate_old , N ,  h_x , h_x, h_t,num_threads);
		//timestep(plate_new , plate_old , N , h_x , h_x,h_t);
		plate_old = plate_new;
	}
	
	output ( plate_old, N+2);
	//free(plate_old);
	//free(plate_new);
    // TODO    

    return 0;
}
