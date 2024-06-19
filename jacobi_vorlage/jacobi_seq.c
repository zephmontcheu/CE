// Kompilieren mit: gcc image_generator.c jacobi_seq.c -o jacobi -lm
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

/*Die Funktion bekommt einen Pointer auf das Gitter u und die Anzahl aller Gitterpunkte in einer Raumdimension übergeben
und erzeugt ein Bild in der Datei output.pgm*/
void generate_image(double *, size_t);

/*Die Funktion bekommt einen Pointer auf das Gitter u und die Anzahl aller Gitterpunkte in einer Raumdimension übergeben
und gibt die Werte des Gitters auf der Kommandozeile aus.*/
void print_grid(double *u, size_t n) {
    for(size_t i = 0; i < n; ++i) {
        for(size_t j = 0; j < n; ++j) {
            printf("%f ", u[i*n+j]);
        }
        printf("\n");
    }
}

/* Die Funktion bekommt einen Pointer auf das Gitter u und die Anzahl aller Gitterpunkte in einer Raumdimension übergeben
und gibt die L2-Norm des Residiuums zurück */
double norm_residual(double const * u, size_t const n) {
    double sum = 0;
    size_t n_t = n+2;
    for (int i =1 ; i <= n ; i++){
    	for (int j =1 ; j <= n ; j++){
    		if( i * n_t +j == n_t * n_t /2)
   			continue;
    	double norm = u[(i +1) * n_t +j]+u[(i-1) * n_t + j] +
    	 u[ i * n_t +(j+1)] + u[ i * n_t +(j-1)]  - 4* u[ i * n_t +j];
   	sum += pow(norm,2);	 						      
    	}
    }
    sum = sqrt(sum);
    printf(" %f  ,  ",sum);
    return sum;
}

/* Die Funktion bekommt Pointer auf das neue (u_new) sowie das alte Gitter u, die Anzahl der inneren Gitterpunkte n in einer Raumdimension und

die maximale Anzahl an Zeitschritten (steps) übergeben, wendet das Jacobiverfahren steps-Mal an und gibt als Rückgabewert steps zurück. 

Aufgabe c) Die Methode bricht ab, wenn die L2-Norm des Residiums kleiner 0,1 ist und gibt die Anzahl der benötigten Zeitschritte aus und als Rückgabewert zurück.
Wurde innerhalb von step Zeitschritten keine (genaue) Lösung gefunden, printet die Methode einen Fehler auf die Kommandozeile (printf) und gibt als Rückgabewert steps zurück.*/
int jacobi(double *u, double *u_new, size_t const n, size_t const steps) {
	 size_t n_t = n +2;
	 int real_steps = 0;
   	for ( int k = 0; k < steps ; k++){
   	if(norm_residual(u,n) < 0.1){
   		printf("steps = %d \n",real_steps);
   			return real_steps;
   			}
   		real_steps++;
   		for ( int i = 1; i <=n ; i++ ){
   			for (int j = 1 ; j <= n ; j++){
   				if( i * n_t +j == n_t * n_t /2)
   					continue;
   				
   		 		u_new[ i * n_t + j]= ( u[(i +1) * n_t +j] + u[(i-1) * n_t + j] +u[ i * n_t +(j+1)] 
   		 						      + u[ i * n_t +(j-1)])/4;
   			}
   		}
   		
   		 double *temp = u;
   		 u = u_new;	
   		 u_new= temp;
	}
    printf( " es wird keine lösungen gefunden \n");
    return  real_steps; 
}
int par_jacobi(double *u, double *u_new, size_t const n, size_t const steps) {
	
}*/
 void init_quadratic_grid ( double *u , size_t const n_ges,  size_t const m){
 	for (int i = 0; i < n_ges ; i++ ){
 		for (int j =  0 ; j < n_ges ; j++){
 			if ( n_ges *n_ges/ 2== (i * n_ges + j)){
 				u[i * n_ges + j]= 1;
 				continue;
 			}
 			 u[ i* n_ges + j] = 0;
 		}
 	}
 }

int main(int argc, char **argv) {
    if(argc != 3) {
        printf("Usage: %s size iterations\n", argv[0]);
        return EXIT_FAILURE;
    }
    size_t m = atol(argv[1]);
    int iterations = atoi(argv[2]);

    double *u;
    double *u_new;
    size_t n, n_ges; //n = Anzahl der inneren Gitterpunkte in einer Raumdimension, n_ges = Anzahl aller Punkte in einer Raumdimension
    n_ges = m * 2 + 1;
    n = n_ges - 2;
     u =  (double *) malloc (n_ges * sizeof(double) * n_ges);
     u_new = (double *) malloc ( n_ges * sizeof(double) * n_ges);
     init_quadratic_grid (u , n_ges , m);
     init_quadratic_grid (u_new , n_ges , m);
    //TODO Speicher für u und u_new allokieren und initialisieren und n und n_ges berechnen
	print_grid(u , n_ges);
	printf("\n");
    int k = jacobi(u, u_new, n, iterations); 
    if (k % 2 == 1) {	//Tauscht die Pointer von u und u_new, in Abhängigkeit, wie oft jacobi iteriert hat, sodass u auf das neueste Gitter zeigt
        double * tmp = u;
        u = u_new;
        u_new = tmp;
    }
    generate_image(u, n_ges);
    print_grid(u,n_ges);
    free(u);
    free(u_new);

}

