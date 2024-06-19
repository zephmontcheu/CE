#include <stdio.h> 
  
void generate_image(double *grid, int size) 
{
    FILE* pgmimg; 
    pgmimg = fopen("output.pgm", "w"); 
  
    // Writing Magic Number to the File 
    fprintf(pgmimg, "P2\n");  
  
  	int factor = 2;
  	int image_size = size * factor;
  
    // Writing Width and Height 
    fprintf(pgmimg, "%d %d\n", size, size);  
  
    // Writing the maximum gray value 
    fprintf(pgmimg, "255\n");  
    
    for (int i = 0; i < size; i++) { 
        for (int j = 0; j < size; j++) {  
            // Writing the gray values in the 2D array to the file 
            fprintf(pgmimg, "%d ", 255 - (int) (255 * grid[i * size + j])); 
        } 
        fprintf(pgmimg, "\n"); 
    } 
    fclose(pgmimg); 
} 
