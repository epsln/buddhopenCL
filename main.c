#include "include/bmp.h"
#include "include/math_utils.h"
#include "include/readFiles.h"
#include "include/mandel.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x100000)

int main(){
	//Hyperparameters read from the config file
	int RESX;
	int RESY;
	int NPOINTS; //Number of point in the file (TODO: remove and add pure random point gen option)
	int GRAY = 0;
	int RANDOM_MODE = 1; //If set at one -> don't read the precomputed points file 
	int JITTER;
	int OPENCL_ITER;//How many rounds of NPOINTS will we pass to the kernels
	int MAXITER;
	int MAXITER_G;
	int MAXITER_B;
	float JITTER_BOUND; //random values of jitter are in this range
	float RED_COEFF   = 1; //In case of grayscale, is the default coefficient
	float GREEN_COEFF = 1;
	float BLUE_COEFF  = 1;
	char filename[256] = "img.bmp";

	int *pRESX        = &RESX;    
	int *pRESY        = &RESY;
	int *pNPOINTS     = &NPOINTS;
	int *pJITTER      = &JITTER;
	int *pGRAY        = &GRAY;
	int *pRAND        = &RANDOM_MODE;
	int *pOPENCL_ITER = &OPENCL_ITER;
	int *pMAXITER     = &MAXITER;
	int *pMAXITER_G   = &MAXITER_G;
	int *pMAXITER_B   = &MAXITER_B;
	float *pJITTER_B  = &JITTER_BOUND;
	float *pRED_C     = &RED_COEFF;
	float *pGRE_C     = &GREEN_COEFF;
	float *pBLU_C     = &BLUE_COEFF;

	readConf(pRESX, pRESY, pNPOINTS, pJITTER, pJITTER_B, pGRAY,pOPENCL_ITER, pMAXITER, pMAXITER_G, pMAXITER_B, pRED_C, pGRE_C, pBLU_C, filename, pRAND);

	printf("[PARAMETERS]\n");
	if(RANDOM_MODE == 1){
		printf("\
				RANDOM MODE: \x1B[32m  [ON]\x1B[0m\n");}
	else{
		printf("\
				RANDOM MODE: \x1B[31m  [OFF]\x1B[0m\n");}
	if(GRAY == 1){
		printf("\
				GRAYSCALE MODE:\x1B[32m[ON]\x1B[0m\n");}
	else{
		printf("\
				GRAYSCALE MODE:\x1B[31m[OFF]\x1B[0m\n");}
	printf("\
			RESX:          %d\n\
			RESY:          %d\n\
			NPOINTS:       %d\n\
			JITTER:        %d\n\
			JITTER BOUNDS: [-%f;%f]\n\
			OPENCL ITER:   %d\n\
			MAXITER:       %d\n", RESX, RESY, NPOINTS, JITTER, JITTER_BOUND, JITTER_BOUND, OPENCL_ITER, MAXITER);
	if (GRAY == 0){
		printf("\
				MAXITER GREEN: %d\n\
				MAXITER BLUE : %d\n\
				RED COEFF    : %f\n\
				GREEN COEFF  : %f\n\
				BLUE COEFF   : %f\n\
				\n", MAXITER_G, MAXITER_B, RED_COEFF, GREEN_COEFF, BLUE_COEFF);

	}
	printf("\tFILENAME:      %s\n", filename);

	float c;
	long long int maxiR = 0;
	long long int maxiG = 0;
	long long int maxiB = 0;

	float **pointsArray;
	pointsArray = malloc(NPOINTS * sizeof(float *));
	for (int i = 0; i < NPOINTS; i++){
		pointsArray[i] = (float *)malloc(2 * sizeof(float));
	}

	//OpenCL point list 
	float *initialPointsA = (float *)malloc(sizeof(float)*NPOINTS);
	float *initialPointsB = (float *)malloc(sizeof(float)*NPOINTS);

	//Output arrays (histograms)
	long long int **collisionsArrayR = malloc(RESX * sizeof(int *));
	long long int **collisionsArrayG = malloc(RESX * sizeof(int *));
	long long int **collisionsArrayB = malloc(RESX * sizeof(int *));

	//Output BMP arrays
	float **bmpR = malloc(RESX * sizeof(float *));
	float **bmpG = malloc(RESX * sizeof(float *));
	float **bmpB = malloc(RESX * sizeof(float *));


	//OpenCL Begin 

	srand(time(NULL));
	// Clean up, release memory.


	for (int i =0; i < RESX; i++) {
		collisionsArrayR[i] = (long long int *)malloc(RESY * sizeof(long long int)); 
		if (GRAY == 0){
			collisionsArrayG[i] = (long long int *)malloc(RESY * sizeof(long long int)); 
			collisionsArrayB[i] = (long long int *)malloc(RESY * sizeof(long long int)); 
		}
		bmpR[i] = (float *)malloc(RESY * sizeof(float)); 
		if (GRAY == 0){
			bmpG[i] = (float *)malloc(RESY * sizeof(float)); 
			bmpB[i] = (float *)malloc(RESY * sizeof(float)); 
		}
	}


	if (collisionsArrayR == NULL || bmpR == NULL ){ // Probably useless..
		// || bmpG == NULL || bmpB == NULL)
		printf("Allocation failed ! Exiting...\n");
		return -1;
	}

	if (RANDOM_MODE == 0){
		readPoints(pointsArray);
	}
	//Zero filling arrays
	for (int i = 0; i < RESX; i++){
		for (int j = 0; j < RESY; j++){
			collisionsArrayR[i][j] = 0;
			if (GRAY == 0){
				collisionsArrayG[i][j] = 0;
				collisionsArrayB[i][j] = 0;
			}
		}
	}

	//Generating points by reading the jungreis file and adding jitter
	if (GRAY == 1){
		//We need to pass a list of points, that will be the size of NPOINTS to the opencl kernel
		//We also need to pass the collisionArray to the kernel, along with info on its dims
		if (RANDOM_MODE == 0){
			for (int i = 0; i < NPOINTS; i++){
				initialPointsA[i] = pointsArray[i][0] ;
				initialPointsB[i] = pointsArray[i][1] ;
			}
			mandelIterOpenCL(initialPointsA, initialPointsB, NPOINTS, MAXITER, RESX, RESY, collisionsArrayR);

		}
		else{
			int stopMark = -1;
			for (int i = 0; i < OPENCL_ITER; i++){
				stopMark = progressBar(i, OPENCL_ITER, stopMark);
				for (int i = 0; i < NPOINTS; i++){
					float a = float_rand(-2.0, 2.0);
					float b = float_rand(-2.0, 2.0);

					initialPointsA[i] = a;
					initialPointsB[i] = b;
				}
				mandelIterOpenCL(initialPointsA, initialPointsB, NPOINTS, MAXITER, RESX, RESY, collisionsArrayR);
			}
		}
	}

	//Getting the max value of the array to get a better coloring (max val == white px)
	for (int i = 0; i < RESX; i++){
		for (int j = 0; j < RESY; j++){
			if (maxiR < collisionsArrayR[i][j]){
				maxiR = collisionsArrayR[i][j];
			}
		}
	}
	printf("MAXI: %lld\n", maxiR);

	if (maxiR == 0){
		printf("EMPTY ARRAY ! Something wrong :(\n");
		exit(-2);
	}

	//Filling the bmp array
	for (int i = 0; i < RESX; i++){
		for (int j = 0; j < RESY; j++){
			c = map(collisionsArrayR[i][j], 0, maxiR, 0, 1);
			bmpR[i][j] = c;
		}
	}
	write2bmp(bmpR, bmpR, bmpR, RESX, RESY, filename);

	if (GRAY == 0){
		for(int i = 0; i < NPOINTS; i++){
			if (RANDOM_MODE == 0){
				for (int i = 0; i < NPOINTS; i++){
					initialPointsA[i] = pointsArray[i][0] ;
					initialPointsB[i] = pointsArray[i][1] ;
				}
				mandelIterOpenCL(initialPointsA, initialPointsB, NPOINTS, MAXITER,   RESX, RESY, collisionsArrayR);
				mandelIterOpenCL(initialPointsA, initialPointsB, NPOINTS, MAXITER_G, RESX, RESY, collisionsArrayG);
				mandelIterOpenCL(initialPointsA, initialPointsB, NPOINTS, MAXITER_B, RESX, RESY, collisionsArrayB);

			}
			else{
				for (int i = 0; i < OPENCL_ITER; i++){
					for (int i = 0; i < NPOINTS; i++){
						float a = float_rand(-2.0, 2.0);
						float b = float_rand(-2.0, 2.0);
						initialPointsA[i] = a;
						initialPointsB[i] = b;
					}
					mandelIterOpenCL(initialPointsA, initialPointsB, NPOINTS, MAXITER,   RESX, RESY, collisionsArrayR);
					mandelIterOpenCL(initialPointsA, initialPointsB, NPOINTS, MAXITER_G, RESX, RESY, collisionsArrayG);
					mandelIterOpenCL(initialPointsA, initialPointsB, NPOINTS, MAXITER_B, RESX, RESY, collisionsArrayB);
					int stopMark = -1;
					if(i % 100 == 0)	
						stopMark = progressBar(i, OPENCL_ITER, stopMark);

				}
			}

		}

		//Getting the max value of the array to get a better coloring (max val == white px)
		for (int i = 0; i < RESX; i++){
			for (int j = 0; j < RESY; j++){
				if (maxiR < collisionsArrayR[i][j]){maxiR = collisionsArrayR[i][j];}
				if (maxiG < collisionsArrayG[i][j]){maxiG = collisionsArrayG[i][j];}
				if (maxiB < collisionsArrayB[i][j]){maxiB = collisionsArrayB[i][j];}
			}
		}

		//Filling the bmp array
		for (int i = 0; i < RESX; i++){
			for (int j = 0; j < RESY; j++){
				c = map(collisionsArrayR[i][j], 0, maxiR, 0, 1)*RED_COEFF;
				bmpR[i][j] = c;
				c = map(collisionsArrayG[i][j], 0, maxiG, 0, 1)*GREEN_COEFF;
				bmpG[i][j] = c;
				c = map(collisionsArrayB[i][j], 0, maxiB, 0, 1)*BLUE_COEFF;
				bmpB[i][j] = c;
			}
		}
		write2bmp(bmpR, bmpG, bmpB, RESX, RESY, filename);
	}
}
