#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

float map(double var1, double min1, double max1, double min2, double max2){
	return min2+(max2-min2)*((var1-min1)/(max1-min1));
}

float float_rand( float min, float max ){
	float scale = rand() / (float) RAND_MAX; /* [0, 1.0] */
	return min + scale * ( max - min );      /* [min, max] */
}

int progressBar(int prog, int iMax, int oldStop){
	//[-------------------------------------------------]
	//[>------------------------------------------------]
	//[x>-----------------------------------------------]
	// #- : 50  
	// x  : 2%
	int stopMark = map(prog, 0, iMax, 0, 50);
	//if (stopMark != oldStop){
		printf("[");
		for (int i = 0; i < 50; i++){
			if (i <= stopMark - 1)
				printf("x");
			else if( i == stopMark)
				printf(">");
			else
				printf("-");
		}
		printf("] %.2f\r", (float)prog/iMax*100);
		fflush(stdout);
	//}
	return stopMark;
}
