__kernel void buddhaTraj(__global const float *initPointsA,
			 __global const float *initPointsB, 
		  	 __global float *trajsA,
			 __global float *trajsB,
			 
			 const int MAXITER, 
		 	 const int NPOINTS,
			 __global float *randomPointsA, 
			 __global float *randomPointsB){

int gid = get_global_id(0);
float2 z0 = (float2)(randomPointsA[gid], randomPointsB[gid]);

float xtemp = 0;
float x = 0;
float y = 0;

int escaped = 0;
int compt = 0;

for (int i = 0; i < MAXITER; i++){
	compt++;
	xtemp = x*x - y*y + z0.x;
	y = 2*x*y + z0.y;
	x = xtemp;
	
	trajsA[MAXITER*gid + i] =x;
	trajsB[MAXITER*gid + i] =y;
	
	if (x*x+y*y > 4 ){
	escaped = 1;
	
	for(int j = i+1; j < MAXITER; j++){
		trajsA[MAXITER * gid + j] = -100;
		trajsB[MAXITER * gid + j] = -100;
	}
	
		break;
	}
	if (escaped == 0 && compt == MAXITER){
	for(int j = 0; j < MAXITER; j++){
		trajsA[MAXITER * gid + j] = -10;
		trajsB[MAXITER * gid + j] = -10;
	}
	}
}
}
