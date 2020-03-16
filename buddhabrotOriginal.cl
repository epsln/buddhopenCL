__kernel void buddhaTraj(__global const float *initPointsA,
			 __global const float *initPointsB, 
		  	 __global float *trajsA,
			 __global float *trajsB,
			 const int MAXITER, 
		 	 const int NPOINTS,
			 __global float *randomPointsA, 
			 __global float *randomPointsB){

int gid = get_global_id(0);
float2 z = (float2)(initPointsA[gid], initPointsB[gid]);

float xtemp = 0;
float oldX = 0;
float oldY = 0;
float x = 0;
float y = 0;
int stepsTaken = 0;
int stepLimit = 2;

int escaped = 0;
int compt = 0;


for (int i = 0; i < MAXITER; i++){
	compt++;
	//Buddhabrot equation :|
	xtemp = x*x - y*y + z.x;
	y = 2*x*y + z.y;
	x = xtemp;
	
	trajsA[MAXITER*gid + i] =z.x;
	trajsB[MAXITER*gid + i] =z.y;
	
	if (z.x*z.x+z.y*z.y > 4){
	escaped = 1;
        	
	for(int j = i+1; j < MAXITER; j++){
		trajsA[MAXITER * gid + j] = -100;
		trajsB[MAXITER * gid + j] = -100;
	}
		break;
	}

		
	if (oldX == z.x && oldY == z.y){

	for(int j = 0; j < MAXITER; j++){
		trajsA[MAXITER * gid + j] = -100;
		trajsB[MAXITER * gid + j] = -100;
		
	}
	break;
	}	
	if (stepsTaken == stepLimit){
		oldX = z.x;
		oldY = z.y;
		stepsTaken = 0;
		stepLimit *= 2;
	}
	stepsTaken++;
	

}
if ((escaped == 0 && compt == MAXITER)){
	for(int j = 0; j < MAXITER; j++){
		trajsA[MAXITER * gid + j] = -100;
		trajsB[MAXITER * gid + j] = -100;
		
	}
	}
}
