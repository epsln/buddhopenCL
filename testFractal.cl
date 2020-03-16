__kernel void buddhaTraj(__global const double *initPointsA,
			 __global const double *initPointsB, 
		  	 __global double *trajsA,
			 __global double *trajsB,
			 const int MAXITER, 
		 	 const int NPOINTS,
			 __global double *randomPointsA, 
			 __global double *randomPointsB){

int gid = get_global_id(0);
double2 z0 = (float2)(initPointsA[gid], initPointsB[gid]);

double2 z = (float2)(0, 0);

double xtemp = 0;
double oldX = 0;
double oldY = 0;

int stepsTaken = 0;
int stepLimit = 2;

int escaped = 0;
int compt = 0;


for (int i = 0; i < MAXITER; i++){
	compt++;
	
	xtemp = fabs(z.x*z.x-z.y*z.y);
	z.y = 2*z.x*z.y + z0.y;
	z.x = xtemp + z0.x;

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
