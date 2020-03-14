__kernel void buddhaTraj(__global const float *initPointsA,
			 __global const float *initPointsB, 
		  	 __global float *trajsA,
			 __global float *trajsB,
			 
			 const int MAXITER, 
		 	 const int NPOINTS,
			 __global float *randomPointsA, 
			 __global float *randomPointsB){

int gid = get_global_id(0);
float2 r = (float2)(initPointsA[gid], initPointsB[gid]);
float2 z = (float2)(randomPointsA[gid], randomPointsB[gid]);

float xtemp = 0;
float x = 0.0;
float y = 0.0;

int escaped = 0;
int compt = 0;

for (int i = 0; i < MAXITER; i++){
	compt++;
	//BuddhaLog equation :)
	//a c + i b c - a c^2 - i b c^2 + i a d - b d - 2 i a c d + 2 b c d + a d^2 + i b d^2
	xtemp = r.x*z.x - r.x*z.x*z.x - r.y*z.y - 2*r.y*z.x*z.y + r.x*z.y*z.y;
	z.y = r.y*z.x - r.y*z.x*z.x + r.x*z.y - 2*r.x*z.x*z.y + r.y*z.y*z.y;	
	z.x = xtemp;
	trajsA[MAXITER*gid + i] =z.x;
	trajsB[MAXITER*gid + i] =z.y;
	
	if (z.x*z.x+z.y*z.y > 4 ){
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
