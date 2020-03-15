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
//float2 z = (float2)(0.1, 0.2);

float xtemp = 0;
float x = z.x;
float y = z.y;

int escaped = 0;
int compt = 0;

// x_n+1 = r*x_n*(1-x)
// (r*x).a = r.a*x.a - r.b*x.b
// (r*x).b = r.y*x.a + r.x*x.b
// z_n+1.a = 




for (int i = 0; i < MAXITER; i++){
	compt++;
	/*Buddhabrot equation :|
	xtemp = x*x - y*y + r.x;
	y = 2*x*y + r.y;
	x = xtemp;
	*/

	//BuddhaLog equation :)
	
	/*
	rxa = r.x*x - r.y * y;
	rxb = r.y*y + r.x * y;
	xtemp = rxa*(1-x) - rxb*y;
	y = rxb*(1-x) + rxa*y;
	x = xtemp;
i (a d - b c^2 + b c - b d^2) - a c^2 + a c - a d^2 - b d
	*/
	xtemp = r.x * x*x + r.x * x - r.x * y*y - r.y * y;
	y = r.x * y - r.y * x * x + r.y * x - r.y*y*y;
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
