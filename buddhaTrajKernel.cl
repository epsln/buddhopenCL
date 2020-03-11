typedef float2 cfloat;
#define I ((cfloat)(0.0, 1.0))

inline float  real(cfloat a){
     return a.x;
}
inline float  imag(cfloat a){
     return a.y;
}

inline cfloat  cmult(cfloat a, cfloat b){
    return (cfloat)( a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

__kernel void buddhaTraj(__global const float *initPointsA,
			 __global const float *initPointsB, 
		  	 __global float *trajsA,
			 __global float *trajsB,
			 
			 const int MAXITER, 
		 	 const int NPOINTS,
			 __global float *randomPointsA, 
			 __global float *randomPointsB){

int gid = get_global_id(0);
cfloat r = (cfloat)(initPointsA[gid], initPointsB[gid]);
//cfloat z = (cfloat)(randomPointsA[gid], randomPointsB[gid]);
cfloat z = (cfloat)(0.1, 0.2);

cfloat rtemp = (cfloat)(0,0);
cfloat ztemp = (cfloat)(0,0);
float x = 0;
float y = 0;

int escaped = 0;
int compt = 0;

// x_n+1 = r*x_n*(1-x)
// (r*x).a = r.a*x.a - r.b*x.b
// (r*x).b = r.y*x.a + r.x*x.b
// z_n+1.a = 




for (int i = 0; i < MAXITER; i++){
	compt++;
	/*Buddhabrot equation :|
	xtemp = x*x - y*y + zx;
	y = 2*x*y + zy;
	x = xtemp;
	*/

	//BuddhaLog equation :)
	/*
	rxa = rx*x - ry * y;
	rxb = ry*y + rx * y;
	xtemp = rxa*(1-x) - rxb*y;
	y = rxb*(1-x) + rxa*y;
	x = xtemp;
	*/


	rtemp = cmult(r, z);
	ztemp = (cfloat)(1-real(z), imag(z));
	z = cmult(rtemp, ztemp);
	x = real(z);
	y = imag(z);
	
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
