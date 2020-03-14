#ifndef mandel 
#define mandel 

void MandelIter(float zx, float zy, int **collisionsArray, int iternum, int RESX, int RESY);
void mandelIterOpenCL(char kernelFilename[256], float *initialPointsA,float *initialPointsB, int NPOINTS, int MAXITER, int RESX, int RESY, long long int **histogram);
#endif
