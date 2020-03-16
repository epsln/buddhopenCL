#include "include/math_utils.h"

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x100000)

void mandelIterOpenCL(char kernelFilename[256], float *initialPointsA,float *initialPointsB, int NPOINTS, int MAXITER, int RESX, int RESY, long long int **histogram){
	// Allocate memories 
	//We can't pass a 2D array to a kernel, so we'll flatten a 2d array to 1D
	float *trajectoriesA = (float*)malloc(sizeof(float)*NPOINTS*MAXITER);
	float *trajectoriesB = (float*)malloc(sizeof(float)*NPOINTS*MAXITER);
	float *randomPointsA = (float*)malloc(sizeof(float)*NPOINTS );
	float *randomPointsB = (float*)malloc(sizeof(float)*NPOINTS );

	for (int i = 0; i < NPOINTS; i++){
		float a = float_rand(-2.0, 2.0);
		float b = float_rand(-2.0, 2.0);

		randomPointsA[i] = a;
		randomPointsB[i] = b;
	}

	// Initialize values for array members.
	// Load kernel from file vecAddKernel.cl

	FILE *kernelFile;
	char *kernelSource;
	size_t kernelSize;

	kernelFile = fopen(kernelFilename, "r");

	if (!kernelFile) {

		fprintf(stderr, "No file named %s was found !\n", kernelFilename);
		exit(-1);

	}
	kernelSource = (char*)malloc(MAX_SOURCE_SIZE);
	kernelSize = fread(kernelSource, 1, MAX_SOURCE_SIZE, kernelFile);
	fclose(kernelFile);

	// Getting platform and device information
	cl_platform_id platformId = NULL;
	cl_device_id deviceID = NULL;
	cl_uint retNumDevices;
	cl_uint retNumPlatforms;
	cl_int ret = clGetPlatformIDs(1, &platformId, &retNumPlatforms);
	ret = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceID, &retNumDevices);

	// Creating context.
	cl_context context = clCreateContext(NULL, 1, &deviceID, NULL, NULL,  &ret);

	// Creating command queue
	cl_command_queue commandQueue = clCreateCommandQueue(context, deviceID, 0, &ret);

	// Memory buffers for each array
	cl_mem aMemObj = clCreateBuffer(context, CL_MEM_READ_ONLY,  NPOINTS * sizeof(float), NULL, &ret);
	cl_mem bMemObj = clCreateBuffer(context, CL_MEM_READ_ONLY,  NPOINTS * sizeof(float), NULL, &ret);

	cl_mem cMemObj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, MAXITER * NPOINTS * sizeof(float), NULL, &ret);
	cl_mem dMemObj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, MAXITER * NPOINTS * sizeof(float), NULL, &ret);

	cl_mem eMemObj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, NPOINTS * sizeof(float), NULL, &ret);
	cl_mem fMemObj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, NPOINTS * sizeof(float), NULL, &ret);

	// Copy lists to memory buffers
	ret = clEnqueueWriteBuffer(commandQueue, aMemObj, CL_TRUE, 0, NPOINTS * sizeof(float), initialPointsA, 0, NULL, NULL);;
	ret = clEnqueueWriteBuffer(commandQueue, bMemObj, CL_TRUE, 0, NPOINTS * sizeof(float), initialPointsB, 0, NULL, NULL);;
	ret = clEnqueueWriteBuffer(commandQueue, eMemObj, CL_TRUE, 0, NPOINTS * sizeof(float), randomPointsA, 0, NULL, NULL);;
	ret = clEnqueueWriteBuffer(commandQueue, fMemObj, CL_TRUE, 0, NPOINTS * sizeof(float), randomPointsB, 0, NULL, NULL);;

	// Create program from kernel source
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, (const size_t *)&kernelSize, &ret);	

	// Build program
	ret = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);

	size_t len = 0;
	ret = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
	char *buffer = calloc(len, sizeof(char));
	ret = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
	if(buffer[0] != '\n' && buffer[0] != '\0'){
		printf("\nOPENCL Compilation Error ! Check logs !\n");
		printf("%s", buffer);
		exit(-1);
	}

	// Create kernel
	cl_kernel kernel = clCreateKernel(program, "buddhaTraj", &ret);

	int maxiter = MAXITER;
	int npoints = npoints;
	// Set arguments for kernel
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&aMemObj);	
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bMemObj);	
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&cMemObj);	
	ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&dMemObj);	

	ret = clSetKernelArg(kernel, 4, sizeof(cl_int), &maxiter);	
	ret = clSetKernelArg(kernel, 5, sizeof(cl_int), &npoints);	

	ret = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&eMemObj);	
	ret = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&fMemObj);	


	// Execute the kernel

	size_t globalItemSize = NPOINTS;
	size_t localItemSize = 1024; // globalItemSize has to be a multiple of localItemSize. 
	ret = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, NULL);	

	// Read from device back to host.
	ret = clEnqueueReadBuffer(commandQueue, cMemObj, CL_TRUE, 0, MAXITER * NPOINTS * sizeof(float), trajectoriesA, 0, NULL, NULL);
	ret = clEnqueueReadBuffer(commandQueue, dMemObj, CL_TRUE, 0, MAXITER * NPOINTS * sizeof(float), trajectoriesB, 0, NULL, NULL);

	// Write result

	for(int i = 0; i < MAXITER*NPOINTS; i++){
		if (trajectoriesA[i] != 0.0 || trajectoriesB[i] != 0.0){
			break;
		}

		if (i == MAXITER*NPOINTS - 1){
			printf("Trajectories are empty !!!!\n");
			exit(-3);
		}

	}

	int x, y;
	for(int i = 0; i < MAXITER*NPOINTS; i++){
		//printf("[%f %f]\n", trajectoriesA[i], trajectoriesB[i]);
		x = (int)map(trajectoriesA[i], -2, 2, 0, RESX);	
		y = (int)map(trajectoriesB[i], -2, 2, 0, RESY);	
		//	if (x != -2000 && x != -24500){
		//		printf("x: %d y: %d\n", x, y);
		//	}
		if(x >= 0 && x < RESX && y >= 0 && y < RESY ){
			histogram[x][y]++;
		}
	}

	ret = clFlush(commandQueue);
	ret = clFinish(commandQueue);
	ret = clReleaseCommandQueue(commandQueue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(aMemObj);
	ret = clReleaseMemObject(bMemObj);
	ret = clReleaseMemObject(cMemObj);
	ret = clReleaseMemObject(dMemObj);
	ret = clReleaseMemObject(eMemObj);
	ret = clReleaseMemObject(fMemObj);
	ret = clReleaseContext(context);

	free(trajectoriesA);
	free(trajectoriesB);
	free(randomPointsA);
	free(randomPointsB);
}
