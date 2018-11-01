// index trap as a 2D-array 
__device__ inline float idx(float *trap, int x, int y, int w) {
	return trap[(y*w)+x];
} 

// compute pixel value
__device__ inline float compute_pixel(float *trap, float omega, int x, int y, int w) {
	return (omega / 4.0) * (idx(trap, x-1, y  , w)
						 +  idx(trap, x+1, y  , w)
						 +  idx(trap, x  , y-1, w)
						 +  idx(trap, x  , y+1, w))
						 + (1.0 - omega) * idx(trap, x,y,w);
}

/*
 * simple
 */
__global__ void simple(float *trap, int h, int w, float omega, float epsilon, int iter, float *delta) {
	// implement me
        const int tx = threadIdx.x + (blockIdx.x * blockDim.x);

	// Calculating the X and Y pixel coordinates, wouldn't need to do this if the kernel was invoked with a 2D grid of threads
	int ax = tx % 300;
	int ay = (tx - ax) / 300;

	
	int it = 0;
	while (it < iter)
	for (y =0;y<h;y++)
	      {
	        for (x =0;x<w;x++)
		{  
		  float old[] = trap[tx];
                  float newvalue[]  =  compute_pixel(trap, omega, x[ax], y[ay], w);
                  //printf("new is %f", newvalue);
                  trap[tx] = newvalue;
                 *delta += fabs(old - newvalue);
		}  
	}
	
}

/*
 * rbshi 
 */
__global__ void rb(float *trap, int h, int w, float omega, float epsilon, int iter, float *delta) {
	// implement me
	 printf ("hi from rb ");
}

/*
 * dbuf
 */
__global__ void dbuf(float *trap, int h, int w, float omega, float epsilon, int iter, float *delta) {
	//implement me
	//printf ("hi from dbuf ");
}
