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
  printf ("hi from simple ");
  
	
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
