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
   	const int t_ID = threadIdx.x + blockIdx.y * blockDim.x;


  if (threadIdx.x == 4 && blockIdx.y == 3){
  	// printf("t_Id=",t_ID);
  int py = threadIdx.x;
  int px = blockIdx.y;
  // if(blockIdx.y == 5 && threadIdx.x == 3){
  	printf("threadIdx.x=%d, blockIdx.y=%d, blockDim.x=%d, t_ID=%d, px=%d, py=%d\n",threadIdx.x, blockIdx.y, blockDim.x, t_ID, px, py);
  	trap1[t_ID] = compute_pixel(trap, omega, py, px, w);
  	// printf("trap[%d]=%f\n", t_ID, trap[t_ID]);
  	delta=fabs(trap1[t_ID]-trap[t_ID]);
	
}

/*
 * rb
 */
__global__ void rb(float *trap, int h, int w, float omega, float epsilon, int iter, float *delta) {
	// implement me
}

/*
 * dbuf
 */
__global__ void dbuf(float *trap, int h, int w, float omega, float epsilon, int iter, float *delta) {
	//implement me
}
