#!/usr/bin/env python3

"""FrostTrap using numpy arrays. 

The source code demonstrates several algorithms for solving the heat distribution problem. 

- frost_iter_jacobi_slices uses numpy slices and array operations, which is significantly faster
  than manual looping in Python. 
- Several algorithms that use C-style loops in Python, which is slow but demonstrates the algorithms. 
  - frost_iter_jacobi_loops     - Jacobi method
  - frost_iter_jacobi_rb_loops  - Jacobi with a red-black scheme
  - frost_iter_gauss_loops      - Gauss-Seidel method 
  - frost_iter_sor_loops        - SOR (Successive Overrelaxation) method


frost_trap() is used to set up and run the execution. You can specify
whether you want visualizations and with of the above step functions
you want. See the bottom of the file for examples.

TODO: 
- a safer termination check might be to look at the max change over all cells instead of the sum of changes. 
  The current scheme might terminate too early for large enough arrays with small enough variations between core and
  edge values. 
"""

import time
import numpy as np
import matplotlib.pyplot as plt

#default is double precision float. Specify -f32 on the command line to change to single precision. 
FP_TYPE = 'float64'  


class Visualizer:
    """Support object to simplify visualization of the computed FrostTrap. As a bonus, it also keeps track 
    of the iterations in the main loop.
    """
    def __init__(self, arr, freq=100, do_vis=True):
        self.arr = arr
        self.freq = freq
        self.do_vis = do_vis
        self.count = 0
        if not do_vis:
            # Trick to disable visualization (use the nop methods instead)
            self.update = self._update_nop
            self.draw   = self._draw_nop
        else:
            # Prepare visualization with matplotlib
            self.fig, self.ax = plt.subplots()
            
    def _update_nop(self):
        self.count+=1 # not quite a nop - it still keeps track of iters
        
    def update(self):
        """Updates iteration counter and draws update if we're at iter % freq == 0"""
        if (self.count % self.freq == 0):
            self.draw()
        self.count += 1
        
    def draw(self):
        plt.imshow(self.arr)
        plt.pause(0.001)
        
    def _draw_nop(self):
        pass


def create_trap(sx, sy):
    """Creates an initial array with zeroes in the middle and -237.15 along left,right and bottowm rows, 
    and 40.0 along the top row.
    """
    arr = np.zeros((sy, sx), dtype=FP_TYPE)
    arr[:,  0]  = -237.15       # left side
    arr[:, -1]  = -237.15       # right side
    arr[0, :]   =   40.00       # top
    arr[-1, :]  = -237.15       # bottom
    return arr


def frost_iter_jacobi_slices(arr, omega=0.8):
    """Runs one iteration of a Jacobi solver over the provided array. 
    The array is updated with the new values. 

    Omega is ignored as it's not a SOR method. 

    Returns the delta (sum of the absolute differences between iterations).

    This makes use of array operations and slices in numpy (running
    operations on entire arrays instead of cell by cell).  
    """
    # The code is slightly verbose to make it easier to spot what is happening. 
    # NB: no optimization has been done here. 
    l = arr[1:-1, 0:-2]          # cells left one
    r = arr[1:-1, 2:  ]          # cells right one
    t = arr[0:-2, 1:-1]          # cells up one
    b = arr[2:  , 1:-1]          # cells down one
    # compute new core cells (not the static sides). 
    nc = (1/4.0) * (l+r+t+b)

    # compute delta (sum of absolute differences between current and new core values) 
    delta = np.sum(abs(arr[1:-1, 1:-1] - nc))

    # update core cells
    arr[1:-1,1:-1] = nc          
    return delta


def frost_iter_jacobi_loops(arr, omega=0.8):
    """Runs one iteration of the Jacobi iterative method over the cells in the arr array. 
    It updates the cells (not the edge cells) as an average of the immediate neighbours 
    up,down,left,and right.
    Omega is ignored. 
    The results are written back to arr. 
    The function returns "delta", which is the sum of the absolute differences between the old and new values.
    """
    # This is not the most efficient way of doing this, but keeps the code simple as we make a
    # copy of arr (src) that we can read from and we write the results directly to arr.
    # To avoid the overhead of creating a copy every time, we could use a double bufring technique and alternate
    # between two arrays (read from one and write to the other, then the other way around). 
    src = np.copy(arr)
    delta = 0.0
    for y in range(1, arr.shape[0]-1):
        for x in range(1, arr.shape[1]-1):
            nc = (src[y-1][x  ] +
                  src[y+1][x  ] +
                  src[y  ][x-1] +
                  src[y  ][x+1]) / 4.0
            delta += abs(nc - arr[y][x])
            arr[y][x] = nc
    return delta


def frost_iter_jacobi_rb_loops(arr, omega=0.8):
    """Runs one iteration of the Jacobi iterative method over the cells in the arr array, 
    using a red-black scheme. 
    It updates the cells (not the edge cells) as an average of the immediate neighbours 
    up,down,left,and right.
    Omega is ignored. 
    The results are written back to arr. 
    The function returns "delta", which is the sum of the absolute differences between the old and new values.
    """
    delta = 0.0
    # Update the cells in two phases: 
    # rb = 0: red phase, rb = 1: black phase. 
    for rb in range(2):
        for y in range(1, arr.shape[0]-1):
            # Starting x offset, influenced by row and rb phase.  
            # This selects which cells to update in this row.
            offset = (y + rb) % 2
            for x in range(1 + offset, arr.shape[1]-1, 2):
                nc = (arr[y-1][x  ] +
                      arr[y+1][x  ] +
                      arr[y  ][x-1] +
                      arr[y  ][x+1]) / 4.0
                delta += abs(nc - arr[y][x])
                arr[y][x] = nc
    return delta


def frost_iter_gauss_loops(arr, omega=0.8):
    """Runs one iteration of the Gauss-Seidel method over the cells in the array. 
    It updates the core cells (not the edge cells) to the average of the immediate 
    neighbours up,down,left,right.
    Omega is ignored. 
    The function returns "delta", which is the sum of the absolute differences between the old and new values.
    """
    delta = 0.0
    for y in range(1, arr.shape[0]-1):
        for x in range(1, arr.shape[1]-1):
            nc = (arr[y-1][x  ] + 
                  arr[y+1][x  ] +
                  arr[y  ][x-1] +
                  arr[y  ][x+1]) / 4.0 
            delta += abs(nc - arr[y][x])
            arr[y][x] = nc
    return delta


def frost_iter_sor_loops(arr, omega=0.8):
    """Runs one iteration of the SOR (Successive Overrelaxation) method over the cells in the array. 
    It updates the core cells (not the edge cells) as a weighted average of current cell and immediate 
    neighbours up,down,left,right.
    Omega is a weight that adjusts how much we attribute to neighbours compared to the center cell. 
    This implements the Succesive Overrelaxation (SOR) method of updating the cells. 
    The function returns "delta", which is the sum of the absolute differences between the old and new values.
    """
    delta = 0.0
    for y in range(1, arr.shape[0]-1):
        for x in range(1, arr.shape[1]-1):
            nc = (omega/4.0)   * (arr[y-1][x] + arr[y+1][x] + arr[y][x-1] + arr[y][x+1]) + \
                 (1.0 - omega) * arr[y][x]
            delta += abs(nc - arr[y][x])
            arr[y][x] = nc
    return delta



def frost_trap(sx, sy, do_vis=False, f_iter=frost_iter_jacobi_slices, vfreq=100):
    """Creates and computes the frost trap. 
    Specify dimensions of trap with sx and sy.
    It can optionally visualize the results while running if you set do_vis to True. 
    Specify f_iter to select the function that computes the next step/iteration in the algorithm. 
    """
    print("Creating trap.")
    trap = create_trap(sx, sy)
    vis = Visualizer(trap, freq=vfreq, do_vis=do_vis)  # Initialize visualization object.
    epsilon = 0.001 * sx * sy                          # Borrowed from the c code. Used to check when we should terminate. 
    delta = epsilon + 10                               # Initial value that doesn't immediately terminate the loop.
    
    vis.draw()
    print(f"Got trap of dimension {trap.shape} and epsilon {epsilon}. Now iterating.")
    t1 = time.time()
    while delta > epsilon:
        delta = f_iter(trap)
        vis.update()
    t2 = time.time()
    vis.draw()
    print(f"Computing took {t2-t1} seconds with final delta {delta} with total {vis.count} iters.")
    print()
    return trap


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", help="Turn on visualisation", action="store_true")
    parser.add_argument("-f32", help="Set datatype to be 32-bit/single precision float", action="store_true")
    args = parser.parse_args()

    if args.f32:
        # Change to single precision float instead of double (default). 
        FP_TYPE="float32"
    
    do_vis = args.vis
    print("Float type: ", FP_TYPE, " (using ", np.zeros(1, dtype=FP_TYPE).nbytes, " bytes per cell).")
    
    print("------ Running with numpy array slice operations. Jacobi method. ---- ")
    frost_trap(150, 150, f_iter = frost_iter_jacobi_slices, do_vis=do_vis, vfreq=400) # 

    # Use smaller arrays to compensate for the slower c-style loops.
    print("------ Running with Jacobi and C style loops (slower) and smaller arrays to compensate. ---- ")
    frost_trap(40, 40, f_iter = frost_iter_jacobi_loops, do_vis = do_vis)
    
    print("------ Running with Jacobi and C style loops with red-black scheme. ---- ")
    frost_trap(40, 40, f_iter = frost_iter_jacobi_rb_loops, do_vis = do_vis)
    
    print("------ Running with Gauss-Seidel method and C style loops. ---- ")
    frost_trap(40, 40, f_iter = frost_iter_gauss_loops, do_vis = do_vis)
    
    print("------ Running with SOR and C style loops (slower) and smaller arrays to compensate. ---- ")
    frost_trap(40, 40, f_iter = frost_iter_sor_loops, do_vis = do_vis)
