#!/usr/bin/env -S python3 -B
"""
FrostTrap - CUDA
INF-3201

To use the nivida profiler your javavm must be set to java-8-jdk
and nvidia-toolkit must be installed. Remember to set the command
line argument <simple|dbuf|rb> -p when you execute this script within nvvp.
"""

import timeit
import argparse
import logging
import json

# import numpy
import numpy as np
import matplotlib.pyplot as plt

# import CUDA
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
from pycuda.compiler import SourceModule


class Visualizer:
    """
    Simple visualizer for the FrostTrap
    """
    def __init__(self, trap, do_vis=False):
        self.trap = trap
        _, self.ax = plt.subplots()
        self.update = self._update if do_vis else lambda: None

    def _update(self):
        """
        fetches the trap from the GPU and draws it.
        """

        plt.imshow(self.trap.get())
        plt.pause(0.0001)
        self.ax.cla()


def create_trap(sx, sy):
    """ creates a trap

    Few nvidia card supports double-precision floating points
    (float64), important to specify single-precision floating
    point (float32).

    Args:
            sx (np.int): the width of the trap
            sy (np.int): the height of the trap

    Return:
            arr (np.ndarry): contains the inital trap values (np.float32)

    """
    arr = np.zeros((sy, sx), dtype=np.float32)
    arr[:,  0] = -237.15       # left side
    arr[:, -1] = -237.15       # right side
    arr[0, :] = 40.00       # top
    arr[-1, :] = -237.15       # bottom
    return arr


def frost_trap(kernel_func, args, ver):
    """ Solves the FrostTrap on a GPU with CUDA

    frost_trap creates and allocates the trap data on the GPU, initialize
    graphics and nvidia profiler, runs the kernel function until the delta
    is smaller than a certain limit, finally, it dumps the execution time
    and parameters.

    Args:
            kernel_func : <simple|rb|dbuf> from kernel.cu
            args        : packed command-line options and arguments as a kwarg
    """

    verbose.info("create trap data and init visualizer")
    trap = gpuarray.to_gpu(create_trap(args.width, args.height))

    # delta   - total change
    # epsilon - minimum convergence limit
    epsilon = np.float32(0.001 * args.width * args.height)
    delta = np.array(epsilon)

    # init visualizer
    vis = Visualizer(trap, do_vis=args.graphic)

    # start nvidia profiler
    if args.profile:
        cuda.start_profiler()

    # start
    verbose.info("start")
    t1 = timeit.default_timer()
    
    while delta <= epsilon:
        # experiment with different block and grid sizes
        kernel_func(trap, args.height, args.width, args.omega, epsilon, args.iter, delta, block=(1, 300, 1), grid=(300,1,1))
        vis.update()

    t2 = timeit.default_timer()
    verbose.info("done")
    # stop

    # stop nvidia profiler
    if args.profile:
        cuda.stop_profiler()

    # dump result and parameters as json
    with open(args.result, "a+") as f:
        verbose.info("dumping results to {args.result}")
        args.secs = t2-t1
        json.dump(vars(args), f, default=lambda x: eval(str(x)), indent=2)


if __name__ == "__main__":
    """ sets up the environment

    Parses all the command-line options and arguments, opens the
    CUDA kernel and fetches the given <scheme> and calls the function
    frost_trap with the kernel function and parsed arguments.

    """
    parser = argparse.ArgumentParser()

    # positional argument
    parser.add_argument("scheme", help="possible schemes: <simple|rb|dbuf>", choices=["simple", "rb", "dbuf"])

    # following arguments are true if set
    parser.add_argument("-v", "--verbose",help="extended prints", action="store_true")
    parser.add_argument("-g", "--graphic",help="visualize heatmap", action="store_true")
    parser.add_argument("-p", "--profile",help="cuda profiler", action="store_true")

    # following optional arguments requires a value
    parser.add_argument("-R", "--result", help="result file", action="store", type=str, default="results.json")
    parser.add_argument("-W", "--width", help="width of trap", action="store", type=np.int32, default=np.int32(300))
    parser.add_argument("-H", "--height", help="height of trap", action="store", type=np.int32, default=np.int32(300))
    parser.add_argument("-I", "--iter", help="number of iter", action="store", type=np.int32, default=np.int32(100))
    parser.add_argument("-O", "--omega", help="convergence rate", action="store", type=np.float32, default=np.float32(2.0))

    # parse arguments
    args = parser.parse_args()

    # define verbose and log format - not important
    verbose = logging.getLogger("FrostTrap")
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)8s : %(lineno)3d] %(message)s"))
    verbose.addHandler(_handler)
    verbose.setLevel(logging.INFO)
    verbose.disabled = not args.verbose

    try:
        with open("kernel.cu", "r") as kernel:
            # fetch function <simple|rb|dbuf> from kernel
            kernel_func = SourceModule(kernel.read()).get_function(args.scheme)
        verbose.info(f"scheme: {args.scheme}, {args.width}x{args.height}")

        # clear cache to ensure no previous allocated variables affects our tests
        pycuda.tools.clear_context_caches()
        frost_trap(kernel_func, args, verbose)

    except pycuda._driver.LogicError as e:
        verbose.error(f"{e}: Check if\"{args.scheme}\" exists \"FrostTrap.cu\"")

    except pycuda.driver.CompileError as e:
        verbose.error(f"{e}")
