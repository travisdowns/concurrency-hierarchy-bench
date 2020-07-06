This benchmark helps illustrates a hierarchy of concurrency costs, as described in [this blog post](https://travisdowns.github.io/blog/2020/07/06/concurrency-costs.html).

It requires a Linux system to build, and has been tested on Ubuntu 19.04 and 20.04. It should work on Windows using Windows Subsystem for Linux (WSL), although I haven't tested it.

## Building

    make

## Running

    ./bench [options]

The list of available options can be obtained by running `./bench --help`. Currently, they are:

~~~
  ./bench {OPTIONS}

    conc-bench: Demonstrate concurrency perforamnce levels

  OPTIONS:

      --help                            Display this help menu
      --force-tsc-calibrate             Force manual TSC calibration loop, even
                                        if cpuid TSC Hz is available
      --no-pin                          Don't try to pin threads to CPU - gives
                                        worse results but works around affinity
                                        issues on TravisCI
      --verbose                         Output more info
      --list                            List the available tests and their
                                        descriptions
      --csv                             Output a csv table instead of the
                                        default
      --progress                        Display progress to stdout
      --algos=[TEST-ID]                 Run only the algorithms in the comma
                                        separated list
      --batch=[BATCH-SIZE]              Make BATCH-SIZE calls to the function
                                        under test in between checks for test
                                        termination
      --trial-time=[TIME-MS]            The time for each trial in ms
      --min-threads=[MIN]               The minimum number of threads to use
      --max-threads=[MAX]               The maximum number of threads to use
      --warmup-ms=[MILLISECONDS]        Warmup milliseconds for each thread
                                        after pinning (default 100)
~~~

## Data Collection and Plotting

You can examine the `scripts/data.sh` script to see how data was collected, and `scripts/all-plots.sh` (all the heavy lifting happens in `plot-bar.py`) to see how the data reshaping and plotting works.
