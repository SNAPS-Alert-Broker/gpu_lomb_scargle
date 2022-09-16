# Lomb-Scargle Algorithm on the GPU with improved python interface

Original Code authors: Mike Gowanlock and Brian Donnelly

Modified by Daniel Kramer

# Paper

If you use this software, please cite our paper below.

M. Gowanlock, D. Kramer, D.E. Trilling, N.R. Butler, B. Donnelly (2021)\
*Fast period searches using the Lomb–Scargle algorithm on Graphics Processing Units for large datasets and real-time applications*\
Astronomy and Computing, Elsevier\
https://doi.org/10.1016/j.ascom.2021.100472

## Install
To install, clone and run ```make```. If your machine has multiple GPUs or CPU cores, update the following lines to your machines values (If you do not intend ot use the CPU version, then you can ignore the NCPU line). 
```
NGPU=1
NCPU=16
``` 
If these values change, you'll have to edit them and reinstall. 

## Python Interface
There are two ways of interfacing with the CUDA coda. 

### lombscargle
The main way to interface is with the `lombscargle` method. This method can work with a single object or multiple objects at the same time. It takes in an array of object IDs, times, magnitudes, the minimum/maximum frequency, the number of frequencies, error values, whether to run  on the GPU or CPU, and the data type of the values.

- The object IDs, times, and magnitudes all need to be 1D and the same length. The object IDs need to be broadcasted over the times/magnitudes.

- The minimum/maximum frequency are NOT angular frequency

- If a value of -1 is used for the number of frequencies, the number of frequencies will automatically be computed. 

- To choose to run on the GPU/CPU and the data type, you must use the provided enums in `gpuls.Mode` and `gpuls.DType`, respectively

### Collector
In order to facilitate mulitproccesing, we've created an interface that allows for multiple threads/processes Lomb-Scargle needs to be computed simultaneously. The Collector interface contains two parts, a context manager that periodically runs GPULS and a method to add a light curve to the queue. 

The `Collector` context manager manages the thread that periodically runs GPULS. It takes in the same min/max frequency and number of frequencies as the lombscargle interface. It also (optionally) requires a `timeout` value, which is how often GPULS runs.

The `queueLightCurve` method queues a light curve for GPULS. It takes a list of 2 numpy arrays in the form of [times, mags]. It returns a tuple of the period corresponding to the highest power and the periodogram.

Example: 
```python
with Collector(1/100, 1/1, int(1e5)):
  per, pgram = queueLightCurve([times, mags])
```





