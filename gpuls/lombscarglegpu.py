import os
from typing import Any, List, Tuple
import numpy.ctypeslib as npct
from ctypes import *
from scipy.signal import peak_widths
import numpy as np
import dataclasses
from collections.abc import Iterable
from .mode import Mode
from .dtype import DType
from .lazydict import LazyDict


# Create variables that define C interface
array_1d_double = npct.ndpointer(dtype=c_double, ndim=1, flags='CONTIGUOUS')
array_1d_float = npct.ndpointer(dtype=c_float, ndim=1, flags='CONTIGUOUS')
array_1d_unsigned = npct.ndpointer(dtype=c_uint, ndim=1, flags='CONTIGUOUS')

#              Error, DType
LS_ARGTYPE = {(False, DType.FLOAT): [array_1d_unsigned, array_1d_float, array_1d_float, array_1d_float, c_uint, c_double, c_double, c_uint, c_int, array_1d_float],
              (False, DType.DOUBLE): [array_1d_unsigned, array_1d_double, array_1d_double, array_1d_double, c_uint, c_double, c_double, c_uint, c_int, array_1d_double],
              (True, DType.FLOAT): [array_1d_unsigned, array_1d_float, array_1d_float, array_1d_float, c_uint, c_double, c_double, c_uint, c_int, array_1d_float],
              (True, DType.DOUBLE): [array_1d_unsigned, array_1d_double, array_1d_double, array_1d_double, c_uint, c_double, c_double, c_uint, c_int, array_1d_double]}


@dataclasses.dataclass
class GPULSResult:
    objID: Any
    period: float
    error: float
    pgram: np.ndarray = None
    maskPeriod: float = None
    maskError: float = None



def loadLib(file, path):
    return npct.load_library(file, path)


lib_path = os.path.join(os.path.dirname(__file__), 'cuda')
tmp = {}
for i in range(1,4+1):

    tmp.update({(False, DType.FLOAT, i): (loadLib, (f'gpu{i}/libpylsnoerrorfloat.so', lib_path)),
                (False, DType.DOUBLE, i): (loadLib, (f'gpu{i}/libpylsnoerrordouble.so', lib_path)),
                (True, DType.FLOAT, i): (loadLib, (f'gpu{i}/libpylserrorfloat.so', lib_path)),
                (True, DType.DOUBLE, i): (loadLib, (f'gpu{i}/libpylserrordouble.so', lib_path))})

LS_SO_FILES = LazyDict(tmp)



def convert_type(in_array, new_dtype):

    ret_array = in_array

    if not isinstance(in_array, np.ndarray):
        ret_array = np.array(in_array, dtype=new_dtype)

    elif in_array.dtype != new_dtype:
        ret_array = np.array(ret_array, dtype=new_dtype)

    if not ret_array.flags['C_CONTIGUOUS']:
        ret_array = np.ascontiguousarray(ret_array)

    return ret_array


def computeIndexRangesForEachObject(objId):
    start_index_arr = []
    end_index_arr = []
    unique_obj_ids_arr = []
    lastId = objId[0]
    unique_obj_ids_arr.append(objId[0])

    start_index_arr.append(0)
    for x in range(0, len(objId)):
        if (objId[x] != lastId):
            end_index_arr.append(x-1)
            start_index_arr.append(x)
            lastId = objId[x]
            # update the list of unique object ids
            unique_obj_ids_arr.append(objId[x])

    # last one needs to be populated
    end_index_arr.append(len(objId)-1)

    start_index_arr = np.asarray(start_index_arr, dtype=int)
    end_index_arr = np.asarray(end_index_arr, dtype=int)
    unique_obj_ids_arr = unique_obj_ids_arr

    return start_index_arr, end_index_arr, unique_obj_ids_arr


def enumerateObjects(start_index_arr, end_index_arr):
    """Get the objects IDs to ints
    ['a','a','b','b','b','c','c','d']-> [0,0,1,1,1,2,2,3]
    """
    enumObjectId = np.empty(end_index_arr[-1]+1, dtype=c_uint32)
    for i, (s, e) in enumerate(zip(start_index_arr, end_index_arr)):

        enumObjectId[s:e+1] = i

    return enumObjectId


# Use the formulation in Richards et al. 2011
def computeNumFreqAuto(objId, timeX, fmin, fmax):

    start_index_arr, end_index_arr, _ = computeIndexRangesForEachObject(objId)

    timeXLocal = np.asfarray(timeX)

    observing_window_arr = []
    for x in range(0, start_index_arr.size):
        idxStart = start_index_arr[x]
        idxEnd = end_index_arr[x]
        observing_window_arr.append(timeXLocal[idxEnd]-timeXLocal[idxStart])

    observing_window_arr = np.asarray(observing_window_arr, dtype=float)

    maximumObservingWindow = np.max(observing_window_arr)

    deltaf = 0.1/maximumObservingWindow

    num_freqs = (fmax-fmin)/deltaf
    num_freqs = int(num_freqs)
    return num_freqs


# wrapper to enable the verbose option
def lombscargle(objId: List[int], timeX: np.ndarray, magY: np.ndarray, minFreq: float, maxFreq: float, error: bool, mode: Mode, magDY=None, freqToTest: int = -1, dtype: DType = DType.FLOAT, mask:Tuple[Tuple[float, float]] = None, getPgram:bool=True, nGPU:int=1) -> List[GPULSResult]:

    # If the user passed in a list of list/numpy arrays
    if isinstance(timeX[0], Iterable) :
        objId = np.concatenate( ([o]*len(r) for o, r in zip(objId, timeX)) )
        timeX = np.concatenate(timeX)
        magY = np.concatenate(magY)
        if error:
            magDY = np.concatenate(magDY)
        


    return _lombscarglemain(objId, timeX, magY, minFreq, maxFreq, error, mode, magDY=magDY, freqToTest=freqToTest, dtype=dtype, mask=mask, getPgram=getPgram, nGPU=nGPU)


# main L-S function
def _lombscarglemain(objId: List[int], timeX: np.ndarray, magY: np.ndarray, minFreq: float, maxFreq: float, error: bool, mode: Mode, magDY=None, freqToTest: int = -1, dtype: DType = DType.FLOAT, mask:Tuple[Tuple[float, float]] = None, getPgram:bool=True, nGPU:int = 1) -> List[GPULSResult]:

    # store the minimum/maximum frequencies (needed later for period calculation)
    minFreqStandard = minFreq
    maxFreqStandard = maxFreq

    # convert oscillating frequencies into angular
    minFreq = 2.0*np.pi*minFreq
    maxFreq = 2.0*np.pi*maxFreq

    ###############################
    # Check for valid parameters and set verbose mode and generate frequencies for auto mode

    # if the user doesn't specify the number of frequencies
    if (freqToTest <= 0):
        freqToTest = computeNumFreqAuto(
            objId, timeX, minFreqStandard, maxFreqStandard)

    # check which mode to use in the C shared library
    # 1- GPU Batch of Objects Lomb-Scargle")
    # 2- GPU Single Object Lomb-Scargle")
    # 3- None
    # 4- CPU Batch of Objects Lomb-Scargle")
    # 5- CPU Single Object Lomb-Scargle")

    numObjects = np.size(np.unique(objId))
    setmode = 3
    if (mode == Mode.GPU and numObjects > 1):
        setmode = 1
    elif (mode == Mode.GPU and numObjects == 1):
        setmode = 2
    elif (mode == Mode.CPU and numObjects > 1):
        setmode = 4
    elif (mode == Mode.CPU and numObjects == 1):
        setmode = 5

    # check that if the error is true, that magDY is not None (None is the default parameter)
    if (error and magDY is None):
        raise ValueError("error is True but no errors provided")

    # enumerate objId so that we can process objects with non-numeric Ids
    # original objects are stored in ret_uniqueObjectIdsOrdered
    start_index_arr, end_index_arr, ret_uniqueObjectIdsOrdered = computeIndexRangesForEachObject(
        objId)
    objId = enumerateObjects(start_index_arr, end_index_arr)

    # total number of rows in file
    sizeData = len(objId)

    # convert input from lists to numpy arrays
    objId = np.asarray(objId, dtype=int)
    timeX = np.asfarray(timeX)
    magY = np.asfarray(magY)

    conv_dtype = c_float if dtype == DType.FLOAT else c_double

    if error:
        c_magDY = np.asarray(magDY, dtype=conv_dtype)
    # if error is false, we still need to send dummy array to C shared library
    # set all values to 1.0, although we don't use it for anything
    else:
        c_magDY = np.ones(sizeData, dtype=conv_dtype)

    c_objId = convert_type(objId, c_uint)
    c_timeX = convert_type(timeX, conv_dtype)
    c_magY = convert_type(magY, conv_dtype)

    # df = (maxFreq-minFreq)/freqToTest
    dfstandard = (maxFreqStandard-minFreqStandard)/freqToTest

    # Allocate arrays for results
    ret_pgram = np.empty(numObjects*freqToTest, dtype=conv_dtype)
    # load the shared library (either the noerror/error and float/double versions)
    libLombScargle = LS_SO_FILES[(error, dtype, nGPU)]
    libLombScargle.LombScarglePy.argtypes = LS_ARGTYPE[(error, dtype)]
    libLombScargle.LombScarglePy(c_objId, c_timeX, c_magY, c_magDY, c_uint(sizeData), c_double(
        minFreq), c_double(maxFreq), c_uint(freqToTest), c_int(setmode), ret_pgram)
    
    # for convenience, reshape the pgrams as a 2-D array
    ret_pgram = ret_pgram.reshape([numObjects, freqToTest])
    
    #Calculate the periods in the pgram
    periods = 1/np.linspace(minFreq, maxFreq, freqToTest)

    if mask is not None:    
        pgramMask = np.ones(freqToTest, dtype=bool)

        for m in mask:
            pgramMask = pgramMask & ~((periods>m[0]) & (periods<m[1]))
        
        periodsMask = periods[pgramMask]


    ret_periods = np.empty(numObjects)
    ret_peakWidths = np.empty(numObjects)

    if mask is not None:
        ret_mask_periods = np.empty(numObjects)
        ret_mask_peakWidths = np.empty(numObjects)
    else:
        ret_mask_periods = (None for _ in range(numObjects))
        ret_mask_peakWidths = (None for _ in range(numObjects))

    # to compute best periods, work back in regular oscillating frequencies (not angular)
    for x in range(numObjects):

        # Normal pgram
        maxIdx = np.argmax(ret_pgram[x])

        ret_periods[x] = periods[maxIdx]

        res = peak_widths(ret_pgram[x], [maxIdx])
        ret_peakWidths[x] = ret_periods[x] - 1.0 / (minFreqStandard+(dfstandard*(maxIdx - (res[0][0] / 2)))) 
    
        #Masked pgram
        if mask is not None:
            maxIdx = np.argmax(ret_pgram[x][pgramMask])

            ret_mask_periods[x] = periodsMask[maxIdx]

            res = peak_widths(ret_pgram[x][pgramMask], [maxIdx])
            ret_mask_peakWidths[x] = ret_mask_periods[x] - 1.0 / (minFreqStandard+(dfstandard*(maxIdx - (res[0][0] / 2)))) 

    if not getPgram:
        
        ret_pgram = (None for _ in range(numObjects))

    ret: List[GPULSResult] = [GPULSResult(*info) 
                              for info in zip(ret_uniqueObjectIdsOrdered, ret_periods, ret_peakWidths, ret_pgram, ret_mask_periods, ret_mask_peakWidths)]

    return ret
