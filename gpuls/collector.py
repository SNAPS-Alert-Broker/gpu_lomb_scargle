from multiprocessing import Queue, current_process, managers, Manager, Process, RawArray
import time
import numpy as np
import threading
from typing import Any, List, Tuple, Union
import dataclasses

from .lombscarglegpu import lombscargle, GPULSResult
from .mode import Mode
from .dtype import DType


if 'SharedMemoryManager' in managers.__all__ and False:
    manager = managers.SharedMemoryManager()    
else:
    #manager = managers.SyncManager()
    manager = Manager()

#manager.start()
inputData = manager.Queue()
outputData = manager.dict()
dataLock = manager.Lock()
finishLock = manager.Lock()
finishCond = manager.Condition(finishLock)
running = manager.Value('b', True)


nFreq = int(5e6)
maxProc = 64
pgramOut = RawArray('d', nFreq*maxProc)
pgramOut_np = np.frombuffer(pgramOut, dtype=np.float64).reshape((maxProc, nFreq))

def stop(thread: threading.Thread):
    running.value = False
    thread.join()
    manager.shutdown()


def _run(min_f, max_f, numFreqs, timeout ,error=False, lsmode=Mode.GPU, dtype=DType.DOUBLE, getPgram:bool=False, nGPU:int=1, mask=None):
    startTime = time.time()
    while running.value:
        dataLock.acquire()

        if (time.time() - startTime) > timeout:
            data: dict = {}
            while not inputData.empty():
                tmp: Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray,
                                                                np.ndarray, np.ndarray]] = inputData.get_nowait()
                data[tmp[1]] = tmp[0]
            if len(data) > 0:
                times: np.ndarray = np.array([], dtype=float)
                mags: np.ndarray = np.array([], dtype=float)
                errors: np.ndarray = None
                if error:
                    errors = np.array([], dtype=float)

                objIds: List[Any] = []
                for tid, d in data.items():

                    times = np.append(times, d[0])
                    mags = np.append(mags, d[1])

                    if error:
                        errors = np.append(errors, d[2])

                    objIds += [tid for _ in range(len(d[0]))]
                derived: List[GPULSResult] = lombscargle(
                    objIds, times, mags, min_f, max_f, error, lsmode, magDY=errors, freqToTest=numFreqs, dtype=dtype, getPgram=getPgram, nGPU=nGPU, mask=mask)

                newData = {}
                for i, objData in enumerate(derived):
                    pgramOut_np[i] = objData.pgram

                    # Can't send custom objects
                    # Will recreate object later
                    newData[objData.objID] = (objData.objID, objData.period, objData.error, i, objData.maskPeriod, objData.maskError)
                outputData.update(newData)
                 
                with finishCond:
                    finishCond.notify_all()

                startTime = time.time()
        dataLock.release()


def queueLightCurve(data: Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]], tid: int = None) -> GPULSResult:
    
    if tid is None:
        tid = hash(current_process())

    dataLock.acquire()
    inputData.put((data, tid))

    dataLock.release()

    with finishCond:
        finishCond.wait()
    outtmp = outputData[tid]
    
    tmp: GPULSResult = GPULSResult(outtmp[0], outtmp[1], outtmp[2], np.array(pgramOut_np[outtmp[3]], copy=True), outtmp[4], outtmp[5])
    del outputData[tid]
    return tmp


class Collector:

    def __init__(self, minFreq: float, maxFreq: float, nFreqs: int, timeout: float = 1, error: bool = False, mode: Mode = Mode.GPU, dtype: DType = DType.DOUBLE, getPgram=False, nGPU:int=1, mask=None) -> None:
        self.runnerThread: threading.Thread = None
        self.minFreq: float = minFreq
        self.maxFreq: float = maxFreq
        self.nFreqs: int = int(nFreqs)
        self.timeout: float = timeout
        self.error: bool = error
        self.mode: Mode = mode
        self.dtype: DType = dtype
        self.getPgram: bool = getPgram
        self.nGPU: int = nGPU
        self.mask = mask

    def __enter__(self):
        #self.runnerThread = threading.Thread(
        #    target=_run, args=(self.minFreq, self.maxFreq, self.nFreqs, self.timeout, self.error, self.mode, self.dtype))
        self.runnerThread = Process(
            target=_run, args=(self.minFreq, self.maxFreq, self.nFreqs, self.timeout,self.error, self.mode, self.dtype, self.getPgram, self.nGPU, self.mask))
        self.runnerThread.start()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        stop(self.runnerThread)
