from multiprocessing import Queue, current_process, managers
import time
import numpy as np
import threading
from typing import List, Tuple

from .lombscarglegpu import lombscargle, GPULSResult
from .mode import Mode
from .dtype import DType


manager = managers.SyncManager()
manager.start()
inputData = manager.Queue()
outputData = manager.dict()
dataLock = manager.Lock()
finishLock = manager.Lock()
finishCond = manager.Condition(finishLock)
startTime = manager.Value('f', time.time())
running = manager.Value('b', True)


def stop(thread):
    running.value = False
    thread.join()
    manager.shutdown()


def _run(min_f, max_f, numFreqs, timeout, error=False, lsmode=Mode.GPU, dtype=DType.DOUBLE):
    while running.value:

        dataLock.acquire()

        if (time.time() - startTime.value) > timeout:
            data = {}
            while not inputData.empty():
                tmp = inputData.get_nowait()
                data[tmp[1]] = tmp[0]
            if len(data) > 0:
                times = np.array([], dtype=float)
                mags = np.array([], dtype=float)
                objIds = []
                for tid, d in data.items():
                    outputData[tid] = sum(d)

                    times = np.append(times, d[0])
                    mags = np.append(mags, d[1])
                    objIds += [tid for _ in range(len(d[0]))]

                derived: List[GPULSResult] = lombscargle(
                    objIds, times, mags, min_f, max_f, error, lsmode, freqToTest=numFreqs, dtype=dtype)
                # print(derived, flush=True)
                for objData in derived:
                    # print(objData.objID, flush=True)
                    outputData[objData.objID] = (objData.period, objData.pgram)
                    # print(objData.objID, flush=True)
                with finishCond:
                    finishCond.notify_all()
                startTime.value = time.time()
        dataLock.release()


def queueLightCurve(data: Tuple[np.ndarray,np.ndarray], tid: int = None) -> GPULSResult:
    if tid is None:
        tid = hash(current_process())

    dataLock.acquire()

    inputData.put((data, tid))

    dataLock.release()

    with finishCond:
        finishCond.wait()

    tmp: GPULSResult = outputData[tid]
    del outputData[tid]

    return tmp


class Collector:

    def __init__(self, minFreq: float, maxFreq: float, nFreqs: int, timeout:float=1) -> None:
        self.runnerThread = None
        self.minFreq = minFreq
        self.maxFreq = maxFreq
        self.nFreqs = int(nFreqs)
        self.timeout = timeout

    def __enter__(self):
        self.runnerThread = threading.Thread(
            target=_run, args=(self.minFreq, self.maxFreq, self.nFreqs, self.timeout))
        self.runnerThread.start()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        stop(self.runnerThread)
