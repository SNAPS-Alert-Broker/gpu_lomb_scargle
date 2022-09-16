from multiprocessing import Queue, current_process, managers
import time
import numpy as np
import threading
from typing import Any, List, Tuple, Union
import dataclasses

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


def stop(thread: threading.Thread):
    running.value = False
    thread.join()
    manager.shutdown()


def _run(min_f, max_f, numFreqs, timeout, error=False, lsmode=Mode.GPU, dtype=DType.DOUBLE):
    while running.value:

        dataLock.acquire()

        if (time.time() - startTime.value) > timeout:
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
                    outputData[tid] = sum(d)

                    times = np.append(times, d[0])
                    mags = np.append(mags, d[1])

                    if error:
                        errors = np.append(errors, d[2])

                    objIds += [tid for _ in range(len(d[0]))]

                derived: List[GPULSResult] = lombscargle(
                    objIds, times, mags, min_f, max_f, error, lsmode, magDY=errors, freqToTest=numFreqs, dtype=dtype)

                for objData in derived:

                    # Can't send custom objects
                    # Will recreate object later
                    outputData[objData.objID] = dataclasses.astuple(objData)

                with finishCond:
                    finishCond.notify_all()

                startTime.value = time.time()
        dataLock.release()


def queueLightCurve(data: Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]], tid: int = None) -> GPULSResult:
    if tid is None:
        tid = hash(current_process())

    dataLock.acquire()

    inputData.put((data, tid))

    dataLock.release()

    with finishCond:
        finishCond.wait()

    tmp: GPULSResult = GPULSResult(*outputData[tid])
    del outputData[tid]

    return tmp


class Collector:

    def __init__(self, minFreq: float, maxFreq: float, nFreqs: int, timeout: float = 1, error: bool = False, mode: Mode = Mode.GPU, dtype: DType = DType.DOUBLE) -> None:
        self.runnerThread: threading.Thread = None
        self.minFreq: float = minFreq
        self.maxFreq: float = maxFreq
        self.nFreqs: int = int(nFreqs)
        self.timeout: float = timeout
        self.error: bool = error
        self.mode: Mode = mode
        self.dtype: DType = dtype

    def __enter__(self):
        self.runnerThread = threading.Thread(
            target=_run, args=(self.minFreq, self.maxFreq, self.nFreqs, self.timeout, self.error, self.mode, self.dtype))
        self.runnerThread.start()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        stop(self.runnerThread)
