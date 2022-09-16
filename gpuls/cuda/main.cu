#include <fstream>
#include <istream>
#include <iostream>
#include <string>
#include <string.h>
#include <sstream>
#include <cstdlib>
#include <stdio.h>
#include <random>
#include "omp.h"
#include <algorithm> 
#include <queue>
#include <iomanip>
#include <set>
#include <algorithm>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <cuda_profiler_api.h>
#include <thrust/extrema.h>
#include "structs.h"

#include "kernel.h"

//Only include parameters file if we're not creating the shared library
#ifndef PYTHON
#include "params.h"
#endif


//for printing defines as strings

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

//Error checking GPU calls
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


//function prototypes
void importObjXYData(char * fnamedata, unsigned int * sizeData, unsigned int ** objectId, DTYPE ** timeX, DTYPE ** magY, DTYPE ** magDY);

//CPU L-S Functions:
void lombscarglecpu(bool mode, DTYPE * x, DTYPE * y, const unsigned int sizeData, const unsigned int numFreqs, const DTYPE minFreq, const DTYPE maxFreq, const DTYPE freqStep, DTYPE * pgram);
void lombscarglecpuinnerloop(int iteration, DTYPE * x, DTYPE * y, DTYPE * pgram, DTYPE * freqToTest, const unsigned int sizeData);
void lombscargleCPUOneObject(DTYPE * timeX,  DTYPE * magY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * foundPeriod, DTYPE * foundPower, DTYPE * pgram);
void lombscargleCPUBatch(unsigned int * objectId, DTYPE * timeX,  DTYPE * magY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * sumPeriods, DTYPE * pgram, DTYPE * foundPeriod, DTYPE * foundPower);

void normalizepgram(DTYPE * pgram, struct lookupObj * objectLookup, DTYPE * magY, uint64_t numUniqueObjects, uint64_t numFreqs);
void normalizepgramsingleobject(DTYPE * pgram, DTYPE * magY, uint64_t sizeDataForObject, uint64_t numFreqs);
// double computeMeanDataSquared(DTYPE * magY, uint64_t sizeData);
double computeStandardDevSquared(DTYPE * magY, uint64_t sizeData);

//With error
void lombscargleCPUOneObjectError(DTYPE * time, DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * foundPeriod, DTYPE * foundPower, DTYPE * pgram);
void lombscarglecpuError(bool mode, DTYPE * x, DTYPE * y, DTYPE *dy, const unsigned int sizeData, const unsigned int numFreqs, const DTYPE minFreq, const DTYPE maxFreq, const DTYPE freqStep, DTYPE * pgram);
void lombscarglecpuinnerloopAstroPy(int iteration, DTYPE * x, DTYPE * y, DTYPE * dy, DTYPE * pgram, DTYPE * freqToTest, const unsigned int sizeData);
void lombscargleCPUBatchError(unsigned int * objectId, DTYPE * timeX,  DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * sumPeriods, DTYPE * pgram, DTYPE * foundPeriod, DTYPE * foundPower);
void updateYerrorfactor(DTYPE * y, DTYPE *dy, const unsigned int sizeData);

//GPU functions
void batchGPULS(unsigned int * objectId, DTYPE * timeX,  DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * sumPeriods, DTYPE * pgram, DTYPE * foundPeriod, DTYPE * foundPower);
void GPULSOneObject(DTYPE * timeX,  DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * periodFound, DTYPE * maxPowerFound, DTYPE ** pgram);
void computeObjectRanges(unsigned int * objectId, unsigned int * sizeData, struct lookupObj ** objectLookup, unsigned int * numUniqueObjects);
void pinnedMemoryCopyDtoH(DTYPE * pinned_buffer, unsigned int sizeBufferElems, DTYPE * dev_data, DTYPE * pageable, unsigned int sizeTotalData);
void computePeriod(DTYPE * pgram, const unsigned int numFreqs, const DTYPE minFreq, const DTYPE freqStep, DTYPE * foundPeriod, DTYPE * foundPower);
unsigned int computeNumBatches(bool mode, bool pgrammode, unsigned int totalLengthTimeSeries, unsigned int numObjects, unsigned int numFreq);
double getGPUCapacity();
void warmUpGPU();

//for Batching and multi-GPU for batch mode
void batchGPULSWrapper(unsigned int * objectId, DTYPE * timeX,  DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * sumPeriods, DTYPE ** pgram, DTYPE ** foundPeriod, DTYPE ** foundPower);

//output to files and stdout:
void outputPeriodsToFile(struct lookupObj * objectLookup, unsigned int numUniqueObjects, DTYPE * foundPeriod, DTYPE * foundPower);
void outputPeriodsToFileTopThree(struct lookupObj * objectLookup, unsigned int numUniqueObjects, DTYPE * foundPeriod, DTYPE * foundPower);
void outputPgramToFile(struct lookupObj * objectLookup, unsigned int numUniqueObjects, unsigned int numFreqs, DTYPE ** pgram);
void outputPeriodsToStdout(struct lookupObj * objectLookup, unsigned int numUniqueObjects, DTYPE * foundPeriod, DTYPE * foundPower);


using namespace std;

#ifndef PYTHON
int main(int argc, char *argv[])
{

	warmUpGPU();
	cudaProfilerStart();
	omp_set_nested(1);

	//validation and output to file
	char fname[]="gpu_stats.txt";
	ofstream gpu_stats;
	gpu_stats.open(fname,ios::app);	

	/////////////////////////
	// Get information from command line
	/////////////////////////

	
	//Input data filename (objId, time, amplitude), minimum frequency, maximum frequency, numer of frequencies, mode
	if (argc!=6)
	{
	cout <<"\n\nIncorrect number of input parameters.\nExpected values: Data filename (objId, time, amplitude), minimum frequency, maximum frequency, number of frequencies, mode\n";
	return 0;
	}
	
	
	char inputFname[500];
	strcpy(inputFname,argv[1]);
	double minFreq=atof(argv[2]); //inclusive
	double maxFreq=atof(argv[3]); //exclusive
	const unsigned int freqToTest=atoi(argv[4]);
   int MODE = atoi(argv[5]);

	
	
	/////////////
	//Import Data
	/////////////
	unsigned int * objectId=NULL; 
	DTYPE * timeX=NULL; 
	DTYPE * magY=NULL;
	DTYPE * magDY=NULL;
	unsigned int sizeData;
	importObjXYData(inputFname, &sizeData, &objectId, &timeX, &magY, &magDY);

	// for (int i=0; i<10000; i++)
	// {
	// 	printf("\nobjectId: %d, %f, %f, %f", objectId[i], timeX[i],magY[i], magDY[i]);	
	// }
	// return 0;
	
	//pgram allocated in the functions below
	//Stores the LS power for each frequency
	DTYPE * pgram=NULL;

	//foundPeriod is allocated in the batch functions below
	DTYPE * foundPeriod=NULL;
	//foundPower is allocated in the batch functions below
	DTYPE * foundPower=NULL;

	//Batch of LS to compute on the GPU
	if (MODE==1)
	{
		DTYPE sumPeriods=0;
		
		double tstart=omp_get_wtime();
		
		//original before using multiple GPUs and batching
		// batchGPULS(objectId, timeX, magY, magDY, &sizeData, minFreq, maxFreq, freqToTest, &sumPeriods, &pgram, foundPeriod);
		batchGPULSWrapper(objectId, timeX, magY, magDY, &sizeData, minFreq, maxFreq, freqToTest, &sumPeriods, &pgram, &foundPeriod, &foundPower);
		
		double tend=omp_get_wtime();
		double totalTime=tend-tstart;
		
	}
	//One object to compute on the GPU
	else if (MODE==2)
	{
		DTYPE periodFound=0;	
		DTYPE maxPowerFound=0;	
		double tstart=omp_get_wtime();
		
		#if ERROR==1
		updateYerrorfactor(magY, magDY, sizeData);
		#endif	

		GPULSOneObject(timeX, magY, magDY, &sizeData, minFreq, maxFreq, freqToTest, &periodFound, &maxPowerFound, &pgram);
		
		double tend=omp_get_wtime();
		double totalTime=tend-tstart;
		
	}
	//CPU- batch processing
	else if (MODE==4)
	{
		DTYPE sumPeriods=0;
		double tstart=omp_get_wtime();
		#if ERROR==0
		lombscargleCPUBatch(objectId, timeX, magY, &sizeData, minFreq, maxFreq, freqToTest, &sumPeriods, pgram, foundPeriod, foundPower);
		#endif
		#if ERROR==1
		lombscargleCPUBatchError(objectId, timeX, magY, magDY, &sizeData, minFreq, maxFreq, freqToTest, &sumPeriods, pgram, foundPeriod, foundPower);
		#endif
		double tend=omp_get_wtime();
		double totalTime=tend-tstart;
		
	}
	//CPU- one object
	else if (MODE==5)
	{
		DTYPE foundPeriod=0;
		DTYPE foundPower=0;
		double tstart=omp_get_wtime();
		#if ERROR==0
		lombscargleCPUOneObject(timeX, magY, &sizeData, minFreq, maxFreq, freqToTest, &foundPeriod, &foundPower, pgram);
		#endif
		#if ERROR==1
		lombscargleCPUOneObjectError(timeX, magY, magDY, &sizeData, minFreq, maxFreq, freqToTest, &foundPeriod, &foundPower, pgram);
		#endif
		double tend=omp_get_wtime();
		double totalTime=tend-tstart;
		
	}



	//free memory
	free(foundPeriod);
	free(objectId);
	free(timeX);
	free(magY);
	free(magDY);
	free(pgram);


	cudaProfilerStop();

	return 0;
}
#endif

//For the Python shared library
#ifdef PYTHON
extern "C" void LombScarglePy(unsigned int * objectId, DTYPE * timeX, DTYPE * magY, DTYPE * magDY, unsigned int sizeData, double minFreq, double maxFreq, unsigned int freqToTest, int MODE, DTYPE * pgram)
{

	
	// cudaProfilerStart();
	omp_set_nested(1);


	/////////////////////////
	// Get information from command line
	/////////////////////////

	
	//Input data filename (objId, time, amplitude), minimum frequency, maximum frequency, numer of frequencies, mode
	// if (argc!=6)
	// {
	// cout <<"\n\nIncorrect number of input parameters.\nExpected values: Data filename (objId, time, amplitude), minimum frequency, maximum frequency, number of frequencies, mode\n";
	// return 0;
	// }
	
	
	// char inputFname[500];
	// strcpy(inputFname,argv[1]);
	// double minFreq=atof(argv[2]); //inclusive
	// double maxFreq=atof(argv[3]); //exclusive
	// const unsigned int freqToTest=atoi(argv[4]);
 //    int MODE = atoi(argv[5]);

	
	
	/////////////
	//Import Data
	/////////////
	// unsigned int * objectId=NULL; 
	// DTYPE * timeX=NULL; 
	// DTYPE * magY=NULL;
	// DTYPE * magDY=NULL;
	// unsigned int sizeData;
	// importObjXYData(inputFname, &sizeData, &objectId, &timeX, &magY, &magDY);

	
	
	//pgram allocated in the functions below
	//Stores the LS power for each frequency
	// DTYPE * pgram=NULL;

	//foundPeriod is allocated in the batch functions below
	DTYPE * foundPeriod=NULL;
	//foundPower is allocated in the batch functions below
	DTYPE * foundPower=NULL;

	//Batch of LS to compute on the GPU
	if (MODE==1)
	{
		warmUpGPU();
		DTYPE sumPeriods=0;
		
		double tstart=omp_get_wtime();
		
		//original before using multiple GPUs and batching
		// batchGPULS(objectId, timeX, magY, magDY, &sizeData, minFreq, maxFreq, freqToTest, &sumPeriods, &pgram, foundPeriod);
		batchGPULSWrapper(objectId, timeX, magY, magDY, &sizeData, minFreq, maxFreq, freqToTest, &sumPeriods, &pgram, &foundPeriod, &foundPower);
		
		double tend=omp_get_wtime();
		double totalTime=tend-tstart;

	}
	//One object to compute on the GPU
	else if (MODE==2)
	{
		warmUpGPU();
		DTYPE periodFound=0;	
		DTYPE maxPowerFound=0;	
		double tstart=omp_get_wtime();
		
		#if ERROR==1
		updateYerrorfactor(magY, magDY, sizeData);
		#endif	

		GPULSOneObject(timeX, magY, magDY, &sizeData, minFreq, maxFreq, freqToTest, &periodFound, &maxPowerFound, &pgram);
		
		double tend=omp_get_wtime();
		double totalTime=tend-tstart;

	}
	//CPU- batch processing
	else if (MODE==4)
	{
		DTYPE sumPeriods=0;
		double tstart=omp_get_wtime();
		#if ERROR==0
		lombscargleCPUBatch(objectId, timeX, magY, &sizeData, minFreq, maxFreq, freqToTest, &sumPeriods, pgram, foundPeriod, foundPower);
		#endif
		#if ERROR==1
		lombscargleCPUBatchError(objectId, timeX, magY, magDY, &sizeData, minFreq, maxFreq, freqToTest, &sumPeriods, pgram, foundPeriod, foundPower);
		#endif
		double tend=omp_get_wtime();
		double totalTime=tend-tstart;
		
	}
	//CPU- one object
	else if (MODE==5)
	{
		DTYPE foundPeriod=0;
		DTYPE foundPower=0;
		double tstart=omp_get_wtime();
		#if ERROR==0
		lombscargleCPUOneObject(timeX, magY, &sizeData, minFreq, maxFreq, freqToTest, &foundPeriod, &foundPower, pgram);
		#endif
		#if ERROR==1
		lombscargleCPUOneObjectError(timeX, magY, magDY, &sizeData, minFreq, maxFreq, freqToTest, &foundPeriod, &foundPower, pgram);
		#endif
		double tend=omp_get_wtime();
		double totalTime=tend-tstart;
		
	}



	//free memory
	free(foundPeriod);
	// free(objectId);
	// free(timeX);
	// free(magY);
	// free(magDY);
	// free(pgram);


	// cudaProfilerStop();

}
#endif

//Estimated memory footprint used to compute the number of batches
//used to compute the number of batches
//mode-0 is original
//mode-1 is floating mean, error propogation
//pgrammode-0 do not return the pgram
//pgrammode-1 return the pgram
//numObjects-- number of objects in the file
//totalLengthTimeSeries-- total lines in the file across all objects
//numFreq- number of frequencies searched
//pass in the underestimated capacity 
unsigned int computeNumBatches(bool mode, bool pgrammode, unsigned int totalLengthTimeSeries, unsigned int numObjects, unsigned int numFreq)
{

  //L-S space complexity (original)
  //2Nt+NoNf //Nt-length of time series, No-number of objects, Nf-number of frequencies searched		

  //L-S space complexity (with error)
  //3Nt+NoNf //Nt-length of time series, No-number of objects, Nf-number of frequencies searched		

  double underestGPUcapacityGiB=getGPUCapacity();
  
  double totalGiB=0;

  //original L-S (no error)
  if (mode==0)
  {
  totalGiB+=(sizeof(DTYPE)*(2.0*totalLengthTimeSeries))/(1024*1024*1024.0);
  }
  //L-S (with error)
  if (mode==1)
  {
  totalGiB+=(sizeof(DTYPE)*(3.0*totalLengthTimeSeries))/(1024*1024*1024.0);	
  }

  //pgram
  if (pgrammode==1)
  {
  double pgramsize=(sizeof(DTYPE)*(1.0*numObjects*numFreq))/(1024*1024*1024.0);		
  totalGiB+=pgramsize;		
  }

  

  unsigned int numBatches=ceil(totalGiB/(underestGPUcapacityGiB));
  numBatches=ceil((numBatches*1.0/NUMGPU))*NUMGPU;

  return numBatches;
}

void computeObjectRanges(unsigned int * objectId, unsigned int * sizeData, struct lookupObj ** objectLookup, unsigned int * numUniqueObjects)
{
	//Scan to find unique object ids;
	unsigned int lastId=objectId[0];
	unsigned int cntUnique=1;
	for (unsigned int i=1; i<*sizeData; i++)
	{
		if (lastId!=objectId[i])
		{
			cntUnique++;
			lastId=objectId[i];
		}
	}

	//allocate memory for the struct
	*objectLookup=(lookupObj*)malloc(sizeof(lookupObj)*cntUnique);

	*numUniqueObjects=cntUnique;


	lastId=objectId[0];
	unsigned int cnt=0;
	for (unsigned int i=1; i<*sizeData; i++)
	{
		if (lastId!=objectId[i])
		{
			(*objectLookup)[cnt].objId=lastId;
			(*objectLookup)[cnt+1].idxMin=i;
			(*objectLookup)[cnt].idxMax=i-1;
			cnt++;
			lastId=objectId[i];
		}
	}

	//first and last ones
	(*objectLookup)[0].idxMin=0;
	(*objectLookup)[cnt].objId=objectId[(*sizeData)-1];
	(*objectLookup)[cnt].idxMax=(*sizeData)-1;


}


void pinnedMemoryCopyDtoH(DTYPE * pinned_buffer, unsigned int sizeBufferElems, DTYPE * dev_data, DTYPE * pageable, unsigned int sizeTotalData)
{

  	cudaStream_t streams[NSTREAMS];
  	//create stream for the device
	for (int i=0; i<NSTREAMS; i++)
	{
	cudaStreamCreate(&streams[i]);
	}
  	

  	unsigned int numIters=sizeTotalData/sizeBufferElems;

  	unsigned int totalelemstransfered=0;
  	#pragma omp parallel for num_threads(NSTREAMS) reduction(+:totalelemstransfered)
  	for (unsigned int i=0; i<=numIters; i++)
  	{
  		int tid=omp_get_thread_num();
	  	unsigned int offsetstart=i*sizeBufferElems;
	  	unsigned int offsetend=(i+1)*sizeBufferElems;
	  	unsigned int elemsToTransfer=sizeBufferElems;
	  	if (offsetend>=sizeTotalData)
	  	{
	  		elemsToTransfer=sizeTotalData-offsetstart; 
	  	}	
	  	totalelemstransfered+=elemsToTransfer;
		
		unsigned int pinnedBufferOffset=tid*sizeBufferElems;
		gpuErrchk(cudaMemcpyAsync(pinned_buffer+pinnedBufferOffset, dev_data+(offsetstart), sizeof(DTYPE)*elemsToTransfer, cudaMemcpyDeviceToHost, streams[tid])); 

		cudaStreamSynchronize(streams[tid]);
		
		//Copy from pinned to pageable memory
		//Nested parallelization with openmp
		#pragma omp parallel for num_threads(4)
		for (unsigned int j=0; j<elemsToTransfer; j++)
		{
			pageable[offsetstart+j]=pinned_buffer[pinnedBufferOffset+j];
		} 	
	
	}

	for (int i=0; i<NSTREAMS; i++)
	{
	cudaStreamDestroy(streams[i]);
	}
}









//Compute pgram for one object, not a batch of objects
void GPULSOneObject(DTYPE * timeX,  DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * periodFound, DTYPE * maxPowerFound, DTYPE ** pgram)
{
	

	DTYPE * dev_timeX;
	DTYPE * dev_magY;
	unsigned int * dev_sizeData;
	DTYPE * dev_pgram;
	DTYPE * dev_foundPeriod;
	
	

	//allocate memory on the GPU
	gpuErrchk(cudaMalloc((void**)&dev_timeX, sizeof(DTYPE)*(*sizeData)));
	gpuErrchk(cudaMalloc((void**)&dev_magY, sizeof(DTYPE)*(*sizeData)));
	
	//If astropy implementation with error
	#if ERROR==1
	DTYPE * dev_magDY;
	gpuErrchk(cudaMalloc((void**)&dev_magDY, sizeof(DTYPE)*(*sizeData)));	
	#endif
	
	// Result periodogram
	//need to allocate it on the GPUeven if we do not return it to the host so that we can find the maximum power
	gpuErrchk(cudaMalloc((void**)&dev_pgram, sizeof(DTYPE)*numFreqs));
	
	

	//the maximum power in the periodogram. Use this when we don't want to return the periodogram
	gpuErrchk(cudaMalloc((void**)&dev_foundPeriod, sizeof(DTYPE)));
	gpuErrchk(cudaMalloc((void**)&dev_sizeData, sizeof(unsigned int)));

	//copy data to the GPU
	gpuErrchk(cudaMemcpy( dev_timeX, timeX, sizeof(DTYPE)*(*sizeData), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy( dev_magY, magY, sizeof(DTYPE)*(*sizeData), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy( dev_sizeData, sizeData, sizeof(unsigned int), cudaMemcpyHostToDevice));

	#if ERROR==1
	gpuErrchk(cudaMemcpy( dev_magDY, magDY, sizeof(DTYPE)*(*sizeData), cudaMemcpyHostToDevice));
	#endif

	const unsigned int szData=*sizeData;
	const unsigned int numBlocks=ceil(numFreqs*1.0/BLOCKSIZE*1.0);
	
	double tstart=omp_get_wtime();
  	//Do lomb-scargle
  	#if ERROR==0 && SHMEM==0
  	lombscargleOneObject<<< numBlocks, BLOCKSIZE>>>(dev_timeX, dev_magY, dev_pgram, szData, minFreq, maxFreq, numFreqs);
  	#elif ERROR==0 && SHMEM==1
  	lombscargleOneObjectSM<<< numBlocks, BLOCKSIZE>>>(dev_timeX, dev_magY, dev_pgram, szData, minFreq, maxFreq, numFreqs);
  	#elif ERROR==1
  	lombscargleOneObjectError<<< numBlocks, BLOCKSIZE>>>(dev_timeX, dev_magY, dev_magDY, dev_pgram, szData, minFreq, maxFreq, numFreqs);
  	#endif


  	//Find the index of the maximum power
  	thrust::device_ptr<DTYPE> maxLoc;
  	thrust::device_ptr<DTYPE> dev_ptr_pgram = thrust::device_pointer_cast(dev_pgram);

  	maxLoc = thrust::max_element(dev_ptr_pgram, dev_ptr_pgram+numFreqs); 
  	unsigned int maxPowerIdx= maxLoc - dev_ptr_pgram;
  	double freqStep=(maxFreq-minFreq)/(numFreqs*1.0);
  	//Validation: total period values
	*periodFound=(1.0/(minFreq+(maxPowerIdx*freqStep)))*2.0*M_PI;
	
  	cudaDeviceSynchronize();

  	double tend=omp_get_wtime();

  	//copy pgram back to the host if enabled
  	#ifndef PYTHON
  	#if RETURNPGRAM==1
  	*pgram=(DTYPE *)malloc(sizeof(DTYPE)*numFreqs);
  	#if PINNED==0
  	//standard if we don't use pinned memory for data transfers.
  	gpuErrchk(cudaMemcpy( *pgram, dev_pgram, sizeof(DTYPE)*numFreqs, cudaMemcpyDeviceToHost));
  	#endif
  	#endif
  	#endif


  	#if PINNED==1
  	DTYPE * pinned_buffer;
	unsigned int sizeBufferElems=(SIZEPINNEDBUFFERMIB*1024*1024)/(sizeof(DTYPE));
	gpuErrchk(cudaMallocHost((void**)&pinned_buffer, sizeof(DTYPE)*sizeBufferElems*NSTREAMS));	

  	unsigned int sizeDevData=numFreqs;
  	pinnedMemoryCopyDtoH(pinned_buffer, sizeBufferElems, dev_pgram, *pgram, sizeDevData);
	#endif

	
  	#if NORMALIZEPGRAM==1 && ERROR==0
  	normalizepgramsingleobject(*pgram, magY, *sizeData, numFreqs);
  	#endif

  	*maxPowerFound=(*pgram)[maxPowerIdx];


  	//free memory-- CUDA

  	cudaFree(dev_timeX);
  	cudaFree(dev_magY);
	cudaFree(dev_sizeData);
	cudaFree(dev_pgram);
	cudaFree(dev_foundPeriod);

	#if ERROR==1
	cudaFree(dev_magDY);
	#endif

	#if PINNED==1 && RETURNPGRAM==1
	cudaFreeHost(pinned_buffer);
	#endif
	
  	


}




















//Wrapper around main function for multi-GPU and batching
void batchGPULSWrapper(unsigned int * objectId, DTYPE * timeX,  DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * sumPeriods, DTYPE ** pgram, DTYPE ** foundPeriod, DTYPE ** foundPower)
{
	//compute total number of unqiue objects in the file and their sizes
	struct lookupObj * objectLookup=NULL;
	unsigned int numUniqueObjects;
	computeObjectRanges(objectId, sizeData, &objectLookup, &numUniqueObjects);
	unsigned int numBatches=computeNumBatches(ERROR, RETURNPGRAM, *sizeData, numUniqueObjects, numFreqs);

	//compute longest time span of an object for paper
	// double maxspan=0;
	// unsigned int objectIdxmaxspan=0;
	// for (int i=0; i<numUniqueObjects; i++)
	// {
	// 	double timespan=timeX[objectLookup[i].idxMax]-timeX[objectLookup[i].idxMin]; 
	// 	if (timespan>maxspan)
	// 	{
	// 		maxspan=timespan;
	// 		objectIdxmaxspan=i;
	// 	}
	// }

	// printf("ObjId: %u, Max span: %f: ", objectLookup[objectIdxmaxspan].objId, maxspan);


	//copy pgram back to the host if enabled
	#ifndef PYTHON
  	#if RETURNPGRAM==1
  	*pgram=(DTYPE *)malloc(sizeof(DTYPE)*(uint64_t)numFreqs*(uint64_t)numUniqueObjects);
  	#endif
  	#endif

	//allocate memory for the found periods
	*foundPeriod=(DTYPE *)malloc(sizeof(DTYPE)*numUniqueObjects);
	//allocate memory for the power corresponding to the period
	*foundPower=(DTYPE *)malloc(sizeof(DTYPE)*numUniqueObjects);

	//store the top 3 periods and their powers
	#if PRINTPERIODS==3
	free(*foundPeriod);
	free(*foundPower);
	//allocate memory for the found periods
	*foundPeriod=(DTYPE *)malloc(sizeof(DTYPE)*(uint64_t)numUniqueObjects*(uint64_t)3);
	//allocate memory for the power corresponding to the period
	*foundPower=(DTYPE *)malloc(sizeof(DTYPE)*(uint64_t)numUniqueObjects*(uint64_t)3);
	#endif


	//if there's only one batch then we execute the function as normal and do not partition the data
	if (numBatches==1)
	{
		batchGPULS(objectId, timeX, magY, magDY, sizeData, minFreq, maxFreq, numFreqs, sumPeriods, *pgram, *foundPeriod, *foundPower);
	}
	//if there is more than 1 batch
	else
	{


	//partition the total dataset
	//use cumulative sum of the length of the time series to perform partitioning
	unsigned int sumBatch=0;
	unsigned int totalSum=0; //sanity check
	unsigned int batchSize=*sizeData/numBatches;

	//batch indices into the main arrays
	unsigned int * dataIdxMin=(unsigned int *) malloc(sizeof(unsigned int)*numBatches);
	unsigned int * dataIdxMax=(unsigned int *) malloc(sizeof(unsigned int)*numBatches);

	//the number of data elems in each batch
	unsigned int * dataSizeBatches=(unsigned int *) malloc(sizeof(unsigned int)*numBatches);

	//the number of objects in the batch
	unsigned int * numObjectsInEachBatch=(unsigned int *) malloc(sizeof(unsigned int)*numBatches);

	//pgram write offset
	uint64_t * pgramWriteOffset=(uint64_t *) malloc(sizeof(uint64_t)*numBatches);

	//period write offset
	uint64_t * periodWriteOffset=(uint64_t *) malloc(sizeof(uint64_t)*numBatches);
	
	
	int batchCnt=0;
	unsigned int numObjectBatchCnt=0;
	
	for (unsigned int i=0; i<numUniqueObjects; i++)
	{
		totalSum+=objectLookup[i].idxMax-objectLookup[i].idxMin+1;
		sumBatch+=objectLookup[i].idxMax-objectLookup[i].idxMin+1;
		numObjectBatchCnt++;

		//if we reach the end of the batch, then we need to store the minimum and maximum ids in the array
		if (sumBatch>=batchSize)
		{
			numObjectsInEachBatch[batchCnt]=numObjectBatchCnt;
			dataIdxMax[batchCnt]=totalSum-1;
			sumBatch=0;	
			if (batchCnt>0)
			{
			dataIdxMin[batchCnt]=dataIdxMax[batchCnt-1]+1; 
			}
			batchCnt++;
			
			numObjectBatchCnt=0;
		}
	}
	//the first value is simply the first element
	dataIdxMin[0]=0;

	//the last data ranges will not get set in the loop
	dataIdxMin[numBatches-1]=dataIdxMax[numBatches-2]+1;
	dataIdxMax[numBatches-1]=objectLookup[numUniqueObjects-1].idxMax;

	//number of objects in each batch for allocating pgram memory
	numObjectsInEachBatch[numBatches-1]=numObjectBatchCnt;
	
	
	for (unsigned int i=0; i<numBatches; i++)
	{
		dataSizeBatches[i]=dataIdxMax[i]-dataIdxMin[i]+1;
		// printf("\nBatch %d, datasize: %u, number of objects: %u, dataIdxMin/Max: %u,%u", i, dataSizeBatches[i], numObjectsInEachBatch[i], dataIdxMin[i], dataIdxMax[i]);	
	}

	//cumulative sum for pgram write offset
	pgramWriteOffset[0]=0;
	periodWriteOffset[0]=0;
	uint64_t cumulativeObjects=0;
	for (unsigned int i=1; i<numBatches; i++)
	{
		cumulativeObjects+=numObjectsInEachBatch[i-1];
		pgramWriteOffset[i]=(uint64_t)cumulativeObjects*(uint64_t)numFreqs;
		periodWriteOffset[i]=cumulativeObjects;
		#if PRINTPERIODS==3
		periodWriteOffset[i]=cumulativeObjects*3;
		#endif
		// printf("\nCumulative objects (batch: %d): %u",i,cumulativeObjects);
	}




	
	//parallelize using the number of GPU's threads (e.g., 2 GPUs use 2 threads)
	#pragma omp parallel for num_threads(NUMGPU) schedule(dynamic)
	for (unsigned int i=0; i<numBatches; i++)
	{

		int tid=omp_get_thread_num();
		cudaSetDevice(tid);
		unsigned int idxMin=dataIdxMin[i];

		DTYPE sumPeriodsBatch=0;
		// unsigned int idxMax=dataIdxMax[i];
		// printf("\nPeriod write offset, batch %d: %u",i,periodWriteOffset[i]);
		batchGPULS(&objectId[idxMin], &timeX[idxMin], &magY[idxMin], &magDY[idxMin], &dataSizeBatches[i], minFreq, maxFreq, numFreqs, &sumPeriodsBatch, *pgram+pgramWriteOffset[i], *foundPeriod+(periodWriteOffset[i]), *foundPower+(periodWriteOffset[i]));

		// printf("\nSum periods batch: %0.9f", sumPeriodsBatch);
		#pragma omp atomic
		*sumPeriods+=sumPeriodsBatch;
	}

	free(dataIdxMin);
	free(dataIdxMax);
	free(dataSizeBatches);
	free(numObjectsInEachBatch);
	free(pgramWriteOffset);

	}//end of else statement for numBatches>1



  	///////////////////////
  	//Output

	//print found periods to stdout
  	#if PRINTPERIODS==1
  	outputPeriodsToStdout(objectLookup, numUniqueObjects, *foundPeriod, *foundPower);
  	#endif

	//print found periods to file
	#if PRINTPERIODS==2
	outputPeriodsToFile(objectLookup, numUniqueObjects, *foundPeriod, *foundPower);
	#endif

	//print top 3 found periods to file and their associated powers
	#if PRINTPERIODS==3
	outputPeriodsToFileTopThree(objectLookup, numUniqueObjects, *foundPeriod, *foundPower);
	#endif
  	
  	//Output pgram to file
  	#if PRINTPGRAM==1
	outputPgramToFile(objectLookup, numUniqueObjects, numFreqs, pgram);  	
  	#endif
  	

  	//End output
  	///////////////////////


	return;

}



void outputPgramToFile(struct lookupObj * objectLookup, unsigned int numUniqueObjects, unsigned int numFreqs, DTYPE ** pgram)
{
	return;
	char fnameoutput[]="pgram.txt";
  	printf("\nPrinting the pgram to file: %s", fnameoutput);
	ofstream pgramoutput;
	pgramoutput.open(fnameoutput,ios::out);	
  	pgramoutput.precision(4);
  	for (unsigned int i=0; i<numUniqueObjects; i++)
	{
		pgramoutput<<objectLookup[i].objId<<", ";
		for (unsigned int j=0; j<numFreqs; j++)
		{
		pgramoutput<<(*pgram)[(i*numFreqs)+j]<<", ";
		}
		pgramoutput<<endl;
	}
  	pgramoutput.close();
}




void outputPeriodsToFile(struct lookupObj * objectLookup, unsigned int numUniqueObjects, DTYPE * foundPeriod, DTYPE * foundPower)
{
	return;
	char fnamebestperiods[]="bestperiods.txt";
  	printf("\nPrinting the best periods/found power to file: %s", fnamebestperiods);
	ofstream bestperiodsoutput;
	bestperiodsoutput.open(fnamebestperiods,ios::out);	
  	bestperiodsoutput.precision(6);
  	for (unsigned int i=0; i<numUniqueObjects; i++)
	{
		bestperiodsoutput<<objectLookup[i].objId<<", "<<foundPeriod[i]<<", "<<foundPower[i]<<endl;
	}
  	bestperiodsoutput.close();
}


void outputPeriodsToFileTopThree(struct lookupObj * objectLookup, unsigned int numUniqueObjects, DTYPE * foundPeriod, DTYPE * foundPower)
{
	return;
	char fnamebestperiods[]="bestperiods_top_three.txt";
  	printf("\nPrinting the top three best periods/found powers to file (object id, period #1, period #2, period #3, power #1, power #2, power #3: %s", fnamebestperiods);
	ofstream bestperiodsoutput;
	bestperiodsoutput.open(fnamebestperiods,ios::out);	
  	bestperiodsoutput.precision(6);
  	for (uint64_t i=0; i<numUniqueObjects; i++)
	{
		uint64_t offset=i*(uint64_t)3;	
		bestperiodsoutput<<objectLookup[i].objId<<", "<<foundPeriod[offset]<<", "<<foundPeriod[offset+(uint64_t)1]<<", "<<foundPeriod[offset+(uint64_t)2]<<", "<<foundPower[offset]<<", "<<foundPower[offset+(uint64_t)1]<<", "<<foundPower[offset+(uint64_t)2]<<endl;
	}
  	bestperiodsoutput.close();
}

void outputPeriodsToStdout(struct lookupObj * objectLookup, unsigned int numUniqueObjects, DTYPE * foundPeriod, DTYPE * foundPower)
{
	return;
	for (unsigned int i=0; i<numUniqueObjects; i++)
  	{
	  	printf("\nObject: %d Period: %f, Power: %f ",objectLookup[i].objId,foundPeriod[i], foundPower[i]);
  	}
}

//Send the minimum and maximum frequency and number of frequencies to test to the GPU (not a list of frequencies)
void batchGPULS(unsigned int * objectId, DTYPE * timeX,  DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * sumPeriods, DTYPE * pgram, DTYPE * foundPeriod, DTYPE * foundPower)
{
	



	DTYPE * dev_timeX;
	DTYPE * dev_magY;
	unsigned int * dev_sizeData;
	DTYPE * dev_foundPeriod;
	DTYPE * dev_pgram;

	struct lookupObj * dev_objectLookup;

	//compute the object ranges in the arrays and store in struct
	//This is given by the objectId
	struct lookupObj * objectLookup=NULL;
	unsigned int numUniqueObjects;
	computeObjectRanges(objectId, sizeData, &objectLookup, &numUniqueObjects);

    // foundPeriod=(DTYPE *)malloc(sizeof(DTYPE)*numUniqueObjects);



	//allocate memory on the GPU
	gpuErrchk(cudaMalloc((void**)&dev_timeX, sizeof(DTYPE)*(*sizeData)));
	gpuErrchk(cudaMalloc((void**)&dev_magY, sizeof(DTYPE)*(*sizeData)));

	//If astropy implementation with error
	#if ERROR==1
	DTYPE * dev_magDY;
	gpuErrchk(cudaMalloc((void**)&dev_magDY, sizeof(DTYPE)*(*sizeData)));

	//Need to incorporate error into magnitudes
	for (unsigned int i=0; i<numUniqueObjects; i++)
	{
		unsigned int idxMin=objectLookup[i].idxMin;
		unsigned int idxMax=objectLookup[i].idxMax;
		unsigned int sizeDataForObject=idxMax-idxMin+1;
		updateYerrorfactor(&magY[idxMin], &magDY[idxMin], sizeDataForObject);
	}	
	#endif
	
	#if RETURNPGRAM==1
	// Result periodogram must be number of unique objects * the size of the frequency array
	gpuErrchk(cudaMalloc((void**)&dev_pgram, sizeof(DTYPE)*numFreqs*numUniqueObjects));

	//Make a small pinned memory buffer for transferring the array back
	DTYPE * pinned_buffer;
	unsigned int sizeBufferElems=(SIZEPINNEDBUFFERMIB*1024*1024)/(sizeof(DTYPE));
	gpuErrchk(cudaMallocHost((void**)&pinned_buffer, sizeof(DTYPE)*sizeBufferElems*NSTREAMS));	
	#endif

	#if RETURNPGRAM==0
	//If not returning the pgram then do not allocate memory
	dev_pgram=NULL;
	#endif

	//the maximum power in each periodogram. Use this when we don't want to return the periodogram
	gpuErrchk(cudaMalloc((void**)&dev_foundPeriod, sizeof(DTYPE)*numUniqueObjects));
	gpuErrchk(cudaMalloc((void**)&dev_sizeData, sizeof(unsigned int)));
	gpuErrchk(cudaMalloc((void**)&dev_objectLookup, sizeof(lookupObj)*(numUniqueObjects)));

	//copy data to the GPU
	gpuErrchk(cudaMemcpy( dev_timeX, timeX, sizeof(DTYPE)*(*sizeData), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy( dev_magY, magY, sizeof(DTYPE)*(*sizeData), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy( dev_sizeData, sizeData, sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy( dev_objectLookup, objectLookup, sizeof(lookupObj)*(numUniqueObjects), cudaMemcpyHostToDevice));

	#if ERROR==1
	gpuErrchk(cudaMemcpy( dev_magDY, magDY, sizeof(DTYPE)*(*sizeData), cudaMemcpyHostToDevice));
	#endif

	
	const int numBlocks=numUniqueObjects;
	double tstart=omp_get_wtime();
  	//Do lomb-scargle
  	#if ERROR==0 && SHMEM==0
  	lombscargleBatch<<< numBlocks, BLOCKSIZE>>>(dev_timeX, dev_magY, dev_objectLookup, dev_pgram, dev_foundPeriod, minFreq, maxFreq, numFreqs);
  	#elif ERROR==0 && SHMEM==1
  	lombscargleBatchSM<<< numBlocks, BLOCKSIZE>>>(dev_timeX, dev_magY, dev_objectLookup, dev_pgram, dev_foundPeriod, minFreq, maxFreq, numFreqs);
  	#elif ERROR==1 //no SHMEM option
  	lombscargleBatchError<<< numBlocks, BLOCKSIZE>>>(dev_timeX, dev_magY, dev_magDY, dev_objectLookup, dev_pgram, dev_foundPeriod, minFreq, maxFreq, numFreqs);
  	#endif

  	cudaDeviceSynchronize();

  	double tend=omp_get_wtime();

  	

  	//copy pgram back to the host if enabled
  	#if RETURNPGRAM==1

  	#if PINNED==0
  	//standard if we don't use pinned memory for data transfers.
  	gpuErrchk(cudaMemcpy( pgram, dev_pgram, sizeof(DTYPE)*numUniqueObjects*numFreqs, cudaMemcpyDeviceToHost));
  	#endif

  	#if PINNED==1
  	unsigned int sizeDevData=numUniqueObjects*numFreqs;
  	pinnedMemoryCopyDtoH(pinned_buffer, sizeBufferElems, dev_pgram, pgram, sizeDevData);
	#endif

  	#endif



  	//Return the maximum power for each object
  	#if RETURNPGRAM==0
  	gpuErrchk(cudaMemcpy( foundPeriod, dev_foundPeriod, sizeof(DTYPE)*(numUniqueObjects), cudaMemcpyDeviceToHost));
  	#endif
  	
  	

  	//For each object, find the maximum power in the pgram

  	#if RETURNPGRAM==1
  	//compute the maximum power using the returned pgram
  	double tstartcpupgram=omp_get_wtime();


  	#if NORMALIZEPGRAM==1 && ERROR==0
  	normalizepgram(pgram, objectLookup, magY, numUniqueObjects, numFreqs);
  	#endif


  	double freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	
  	#pragma omp parallel for num_threads(NTHREADSCPU)
  	for (uint64_t i=0; i<numUniqueObjects; i++)
  	{
  		DTYPE maxPower=0;
  		uint64_t maxPowerIdx=0;
	  	for (uint64_t j=0; j<(uint64_t)numFreqs; j++)
	  	{
	  		if (maxPower<pgram[i*(uint64_t)numFreqs+j])
	  		{
	  			maxPower=pgram[i*(uint64_t)numFreqs+j];
	  			maxPowerIdx=j;
	  		}
	  	}

	  	//Validation: total period values
		foundPeriod[i]=(1.0/(minFreq+(maxPowerIdx*freqStep)))*2.0*M_PI;
		foundPower[i]=maxPower;
	  	
 		 	
	  	// if (i==0)
	  	// {
	  	// 	printf("\nmaxPowerIdx: %u", maxPowerIdx);
	  	// }
  	}

  	//capture top 3 periods and associated powers
  	//not for timing, but for analysis
  	//use 3 scans
  	#if PRINTPERIODS==3

  	// #pragma omp parallel for num_threads(NTHREADSCPU)
  	for (uint64_t i=0; i<(uint64_t)numUniqueObjects; i++)
  	{


  		DTYPE maxPower[3]={0.0,0.0,0.0};
  		uint64_t maxPowerIdx[3]={0,0,0};

  		//get the maximum power #1
	  	for (uint64_t j=0; j<(uint64_t)numFreqs; j++)
	  	{
	  		if (maxPower[0]<pgram[i*(uint64_t)numFreqs+j])
	  		{
	  			maxPower[0]=pgram[i*(uint64_t)numFreqs+j];
	  			maxPowerIdx[0]=j;
	  		}
	  	}

	  	//get the second maximum power #2
	  	for (uint64_t j=0; j<(uint64_t)numFreqs; j++)
	  	{
	  		if ((j!=maxPowerIdx[0]) && (maxPower[1]<pgram[i*(uint64_t)numFreqs+j]))
	  		{
	  			maxPower[1]=pgram[i*(uint64_t)numFreqs+j];
	  			maxPowerIdx[1]=j;
	  		}
	  	}

	  	// //get the third maximum power #3
	  	for (uint64_t j=0; j<(uint64_t)numFreqs; j++)
	  	{
	  		if ((j!=maxPowerIdx[0]) && (j!=maxPowerIdx[1]) && (maxPower[2]<pgram[i*(uint64_t)numFreqs+j]))
	  		{
	  			maxPower[2]=pgram[i*(uint64_t)numFreqs+j];
	  			maxPowerIdx[2]=j;
	  		}
	  	}
	  	

	  	//Validation: total period values
	  	uint64_t offset=i*3;
		foundPeriod[offset+0]=(1.0/(minFreq+(maxPowerIdx[0]*freqStep)))*2.0*M_PI;
		foundPeriod[offset+1]=(1.0/(minFreq+(maxPowerIdx[1]*freqStep)))*2.0*M_PI;
		foundPeriod[offset+2]=(1.0/(minFreq+(maxPowerIdx[2]*freqStep)))*2.0*M_PI;
		foundPower[offset+0]=maxPower[0];
		foundPower[offset+1]=maxPower[1];
		foundPower[offset+2]=maxPower[2];

		// printf("\n object id: %d, %f, %f, %f, %f, %f, %f", i,foundPeriod[offset+(uint64_t)0], foundPeriod[offset+(uint64_t)1], foundPeriod[offset+(uint64_t)2], foundPower[offset+(uint64_t)0], foundPower[offset+(uint64_t)1], foundPower[offset+(uint64_t)2]);
  	}


  	#endif



  	double tendcpupgram=omp_get_wtime();
  	#endif	


  	
  	//for validation
  	#if PRINTPERIODS!=3
	for (unsigned int i=0; i<numUniqueObjects; i++)
  	{
	  	(*sumPeriods)+=foundPeriod[i];
  	}
  	#endif

  	#if PRINTPERIODS==3
  	for (uint64_t i=0; i<numUniqueObjects; i++)
  	{
	  	(*sumPeriods)+=foundPeriod[i*(uint64_t)3];
  	}
  	#endif
  	
  	

	

  	//free memory
  	// free(foundPeriod);
  	free(objectLookup);

  	//free memory-- CUDA
  	cudaFree(dev_timeX);
  	cudaFree(dev_magY);
	cudaFree(dev_sizeData);
	cudaFree(dev_pgram);
	cudaFree(dev_foundPeriod);
	cudaFree(dev_objectLookup);

	#if ERROR==1
	cudaFree(dev_magDY); 
	#endif

	#if PINNED==1 && RETURNPGRAM==1
	cudaFreeHost(pinned_buffer);
	#endif

	
	


  	


}

// double computeMeanDataSquared(DTYPE * magY, uint64_t sizeData)
// {
// 	//compute the mean of the data squared
// 	double sum=0;
// 	for (uint64_t k=0; k<sizeData; k++)
// 	{
// 		sum+=(magY[k]*magY[k]);
// 	}

// 	double meandatasquared=sum/(sizeData*1.0);
// 	return meandatasquared;
// }

double computeStandardDevSquared(DTYPE * magY, uint64_t sizeData)
{

	//Step 1: compute the mean
	double sum=0;
	for (uint64_t k=0; k<sizeData; k++)
	{
		sum+=magY[k];
	}

	double mean=sum/(sizeData*1.0);

	//Step 2: compute the standard deviation
	double sum2=0;	
	for (uint64_t k=0; k<sizeData; k++)
	{
		sum2+=(magY[k]-mean)*(magY[k]-mean);
	}

	double sigma=sqrt(sum2/(sizeData*1.0));
	
	return sigma*sigma;
}

void normalizepgram(DTYPE * pgram, struct lookupObj * objectLookup, DTYPE * magY, uint64_t numUniqueObjects, uint64_t numFreqs)
{

		
	#pragma omp parallel for num_threads(NTHREADSCPU)
  	for (uint64_t i=0; i<numUniqueObjects; i++)
  	{
  		//get the data size for the object
		uint64_t idxMin=objectLookup[i].idxMin;
		uint64_t idxMax=objectLookup[i].idxMax;
		uint64_t sizeDataForObject=idxMax-idxMin+1;
		
		double stddevsquared=computeStandardDevSquared(magY+idxMin, sizeDataForObject);

	  	for (uint64_t j=0; j<numFreqs; j++)
	  	{
	  		pgram[i*numFreqs+j]*=2.0/(sizeDataForObject*stddevsquared);	
	  	}
  	}

}

void normalizepgramsingleobject(DTYPE * pgram, DTYPE * magY, uint64_t sizeDataForObject, uint64_t numFreqs)
{
		
		//compute the mean of the data squared
		double stddevsquared=computeStandardDevSquared(magY, sizeDataForObject);

  		#pragma omp parallel for num_threads(NTHREADSCPU)
	  	for (uint64_t j=0; j<numFreqs; j++)
	  	{
	  		pgram[j]*=2.0/(sizeDataForObject*stddevsquared);	
	  	}
}




//parallelize over the frequency if computing a single object
void lombscargleCPUOneObject(DTYPE * timeX,  DTYPE * magY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * foundPeriod, DTYPE * foundPower, DTYPE * pgram)
{
	#ifndef PYTHON
	pgram=(DTYPE *)malloc(sizeof(DTYPE)*(numFreqs));
	#endif

	const DTYPE freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	

	//1 refers to the mode of executing in parallel inside the LS algorithm
	lombscarglecpu(1, timeX, magY, *sizeData, numFreqs, minFreq, maxFreq, freqStep, pgram);	
	#if NORMALIZEPGRAM==1 && ERROR==0
  	normalizepgramsingleobject(pgram, magY, *sizeData, numFreqs);
  	#endif
	computePeriod(pgram, numFreqs, minFreq, freqStep, foundPeriod, foundPower);

}

//uses error propogation
void lombscargleCPUOneObjectError(DTYPE * timeX,  DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * foundPeriod, DTYPE * foundPower, DTYPE * pgram)
{
	#ifndef PYTHON
	pgram=(DTYPE *)malloc(sizeof(DTYPE)*(numFreqs));
	#endif

	const DTYPE freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	

	//1 refers to the mode of executing in parallel inside the LS algorithm
	lombscarglecpuError(1, timeX, magY, magDY, *sizeData, numFreqs, minFreq, maxFreq, freqStep, pgram);	
	#if NORMALIZEPGRAM==1 && ERROR==0
  	normalizepgramsingleobject(pgram, magY, *sizeData, numFreqs);
  	#endif
	computePeriod(pgram, numFreqs, minFreq, freqStep, foundPeriod, foundPower);

}


void lombscargleCPUBatchError(unsigned int * objectId, DTYPE * timeX,  DTYPE * magY, DTYPE * magDY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * sumPeriods, DTYPE * pgram, DTYPE * foundPeriod, DTYPE * foundPower)
{


	//compute the object ranges in the arrays and store in struct
	//This is given by the objectId
	struct lookupObj * objectLookup=NULL;
	unsigned int numUniqueObjects;
	computeObjectRanges(objectId, sizeData, &objectLookup, &numUniqueObjects);
	#ifndef PYTHON
	pgram=(DTYPE *)malloc(sizeof(DTYPE)*(numFreqs)*numUniqueObjects);
	#endif
	foundPeriod=(DTYPE *)malloc(sizeof(DTYPE)*numUniqueObjects);
	foundPower=(DTYPE *)malloc(sizeof(DTYPE)*numUniqueObjects);

	const DTYPE freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	

	//for each object, call the sequential cpu algorithm
	#pragma omp parallel for num_threads(NTHREADSCPU) schedule(dynamic)
	for (unsigned int i=0; i<numUniqueObjects; i++)
	{
		unsigned int idxMin=objectLookup[i].idxMin;
		unsigned int idxMax=objectLookup[i].idxMax;
		unsigned int sizeDataForObject=idxMax-idxMin+1;
		uint64_t pgramWriteOffset=(uint64_t)i*(uint64_t)numFreqs;
		//0 refers to the mode of executing sequentially inside the LS algorithm
		lombscarglecpuError(0, &timeX[idxMin], &magY[idxMin], &magDY[idxMin], sizeDataForObject, numFreqs, minFreq, maxFreq, freqStep, pgram+pgramWriteOffset);	
		#if NORMALIZEPGRAM==1 && ERROR==0
	  	normalizepgramsingleobject(pgram+pgramWriteOffset, &magY[idxMin], sizeDataForObject, numFreqs);
	  	#endif
		computePeriod(pgram+pgramWriteOffset, numFreqs, minFreq, freqStep, &foundPeriod[i], &foundPower[i]);
	}

	

	///////////////////////
  	//Output

	//print found periods to stdout
  	#if PRINTPERIODS==1
  	outputPeriodsToStdout(objectLookup, numUniqueObjects, foundPeriod, foundPower);
  	#endif

	//print found periods to file
	#if PRINTPERIODS==2
	outputPeriodsToFile(objectLookup, numUniqueObjects, foundPeriod, foundPower);
	#endif
  	
  	//Output pgram to file
  	#if PRINTPGRAM==1
	outputPgramToFile(objectLookup, numUniqueObjects, numFreqs, &pgram);  	
  	#endif
  	

  	//End output
  	///////////////////////

	//Validation
 	for (unsigned int i=0; i<numUniqueObjects; i++)
  	{
	  	(*sumPeriods)+=foundPeriod[i];
  	}

	

}


void lombscargleCPUBatch(unsigned int * objectId, DTYPE * timeX,  DTYPE * magY, unsigned int * sizeData, const DTYPE minFreq, const DTYPE maxFreq, const unsigned int numFreqs, DTYPE * sumPeriods, DTYPE * pgram, DTYPE * foundPeriod, DTYPE * foundPower)
{


	//compute the object ranges in the arrays and store in struct
	//This is given by the objectId
	struct lookupObj * objectLookup=NULL;
	unsigned int numUniqueObjects;
	computeObjectRanges(objectId, sizeData, &objectLookup, &numUniqueObjects);
	#ifndef PYTHON
	pgram=(DTYPE *)malloc(sizeof(DTYPE)*(numFreqs)*numUniqueObjects);
	#endif
	foundPeriod=(DTYPE *)malloc(sizeof(DTYPE)*numUniqueObjects);
	foundPower=(DTYPE *)malloc(sizeof(DTYPE)*numUniqueObjects);

	const DTYPE freqStep=(maxFreq-minFreq)/(numFreqs*1.0);	

	//for each object, call the sequential cpu algorithm
	#pragma omp parallel for num_threads(NTHREADSCPU) schedule(dynamic)
	for (unsigned int i=0; i<numUniqueObjects; i++)
	{
		unsigned int idxMin=objectLookup[i].idxMin;
		unsigned int idxMax=objectLookup[i].idxMax;
		unsigned int sizeDataForObject=idxMax-idxMin+1;
		uint64_t pgramWriteOffset=(uint64_t)i*(uint64_t)numFreqs;
		//0 refers to the mode of executing sequentially inside the LS algorithm
		lombscarglecpu(0, &timeX[idxMin], &magY[idxMin], sizeDataForObject, numFreqs, minFreq, maxFreq, freqStep, pgram+pgramWriteOffset);	

		#if NORMALIZEPGRAM==1 && ERROR==0
	  	normalizepgramsingleobject(pgram+pgramWriteOffset, &magY[idxMin], sizeDataForObject, numFreqs);
	  	#endif
		
		computePeriod(pgram+pgramWriteOffset, numFreqs, minFreq, freqStep, &foundPeriod[i], &foundPower[i]);
	}

	// #if PRINTPERIODS==1
	// for (int i=0; i<numUniqueObjects; i++)
	// {
	// printf("\nObject: %d, Period: %f, Power: %f",objectLookup[i].objId, foundPeriod[i], foundPower[i]);
	// }
	// #endif

	 ///////////////////////
  	//Output

	//print found periods to stdout
  	#if PRINTPERIODS==1
  	outputPeriodsToStdout(objectLookup, numUniqueObjects, foundPeriod, foundPower);
  	#endif

	//print found periods to file
	#if PRINTPERIODS==2
	outputPeriodsToFile(objectLookup, numUniqueObjects, foundPeriod, foundPower);
	#endif
  	
  	//Output pgram to file
  	#if PRINTPGRAM==1
	outputPgramToFile(objectLookup, numUniqueObjects, numFreqs, &pgram);  	
  	#endif
  	

  	//End output
  	///////////////////////



	//Validation
 	for (unsigned int i=0; i<numUniqueObjects; i++)
  	{
	  	(*sumPeriods)+=foundPeriod[i];
  	}

	

}

void computePeriod(DTYPE * pgram, const unsigned int numFreqs, const DTYPE minFreq, const DTYPE freqStep, DTYPE * foundPeriod, DTYPE * foundPower)
{
		DTYPE maxPower=0;
  		unsigned int maxPowerIdx=0;
	  	for (unsigned int i=0; i<numFreqs; i++)
	  	{
	  		if (maxPower<pgram[i])
	  		{
	  			maxPower=pgram[i];
	  			maxPowerIdx=i;
	  		}
	  	}

	  	// printf("\nMax power idx: %d", maxPowerIdx);
	  	
	  	//Validation: total period values
		*foundPeriod=(1.0/(minFreq+(maxPowerIdx*freqStep)))*2.0*M_PI;
		*foundPower=maxPower;	  	
}


void lombscarglecpuinnerloop(int iteration, DTYPE * x, DTYPE * y, DTYPE * pgram, DTYPE * freqToTest, const unsigned int sizeData)
{
			DTYPE c, s, xc, xs, cc, ss, cs;
	    	DTYPE tau, c_tau, s_tau, c_tau2, s_tau2, cs_tau;
	        xc = 0.0;
	        xs = 0.0;
	        cc = 0.0;
	        ss = 0.0;
	        cs = 0.0;

	        for (unsigned int j=0; j<sizeData; j++)
	        {	
	            c = cos((*freqToTest) * x[j]);
	            s = sin((*freqToTest) * x[j]);

	            xc += y[j] * c;
	            xs += y[j] * s;
	            cc += c * c;
	            ss += s * s;
	            cs += c * s;
	        }
	            
	        tau = atan2(2.0 * cs, cc - ss) / (2.0 * (*freqToTest));
	        c_tau = cos((*freqToTest) * tau);
	        s_tau = sin((*freqToTest) * tau);
	        c_tau2 = c_tau * c_tau;
	        s_tau2 = s_tau * s_tau;
	        cs_tau = 2.0 * c_tau * s_tau;

	        pgram[iteration] = 0.5 * ((((c_tau * xc + s_tau * xs)*(c_tau * xc + s_tau * xs)) / 
	            (c_tau2 * cc + cs_tau * cs + s_tau2 * ss)) + 
	            (((c_tau * xs - s_tau * xc)*(c_tau * xs - s_tau * xc)) / 
	            (c_tau2 * ss - cs_tau * cs + s_tau2 * cc)));
}

//lombsscarge on the CPU 
//Mode==0 means run sequentially
//Mode==1 means parallelize over the frequency loop
void lombscarglecpu(bool mode, DTYPE * x, DTYPE * y, const unsigned int sizeData, const unsigned int numFreqs, const DTYPE minFreq, const DTYPE maxFreq, const DTYPE freqStep, DTYPE * pgram)
{

	    if (mode==0)
	    {	
			for (unsigned int i=0; i<numFreqs; i++)
			{
				DTYPE freqToTest=minFreq+(freqStep*i);
				lombscarglecpuinnerloop(i, x, y, pgram, &freqToTest, sizeData);
		    }
		}
		else if(mode==1)
		{
			#pragma omp parallel for num_threads(NTHREADSCPU) schedule(static)
			for (unsigned int i=0; i<numFreqs; i++)
			{
				DTYPE freqToTest=minFreq+(freqStep*i);
				lombscarglecpuinnerloop(i, x, y, pgram, &freqToTest, sizeData);
		    }
		}

}


//Pre-center the data with error:		
// w = dy ** -2
//    y = y - np.dot(w, y) / np.sum(w)
void updateYerrorfactor(DTYPE * y, DTYPE *dy, const unsigned int sizeData)
{

		//Pre-center the data with error:
		//w = dy ** -2
		//sum w
		DTYPE * w =(DTYPE *)malloc(sizeof(DTYPE)*sizeData);
		DTYPE sumw=0;
		#pragma omp parallel for num_threads(NTHREADSCPU) reduction(+:sumw)
		for (unsigned int i=0; i<sizeData; i++)
		{
			w[i]=1.0/sqrt(dy[i]);
			sumw+=w[i];
		}
		//compute dot product w,y
		DTYPE dotwy=0;
		#pragma omp parallel for num_threads(NTHREADSCPU) reduction(+:dotwy)
		for (unsigned int i=0; i<sizeData; i++)
		{
			dotwy+=w[i]*y[i];
		}

		//update y to account for dot product and sum w
		//y = y - dot(w, y) / np.sum(w)	
		#pragma omp parallel for num_threads(NTHREADSCPU)
		for (unsigned int i=0; i<sizeData; i++)
		{
			y[i]=y[i]-dotwy/sumw;
		}

		free(w);
}

//lombsscarge on the CPU for AstroPy with error
//Mode==0 means run sequentially (batch mode)
//Mode==1 means parallelize over the frequency loop (multiobject)
void lombscarglecpuError(bool mode, DTYPE * x, DTYPE * y, DTYPE *dy, const unsigned int sizeData, const unsigned int numFreqs, const DTYPE minFreq, const DTYPE maxFreq, const DTYPE freqStep, DTYPE * pgram)
{
		// printf("\nExecuting astropy version");


		updateYerrorfactor(y, dy, sizeData);


	    if (mode==0)
	    {	
			for (unsigned int i=0; i<numFreqs; i++)
			{
				DTYPE freqToTest=minFreq+(freqStep*i);
				lombscarglecpuinnerloopAstroPy(i, x, y, dy, pgram, &freqToTest, sizeData);
		    }
		}
		else if(mode==1)
		{
			
			#pragma omp parallel for num_threads(NTHREADSCPU) schedule(static)
			for (unsigned int i=0; i<numFreqs; i++)
			{
				DTYPE freqToTest=minFreq+(freqStep*i);
				lombscarglecpuinnerloopAstroPy(i, x, y, dy, pgram, &freqToTest, sizeData);
		    }
		}

}


//AstroPy has error propogration and fits to the mean
//Ported from here:
//https://github.com/astropy/astropy/blob/master/astropy/timeseries/periodograms/lombscargle/implementations/cython_impl.pyx
//Uses the generalized periodogram in the cython code
void lombscarglecpuinnerloopAstroPy(int iteration, DTYPE * x, DTYPE * y, DTYPE * dy, DTYPE * pgram, 
	DTYPE * freqToTest, const unsigned int sizeData)
{

	DTYPE w, omega_t, sin_omega_t, cos_omega_t, S, C, S2, C2, tau, Y, wsum, YY, Stau, Ctau, YCtau, YStau, CCtau, SStau; 

	wsum = 0.0;
	S = 0.0;
	C = 0.0;
	S2 = 0.0;
	C2 = 0.0;

	//first pass: determine tau
	for (unsigned int j=0; j<sizeData; j++)
	{
    w = 1.0 / dy[j];
    w *= w;
    wsum += w;
    omega_t = (*freqToTest) * x[j];
    sin_omega_t = sin(omega_t);
    cos_omega_t = cos(omega_t);
    S += w * sin_omega_t;
    C += w * cos_omega_t;
    S2 += 2.0 * w * sin_omega_t * cos_omega_t;
    C2 += w - 2.0 * w * sin_omega_t * sin_omega_t;
	}	    

		S2 /= wsum;
		C2 /= wsum;
		S /= wsum;
		C /= wsum;
		S2 -= (2.0 * S * C);
		C2 -= (C * C - S * S);
		tau = 0.5 * atan2(S2, C2) / (*freqToTest);
		Y = 0.0;
		YY = 0.0;
		Stau = 0.0;
		Ctau = 0.0;
		YCtau = 0.0;
		YStau = 0.0;
		CCtau = 0.0;
		SStau = 0.0;
		// second pass: compute the power
		for (unsigned int j=0; j<sizeData; j++)
		{
		    w = 1.0 / dy[j];
		    w *= w;
		    omega_t = (*freqToTest) * (x[j] - tau);
		    sin_omega_t = sin(omega_t);
		    cos_omega_t = cos(omega_t);
		    Y += w * y[j];
		    YY += w * y[j] * y[j];
		    Ctau += w * cos_omega_t;
		    Stau += w * sin_omega_t;
		    YCtau += w * y[j] * cos_omega_t;
		    YStau += w * y[j] * sin_omega_t;
		    CCtau += w * cos_omega_t * cos_omega_t;
		    SStau += w * sin_omega_t * sin_omega_t;
		}
		Y /= wsum;
		YY /= wsum;
		Ctau /= wsum;
		Stau /= wsum;
		YCtau /= wsum;
		YStau /= wsum;
		CCtau /= wsum;
		SStau /= wsum;
		YCtau -= Y * Ctau;
		YStau -= Y * Stau;
		CCtau -= Ctau * Ctau;
		SStau -= Stau * Stau;
		YY -= Y * Y;
		


    pgram[iteration] = (YCtau * YCtau / CCtau + YStau * YStau / SStau) / YY;
}



void importObjXYData(char * fnamedata, unsigned int * sizeData, unsigned int ** objectId, DTYPE ** timeX, DTYPE ** magY, DTYPE ** magDY)
{

	//import objectId, timeX, magY

	std::vector<DTYPE>tmpAllData;
	std::ifstream in(fnamedata);
	unsigned int cnt=0;
	for (std::string f; getline(in, f, ',');){

	DTYPE i;
		 std::stringstream ss(f);
	    while (ss >> i)
	    {
	        tmpAllData.push_back(i);
	        // array[cnt]=i;
	        cnt++;
	        if (ss.peek() == ',')
	            ss.ignore();
	    }

  	}




  	
  	#if ERROR==0
  	*sizeData=(unsigned int)tmpAllData.size()/3;
  	#endif

  	#if ERROR==1
  	*sizeData=(unsigned int)tmpAllData.size()/4;
  	#endif
  	
  	*objectId=(unsigned int *)malloc(sizeof(DTYPE)*(*sizeData));
  	*timeX=   (DTYPE *)malloc(sizeof(DTYPE)*(*sizeData));
  	*magY=    (DTYPE *)malloc(sizeof(DTYPE)*(*sizeData));

  	#if ERROR==1
  	*magDY=    (DTYPE *)malloc(sizeof(DTYPE)*(*sizeData));
  	#endif


  	#if ERROR==0
  	for (unsigned int i=0; i<*sizeData; i++){
  		(*objectId)[i]=tmpAllData[(i*3)+0];
  		(*timeX)[i]   =tmpAllData[(i*3)+1];
  		(*magY)[i]    =tmpAllData[(i*3)+2];
  	}
  	#endif

  	#if ERROR==1
  	for (unsigned int i=0; i<*sizeData; i++){
  		(*objectId)[i]=tmpAllData[(i*4)+0];
  		(*timeX)[i]   =tmpAllData[(i*4)+1];
  		(*magY)[i]    =tmpAllData[(i*4)+2];
  		(*magDY)[i]    =tmpAllData[(i*4)+3];
  	}
  	#endif

}




void warmUpGPU(){

for (int i=0; i<NUMGPU; i++)
{
cudaSetDevice(i); 	
cudaDeviceSynchronize();
}

}


double getGPUCapacity()
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  
  //Read the global memory capacity from the device.
  unsigned long int globalmembytes=0;
  gpuErrchk(cudaMemGetInfo(NULL,&globalmembytes));
  double totalcapacityGiB=globalmembytes*1.0/(1024*1024*1024.0);

  double underestcapacityGiB=totalcapacityGiB*ALPHA;
  return underestcapacityGiB;
}