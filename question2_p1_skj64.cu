#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <math.h>
#include <omp.h> 

//See values of N in assignment instructions.
#define N 10000

using namespace std;

//Do not change the seed, or your answer will not be correct
#define SEED 72

//For GPU implementation
#define BLOCKSIZE 1024


struct pointData{
double x;
double y;
};



void generateDataset(struct pointData * data);
__global__ void calcDistances(struct pointData * inputData, double * inputEpsilon, unsigned long long int  * numOfDistancesWithinEps);

int main(int argc, char *argv[])
{
	
	//Read epsilon distance from command line
	if (argc!=2)
	{
	printf("\nIncorrect number of input parameters. Please input an epsilon distance.\n");
	return 0;
	}
	
	
	char inputEpsilon[20];
	strcpy(inputEpsilon,argv[1]);
	double epsilon=atof(inputEpsilon);
	
	

	//generate dataset:
	struct pointData * data;
	data=(struct pointData*)malloc(sizeof(struct pointData)*N);
	printf("\nSize of dataset (MiB): %f",(2.0*sizeof(double)*N*1.0)/(1024.0*1024.0));
	generateDataset(data);

	omp_set_num_threads(1);
	
	///////////////////Time set ups
	double totalTranferingTimeFromCPUToGPU;
	double startTransferTimeGPU;
	double endTransferTimeGPU;
	
	double totalTransferingTimeFromGPUToCPU;
	double startTransferCPU;
	double endTransferCPU;
	
	double totalTransferTime;
	
	double totalGPUKernelTime;
	double startGPUKern;
	double endGPUKern;
	///////////////////////////////////
	
	double tstart=omp_get_wtime();
	

	//Write your code here:
	//The data you need to use is stored in the variable "data", 
	//which is of type pointData
	
	cudaError_t errCode=cudaSuccess;
	
	if(errCode != cudaSuccess)
	{
		cout << "\nLast error: " << errCode << endl; 	
	}
	
	struct pointData * dev_Data;
	double * dev_Epsilon;
	unsigned long long int * countOfPointInEpsilon;
	unsigned long long int * dev_countOfPointInEpsilon;
	countOfPointInEpsilon = (unsigned long long int *)malloc(sizeof(unsigned long long int));
	dev_countOfPointInEpsilon = (unsigned long long int *)malloc(sizeof(unsigned long long int));
	*countOfPointInEpsilon = 0;
	
	
	startTransferTimeGPU = omp_get_wtime();
	
	//allocate on the device: data
	errCode=cudaMalloc((struct pointData**)&dev_Data, sizeof(struct pointData)*N);	
	if(errCode != cudaSuccess) {
	cout << "\nError: point list error with code " << errCode << endl; 
	}
	
	//allocate epsilon value on device
	errCode=cudaMalloc((double**)&dev_Epsilon, sizeof(double));
    if(errCode != cudaSuccess) {
    cout << "\nError: B error with code " << errCode << endl;
    }
	
	//allocate the number of points in the epsilon on device
	errCode=cudaMalloc((unsigned long long int**)&dev_countOfPointInEpsilon, sizeof(unsigned long long int));	
	if(errCode != cudaSuccess) {
	cout << "\nError: points in Epsilon error with code " << errCode << endl; 
	}
	
	errCode=cudaMemcpy( dev_Data, data, sizeof(struct pointData)*N, cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	cout << "\nError: dev_Data memcpy error with code " << errCode << endl; 
	}
	
	errCode=cudaMemcpy( dev_Epsilon, &epsilon, sizeof(double), cudaMemcpyHostToDevice);
    if(errCode != cudaSuccess) {
    cout << "\nError: B memcpy error with code " << errCode << endl;
    }
	
	errCode=cudaMemcpy( dev_countOfPointInEpsilon, countOfPointInEpsilon, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
	if(errCode != cudaSuccess) {
	cout << "\nError: count in circle memcpy error with code " << errCode << endl; 
	}
	cudaDeviceSynchronize();
	
	endTransferTimeGPU = omp_get_wtime();
	
	totalTranferingTimeFromCPUToGPU = endTransferTimeGPU - startTransferTimeGPU;
	
	printf("\nTotal time to transfer from CPU to GPU(s): %f",totalTranferingTimeFromCPUToGPU);
	
	
	startGPUKern = omp_get_wtime();
	
	const unsigned int totalBlocks=ceil(N*1.0/1024.0);
	printf("\ntotal blocks: %d",totalBlocks);
	calcDistances<<<totalBlocks,1024>>>(dev_Data, dev_Epsilon, dev_countOfPointInEpsilon);
	cudaDeviceSynchronize();
	
	endGPUKern = omp_get_wtime();
	
	totalGPUKernelTime = endGPUKern - startGPUKern;
	
	printf("\nTotal time for GPU Kernel(s): %f",totalGPUKernelTime);
	
	
	startTransferCPU = omp_get_wtime();
	
	errCode=cudaMemcpy( countOfPointInEpsilon, dev_countOfPointInEpsilon, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    if(errCode != cudaSuccess) {
    cout << "\nError: getting result form GPU error with code " << errCode << endl;
    }
    cudaDeviceSynchronize();
    
    endTransferCPU = omp_get_wtime();
    
    totalTransferingTimeFromGPUToCPU = endTransferCPU - startTransferCPU;
    
    printf("\nTotal number of points within epsilon (GPU): %llu",*countOfPointInEpsilon);
	
	cudaDeviceSynchronize();

	double tend=omp_get_wtime();
	
	printf("\nTotal time (s): %f",tend-tstart);
	
	totalTransferTime = totalTranferingTimeFromCPUToGPU + totalTransferingTimeFromGPUToCPU;
	
	printf("\nTotal transfer time (s): %f",totalTransferTime);
	
	free(data);
	printf("\n");
	return 0;
}

__global__ void calcDistances(struct pointData * inputData, double * inputEpsilon, unsigned long long int  * numOfDistancesWithinEps) {

unsigned int tid=threadIdx.x+ (blockIdx.x*blockDim.x);
if (tid>=N){
    return;
}
    
int dataIndex;
for (dataIndex = 0; dataIndex < N; dataIndex++){
    if ((sqrt(((inputData[tid].x - inputData[dataIndex].x) * (inputData[tid].x - inputData[dataIndex].x)) + 
        ((inputData[tid].y - inputData[dataIndex].y) * (inputData[tid].y - inputData[dataIndex].y)))) <= *inputEpsilon){
        atomicAdd(numOfDistancesWithinEps, int(1));
    }
}

return;
}

//Do not modify the dataset generator or you will get the wrong answer
void generateDataset(struct pointData * data)
{

	//seed RNG
	srand(SEED);


	for (unsigned int i=0; i<N; i++){
		data[i].x=1000.0*((double)(rand()) / RAND_MAX);	
		data[i].y=1000.0*((double)(rand()) / RAND_MAX);	
	}
	

}

void warmUpGPU(){
cudaDeviceSynchronize();
return;
}