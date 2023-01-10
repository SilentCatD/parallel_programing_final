
#include <stdio.h>
#include <stdint.h>

// !! Not working when there're multiple block doing constructMinCostTable | DEADLOCK

// - Use shared memory and constant memory in 2 somber convo steps
// - Use cuda stream to parallel 2 somber convo step
// - cross-block communication using atomicAdd (not working)

#define FILTER_WIDTH 3
__constant__ int dc_xSombelFilter[FILTER_WIDTH * FILTER_WIDTH];
__constant__ int dc_ySombelFilter[FILTER_WIDTH * FILTER_WIDTH];

#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
		fprintf(stderr, "code: %d, reason: %s\n", error,\
				cudaGetErrorString(error));\
		exit(EXIT_FAILURE);\
	}\
}

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);                                                                 
		cudaEventSynchronize(start);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

__global__ void convoImageKernel(unsigned char* inPixels, int width, int height,
        int filterWidth, int * outPixels, bool isXSombel = true)
{
	extern __shared__ unsigned char s_inPixels[];

	int sharedWidth = blockDim.x + filterWidth - 1;
	int filterRadius = filterWidth / 2;


	// Copy batch 1
	int dest = threadIdx.y * blockDim.x + threadIdx.x;
	int destY = dest / sharedWidth; 
	int destX = dest % sharedWidth;
	int s_inPixelsIdx = destY * sharedWidth + destX;
	int srcY = blockIdx.y * blockDim.y + destY - filterRadius;
	int srcX = blockIdx.x * blockDim.x + destX - filterRadius;

	srcY = min(max(0, srcY), height - 1);
	srcX = min(max(0, srcX), width - 1);
	
	int srcIdx = srcY * width + srcX;
	s_inPixels[s_inPixelsIdx] = inPixels[srcIdx];


	// Copy batch 2
	dest =  threadIdx.y * blockDim.x + threadIdx.x +  blockDim.x * blockDim.y;
	destY = dest / sharedWidth;
	destX = dest % sharedWidth;
	s_inPixelsIdx = destY * sharedWidth + destX;
	srcY = blockIdx.y * blockDim.y + destY - filterRadius;
	srcX = blockIdx.x * blockDim.x + destX - filterRadius;

	srcY = min(max(0, srcY), height - 1);
	srcX = min(max(0, srcX), width - 1);

	srcIdx = srcY * width + srcX; 
	if(destY < sharedWidth){
		s_inPixels[s_inPixelsIdx] = inPixels[srcIdx];
	}
	__syncthreads();
	// convo
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	if (r < height && c < width){

		int *filter;
		if(isXSombel){
			filter = dc_xSombelFilter;
		}else{
			filter = dc_ySombelFilter;
		}

		int outPixel = 0;
		for (int filterR = 0; filterR < filterWidth; filterR++){
			for(int filterC = 0; filterC < filterWidth; filterC++){
				unsigned char s_inPixel = s_inPixels[(threadIdx.y + filterR) * sharedWidth + (threadIdx.x + filterC)];
				int filterVal =  filter[filterR * filterWidth + filterC];
				outPixel += s_inPixel * filterVal; 
			}
		}
		outPixels[r * width + c] = outPixel;	
	}
}							


__global__ void energyCalcKernel(int* xSombelOut, int* ySombelOut, int width, int height, int *outPixels){
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	if (c < width && r < height) {
		int i = r * width + c;
		outPixels[i] = abs(xSombelOut[i]) + abs(ySombelOut[i]);
	}

}

__global__ void convertRgb2GrayKernel(unsigned char * inPixels, int width, int height, 
		unsigned char * outPixels)
{
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	if (c < width && r < height) {
		int i = r * width + c;
		unsigned char red = inPixels[3 * i];
		unsigned char green = inPixels[3 * i + 1];
		unsigned char blue = inPixels[3 * i + 2];
		outPixels[i] = 0.299f * red + 0.587f * green + 0.114f * blue;
	}
}

__device__ int bCount = 0;
__global__ void constructEnergyCostPathTableKernel(int* energy, int width, int height, int* costTable, int*pathTable){
	int cIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if(cIdx< width){
		costTable[cIdx] = energy[cIdx];
		pathTable[cIdx] = 0;
	}
	__syncthreads();

	if(cIdx < width){
		for(int r = 1; r < height; r++){

			int minRIdx = r - 1;

			int leftC = max(0, cIdx - 1);
			int rightC = min(width - 1, cIdx + 1);

			int leftVal = costTable[minRIdx * width + leftC];
			int midVal = costTable[minRIdx * width + cIdx];
			int rightVal = costTable[minRIdx * width + rightC];

			int minVal = min(leftVal, min(midVal, rightVal));

			int idx = r * width + cIdx;
			int costVal = minVal + energy[idx];
			costTable[idx] = costVal;
			if(minVal == leftVal){
				if(leftC == threadIdx.x){
					pathTable[idx] = 0;
				}else{
					pathTable[idx] = -1;
				}
			}else if(minVal == midVal){
				pathTable[idx] = 0;
			}else{
				if(rightC == threadIdx.x){
					pathTable[idx] = 0;
				}else{
					pathTable[idx] = 1;
				}
			}

			if(threadIdx.x == 0){
				atomicAdd(&bCount ,1);
				__threadfence();
				while(bCount < r * gridDim.x){}
			}
			__syncthreads();
		}
	}

}

__global__ void findMinCIdxKernel(int *costTable, int width, int height, int sharedArrMemSize, int* localMinIdx, int* localMin){
	extern __shared__ int s_mem[];
	int* s_costVal = (int*) s_mem;	
	int* s_costIdx = (int *) &(s_mem[sharedArrMemSize]);

	int lastRowIdx = (height - 1) * width;
	
	// Load data to shared mem
	int i1 = 2 * blockDim.x * blockIdx.x + threadIdx.x;
	int i2 = i1 +  blockDim.x;

	if(i1 < width){
		s_costVal[threadIdx.x] = costTable[lastRowIdx + i1];
		s_costIdx[threadIdx.x] = i1;
	}
	if(i2 < width){
		s_costVal[threadIdx.x + blockDim.x] = costTable[lastRowIdx + i2];
		s_costIdx[threadIdx.x + blockDim.x] = i2;
	}
	__syncthreads();

	for(int stride = blockDim.x; stride > 0; stride/=2){
		if(threadIdx.x < stride){
			int value1 = s_costVal[threadIdx.x];
			int value2 = s_costVal[threadIdx.x + stride];
			int index1= s_costIdx[threadIdx.x];
			int index2 = s_costIdx[threadIdx.x + stride];
			if(index1 < width && index2 < width && value2 < value1){
				s_costVal[threadIdx.x] = value2;
				s_costIdx[threadIdx.x] = index2;
			}
		}
		__syncthreads();
	}

	if(threadIdx.x == 0){
		localMinIdx[blockIdx.x] = s_costIdx[0];
		localMin[blockIdx.x] = s_costVal[0];
	}

}

void findSeam(int minCIdx, int* pathTable, int width, int height, int* seamPos){
	for(int r = height - 1; r >= 0; r--){
		seamPos[r * 2] = r;		
		seamPos[r * 2 + 1] = minCIdx;		

		int nextC = pathTable[r * width + minCIdx];
		if(nextC == -1){
			minCIdx --;

		}else if(nextC == 1){
			minCIdx++;
		}
	}
}

__global__ void addSeamOnDevice(unsigned char* inPixels, unsigned char* inPixels1, int width, int height, int* deviceSeamPos)
{
	// do stuff
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	if (c < width && r < height) {
		int i = r * width + c;
		if (c < deviceSeamPos[2 * r + 1]) {
			inPixels1[3 * i] = inPixels[3 * i];
			inPixels1[3 * i + 1] = inPixels[3 * i + 1];
			inPixels1[3 * i + 2] = inPixels[3 * i + 2];
		}
		else if (c == deviceSeamPos[2 * r + 1]) {		
			int o = r * width + c + 1;
			inPixels1[3 * i] = inPixels[3 * i];
			inPixels1[3 * i + 1] = inPixels[3 * i + 1];
			inPixels1[3 * i + 2] = inPixels[3 * i + 2];
			inPixels1[3 * o] = inPixels[3 * i];
			inPixels1[3 * o + 1] = inPixels[3 * i + 1];
			inPixels1[3 * o + 2] = inPixels[3 * i + 2];
		}
		else {	
			int o = r * width + c + 1;
			inPixels1[3 * o] = inPixels[3 * i];
			inPixels1[3 * o + 1] = inPixels[3 * i + 1];
			inPixels1[3 * o + 2] = inPixels[3 * i + 2];
		}
	}
}

void findSeamOnDeivce(unsigned char* inPixels, int& width, int height, int* deviceSeamPos, int* outCostTable, int* outPathTable, int &outMinColIdx, dim3 convoBlockSize = dim3(1, 1), int costTableBlockSize = 1024, int minColIdxBlockSize = 512){

	// Allocate memory
	GpuTimer timer;
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
	printf("GPU name: %s\n", devProp.name);
	printf("GPU compute capability: %d.%d\n", devProp.major, devProp.minor);

	unsigned char* d_inPixels;
	size_t nBytes = width * height * sizeof(unsigned char);
	CHECK(cudaMalloc(&d_inPixels, nBytes*3));
	CHECK(cudaMemcpy(d_inPixels, inPixels, nBytes * 3, cudaMemcpyHostToDevice));
	dim3 gridSize((width - 1) / convoBlockSize.x + 1, (height - 1) / convoBlockSize.y + 1);
	int costTableGridSize = (width - 1) / costTableBlockSize + 1;
	int minColIdxGridSize  = (width - 1)/ (minColIdxBlockSize * 2) + 1; 
	int minColIdxSharedDataSize = minColIdxBlockSize * 4 * sizeof(int);

	// grayScale
	unsigned char *d_outGrayScale;
	CHECK(cudaMalloc(&d_outGrayScale, nBytes));
	

	// Sombel
	int s_inPixelsSize = ((convoBlockSize.x + FILTER_WIDTH - 1) * (convoBlockSize.y + FILTER_WIDTH - 1)) * sizeof(unsigned char);

	// xSombel
	int xSombelFilter[] = {1, 0, -1, 2, 0, -2, 1, 0, -1}; 
	int *d_xSombelOut;
    cudaStream_t xSombelConvoStream;
    CHECK(cudaStreamCreate(&xSombelConvoStream));
	CHECK(cudaMalloc(&d_xSombelOut, width * height * sizeof(int)));
	CHECK(cudaMemcpyToSymbol(dc_xSombelFilter, xSombelFilter, sizeof(xSombelFilter)));

	// ySombel	
	int ySombelFilter[] = {1, 2, 1, 0, 0, 0, -1, -2, -1}; 
	int  * d_ySombelOut;
    cudaStream_t ySombelConvoStream;
    CHECK(cudaStreamCreate(&ySombelConvoStream));
	CHECK(cudaMalloc(&d_ySombelOut, width * height * sizeof(int)));
	CHECK(cudaMemcpyToSymbol(dc_ySombelFilter, ySombelFilter, sizeof(ySombelFilter)));

	// energy
	int* d_energy;
	CHECK(cudaMalloc(&d_energy, width * height * sizeof(int)));

	// construct energy cost table
	int* d_costTable, *d_pathTable;
	CHECK(cudaMalloc(&d_costTable, width * height * sizeof(int)));
	CHECK(cudaMalloc(&d_pathTable, width * height * sizeof(int)));

	// find min idx
	int* d_localMinIdx,* d_localMin;
	CHECK(cudaMalloc(&d_localMinIdx, minColIdxGridSize * sizeof(int)));
	CHECK(cudaMalloc(&d_localMin, minColIdxGridSize * sizeof(int)));
	int * localMin = (int*)malloc(minColIdxGridSize * sizeof(int));
	int* localMinIdx = (int*) malloc(minColIdxGridSize * sizeof(int));

	// Execute
	timer.Start();

	// grayScale
	convertRgb2GrayKernel<<<gridSize, convoBlockSize>>>(d_inPixels, width, height, d_outGrayScale);
	
	cudaDeviceSynchronize();
	// xSombel
	convoImageKernel<<<gridSize, convoBlockSize, s_inPixelsSize, xSombelConvoStream>>>(d_outGrayScale, width, height, FILTER_WIDTH, d_xSombelOut, true);

	// ySombel	
	convoImageKernel<<<gridSize, convoBlockSize, s_inPixelsSize, ySombelConvoStream>>>(d_outGrayScale, width, height, FILTER_WIDTH, d_ySombelOut, false);
	cudaDeviceSynchronize();

	// energy
	energyCalcKernel<<<gridSize, convoBlockSize>>>(d_xSombelOut, d_ySombelOut, width, height, d_energy);

	// construct energy cost table
	constructEnergyCostPathTableKernel<<<costTableGridSize, costTableBlockSize>>>(d_energy, width, height, d_costTable, d_pathTable);

	// find min idx
	findMinCIdxKernel<<<minColIdxGridSize, minColIdxBlockSize, minColIdxSharedDataSize>>>(d_costTable, width, height, minColIdxBlockSize * 2, d_localMinIdx, d_localMin);
	CHECK(cudaMemcpy(localMin, d_localMin, minColIdxGridSize * sizeof(int), cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(localMinIdx, d_localMinIdx, minColIdxGridSize * sizeof(int), cudaMemcpyDeviceToHost));

	int minColumn = localMinIdx[0];
	int minVal = localMin[0];
	for(int i = 0 ; i < minColIdxGridSize; i++){
		if(localMin[i] < minVal){
			minVal = localMin[i];
			minColumn = localMinIdx[i];
		}
	}

	CHECK(cudaMemcpy(outPathTable, d_pathTable, width * height * sizeof(int), cudaMemcpyDeviceToHost));
	findSeam(minColumn, outPathTable, width, height, deviceSeamPos);
	
	cudaDeviceSynchronize();

	cudaDeviceSynchronize();
	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time of device: %f ms\n\n", time);

	CHECK(cudaMemcpy(outCostTable, d_costTable, width * height * sizeof(int), cudaMemcpyDeviceToHost));
	outMinColIdx = minColumn;

	// add seam
	int* d_SeamPos;
	CHECK(cudaMalloc(&d_SeamPos, height * sizeof(int) * 2));
	CHECK(cudaMemcpy(d_SeamPos, deviceSeamPos, height * sizeof(int) * 2, cudaMemcpyHostToDevice));
	unsigned char* d_inPixels1;
	for (int num = 0; num < 50; num++) {
		CHECK(cudaMalloc(&d_inPixels1, (width + 1) * height * sizeof(unsigned char) * 3))
		addSeamOnDevice<<<gridSize, convoBlockSize>>>(d_inPixels, d_inPixels1, width, height, d_SeamPos);
		CHECK(cudaFree(d_inPixels));
		d_inPixels = d_inPixels1;
		width++;
	}

	CHECK(cudaFree(d_inPixels));
	CHECK(cudaFree(d_outGrayScale));
	CHECK(cudaFree(d_xSombelOut));
	CHECK(cudaFree(d_ySombelOut));
	CHECK(cudaFree(d_energy));
	CHECK(cudaFree(d_costTable));
	CHECK(cudaFree(d_pathTable));
    CHECK(cudaStreamDestroy(xSombelConvoStream));
    CHECK(cudaStreamDestroy(ySombelConvoStream));
}

void convertRgb2Gray(unsigned char * inPixels, int width, int height,
		unsigned char * outPixels)
{
	for (int r = 0; r < height; r++) {
		for (int c = 0; c < width; c++)
		{
			int i = r * width + c;
			unsigned char red = inPixels[3 * i];
			unsigned char green = inPixels[3 * i + 1];
			unsigned char blue = inPixels[3 * i + 2];
			outPixels[i] = 0.299f*red + 0.587f*green + 0.114f*blue;
		}
	}
}


void convoImage(unsigned char * inPixels, int width, int height, int * filter, int filterWidth, 
        int* outPixels)
{
	for (int outPixelsR = 0; outPixelsR < height; outPixelsR++)
	{
		for (int outPixelsC = 0; outPixelsC < width; outPixelsC++)
		{
			int outPixel = 0;
			for (int filterR = 0; filterR < filterWidth; filterR++)
			{
				for (int filterC = 0; filterC < filterWidth; filterC++)
				{
					int filterVal = filter[filterR*filterWidth + filterC];
					int inPixelsR = outPixelsR - filterWidth/2 + filterR;
					int inPixelsC = outPixelsC - filterWidth/2 + filterC;
					inPixelsR = min(max(0, inPixelsR), height - 1);
					inPixelsC = min(max(0, inPixelsC), width - 1);
					unsigned char inPixel = inPixels[inPixelsR*width + inPixelsC];
					outPixel += filterVal * inPixel;
				}
			}
			outPixels[outPixelsR*width + outPixelsC] = outPixel; 
		}
	}
}

void energyCalc(int* xSombelOut, int* ySomberOut, int width, int height, int* outPixels){
	for(int i = 0; i < width * height; i++){
		outPixels[i] = abs(xSombelOut[i]) + abs(ySomberOut[i]);
	}
}

void constructEnergyCostPathTable(int *energy, int width, int height, int* costTable,int* pathTable){
	for(int r = 0; r < height; r++){
		for(int c = 0; c < width; c++){
			if(r == 0){
				costTable[c] = energy[c];
				pathTable[c] = 0;
			}else{
				int idx = r * width + c;

				int minRIdx = r - 1;
				
				int leftC = max(0, c - 1);

				int rightC = min(width - 1, c + 1);

				int leftVal = costTable[minRIdx * width + leftC];
				int midVal = costTable[minRIdx * width + c];
				int rightVal = costTable[minRIdx * width + rightC];

				int minVal = min(leftVal, min(midVal, rightVal));

				costTable[idx] = minVal + energy[idx];
				if(minVal == leftVal){
					if(leftC == c){
						pathTable[idx] = 0;
					}else{
						pathTable[idx] = -1;
					}
				}else if(minVal == midVal){
					pathTable[idx] = 0;
				}else{
					if(rightC == c){
						pathTable[idx] = 0;
					}else{
						pathTable[idx] = 1;
					}
				}

			}
		}
	}
}

int findMinCIdx(int* costTable, int width, int height){
	int idx = (height -1) * width;
	int result = 0;
	int currentMin = costTable[idx];
	for(int i = 0; i < width; i++){
		if(costTable[idx + i] <  currentMin){
			currentMin = costTable[idx + i];
			result = i;
		}
	}
	return result;
}


void findSeamOnHost(unsigned char* inPixels, int width, int height, int* seamPos, int * outCostTable, int* outPathTable, int &outMinColIdx){

	// Allocate memory
	GpuTimer timer;
	// grayScale
	unsigned char *grayScale = (unsigned char*) malloc(width * height * sizeof(unsigned char));
	// xSomber
	int xSombelFilter[] = {1, 0, -1, 2, 0, -2, 1, 0, -1}; 
	int* xSombelOut = (int*) malloc(width * height * sizeof(int));
	// ySomber	
	int ySombelFilter[] = {1, 2, 1, 0, 0, 0, -1, -2, -1}; 
	int* ySombelOut = (int*) malloc(width * height *  sizeof(int));
	// energy
	int *energy = (int*) malloc(width * height * sizeof(int));

	// construct energy cost table
	int *costTable = (int*) malloc(width * height * sizeof(int));
	int *pathTable = (int*) malloc(width * height * sizeof(int));

	// Execute
	timer.Start();
	// grayScale
	convertRgb2Gray(inPixels,width, height, grayScale);

	// xSomber
	convoImage(grayScale, width, height, xSombelFilter, 3, xSombelOut);

	// ySomber	
	convoImage(grayScale, width, height, ySombelFilter, 3, ySombelOut);

	// energy
	energyCalc(xSombelOut, ySombelOut, width, height, energy);

	// construct energy cost table
	constructEnergyCostPathTable(energy, width, height, costTable, pathTable);

	// find min column
	int minColumn = findMinCIdx(costTable, width, height);

	// find seam
	findSeam(minColumn, pathTable, width, height, seamPos);

	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time of host: %f ms\n\n", time);

	for(int i = 0; i < width * height; i++){
		outCostTable[i] = costTable[i];
		outPathTable[i] = pathTable[i];
	}
	outMinColIdx = minColumn;

	free(grayScale);
	free(costTable); 
	free(pathTable);
	free(xSombelOut);
	free(ySombelOut);
	free(energy);
}


void readPnm(char * fileName, int &width, int &height, unsigned char * &pixels)
{
	FILE * f = fopen(fileName, "rb");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	
	if (strcmp(type, "P6") != 0) // In this exercise, we don't touch other types
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	int c = getc(f);
    while (c == '#') {
    while (getc(f) != '\n') ;
         c = getc(f);
    }
    ungetc(c, f);

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);
	
	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) // In this exercise, we assume 1 byte per value
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}
	size_t nBytes = width * height * 3 * sizeof(unsigned char);
	pixels = (unsigned char*)malloc(nBytes);
	fread(pixels, nBytes, 1, f);
	fclose(f);
}

void writePnm(unsigned char* pixels, int numChannels, int width, int height, char * fileName)
{
	FILE * f = fopen(fileName, "wb");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	
	if (numChannels == 1)
		fprintf(f, "P5\n");
	else if (numChannels == 3)
		fprintf(f, "P6\n");
	else
	{
		fclose(f);
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	fprintf(f, "%i %i\n255", width, height); 
	size_t nBytes = width * height * numChannels * sizeof(unsigned char);

	fwrite(pixels, nBytes, 1, f);
	
	fclose(f);
}

char * concatStr(const char * s1, const char * s2)
{
    char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

double checkCorrect(int* out, int* out2, int width, int height){
	float err = 0;
	int n =  width * height;
	for (int i = 0; i < n; i++)
		err += abs(out[i] - out2[i]);
	err /= n;
	return err;
}

double checkCorrectPos(int* out, int *out2, int height){
	float err = 0;
	for(int i = 0; i < height; i++){
		err += abs(out[i*2] - out2[i*2]) + abs(out[i * 2 + 1] - out2[i * 2 + 1]);
	}
	return err / (height);
}

int main(int argc, char ** argv)
{

	// Read input image file
	int width, height;
	unsigned char * inPixels;
	readPnm(argv[1], width, height, inPixels);
	printf("\nImage size (width x height): %i x %i\n", width, height);

	int *hostSeamPos = (int*) malloc(height * 2 * sizeof(int));
	int *outCostTableHost = (int*) malloc(width * height * sizeof(int));
	int *outPathTableHost = (int*) malloc(width * height * sizeof(int));
	int outMinColIdxHost;
	findSeamOnHost(inPixels, width, height, hostSeamPos, outCostTableHost, outPathTableHost, outMinColIdxHost);

	int *deviceSeamPos = (int*) malloc(height * 2 * sizeof(int));
	int *outCostTableDevice = (int*) malloc(width * height * sizeof(int));
	int *outPathTableDevice = (int*) malloc(width * height * sizeof(int));
	int outMinColIdxDevice;
	findSeamOnDeivce(inPixels, width, height,deviceSeamPos ,outCostTableDevice, outPathTableDevice, outMinColIdxDevice,dim3(32, 32), 1024, 256);

	double errCostTable = checkCorrect(outCostTableDevice, outCostTableHost, width, height);
	double errPathTable = checkCorrect(outPathTableDevice, outPathTableHost, width, height);
	double errPos =  checkCorrectPos(hostSeamPos, deviceSeamPos, height);
	printf("Error cost table: %f\n", errCostTable);
	printf("Error path table: %f\n", errPathTable);
	printf("min col host: %d | min col device: %d\n", outMinColIdxHost, outMinColIdxDevice);
	printf("Error seam pos: %f\n", errPos);

	char * outFileNameBase = strtok(argv[2], "."); // Get rid of extension
	writePnm(outPixels, 1, width, height, concatStr(outFileNameBase, "_device.pnm"));
	// for(int i = 0; i < height; i++){
	// 	printf("host pos: %d, %d | device pos: %d %d\n", hostSeamPos[i*2], hostSeamPos[i*2+1], deviceSeamPos[i*2], deviceSeamPos[i*2 +  1]);
	// }


	free(inPixels);
	free(hostSeamPos);
	free(deviceSeamPos);
	free(outCostTableHost);
	free(outPathTableHost);
	free(outCostTableDevice);
	free(outPathTableDevice);
}
