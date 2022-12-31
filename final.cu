#include <stdio.h>
#include <stdint.h>

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

__global__ void convoImageKernel(unsigned char * inPixels, int width, int height, 
        int * filter, int filterWidth, 
        int * outPixels)
{
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	if (c < width && r < height) {
		int outPixel = 0;
		for (int filterR = 0; filterR < filterWidth; filterR++)
		{
			for (int filterC = 0; filterC < filterWidth; filterC++)
				{
					int filterVal = filter[filterR*filterWidth + filterC];
					int inPixelsR = r - filterWidth/2 + filterR;
					int inPixelsC = c - filterWidth/2 + filterC;
					inPixelsR = min(max(0, inPixelsR), height - 1);
					inPixelsC = min(max(0, inPixelsC), width - 1);
					unsigned char inPixel = inPixels[inPixelsR*width + inPixelsC];
					outPixel += filterVal * inPixel;
				}
		}
		outPixels[r*width + c] = outPixel; 
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

void seamOnDeivce(unsigned char* inPixels, int width, int height, unsigned char* outPixels, dim3 blockSize = dim3(1, 1)){

	// Allocate memory
	GpuTimer timer;
	int filterWidth = 3;
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
	printf("GPU name: %s\n", devProp.name);
	printf("GPU compute capability: %d.%d\n", devProp.major, devProp.minor);

	unsigned char* d_inPixels;
	size_t nBytes = width * height * sizeof(unsigned char);
	CHECK(cudaMalloc(&d_inPixels, nBytes*3));
	CHECK(cudaMemcpy(d_inPixels, inPixels, nBytes * 3, cudaMemcpyHostToDevice));
	dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

	// grayScale
	unsigned char *d_outGrayScale;
	CHECK(cudaMalloc(&d_outGrayScale, nBytes));
	
	// xSombel
	int xSombelFilter[] = {1, 0, -1, 2, 0, -2, 1, 0, -1}; 
	int* d_xSombelFilter, *d_xSombelOut;
	CHECK(cudaMalloc(&d_xSombelOut, width * height * sizeof(int)));
	CHECK(cudaMalloc(&d_xSombelFilter, sizeof(xSombelFilter)));
	CHECK(cudaMemcpy(d_xSombelFilter, xSombelFilter, sizeof(xSombelFilter), cudaMemcpyHostToDevice));

	// ySombel	
	int ySombelFilter[] = {1, 2, 1, 0, 0, 0, -1, -2, -1}; 
	int *d_ySombelFilter, * d_ySombelOut;
	CHECK(cudaMalloc(&d_ySombelOut, width * height * sizeof(int)));
	CHECK(cudaMalloc(&d_ySombelFilter, sizeof(ySombelFilter)));
	CHECK(cudaMemcpy(d_ySombelFilter, ySombelFilter, sizeof(ySombelFilter), cudaMemcpyHostToDevice));

	// energy
	int* d_energy;
	CHECK(cudaMalloc(&d_energy, width * height * sizeof(int)));

	// Execute
	timer.Start();

	// grayScale
	convertRgb2GrayKernel<<<gridSize, blockSize>>>(d_inPixels, width, height, d_outGrayScale);
	
	// xSombel
	convoImageKernel<<<gridSize, blockSize>>>(d_outGrayScale, width, height, d_xSombelFilter, filterWidth, d_xSombelOut);

	// ySombel	
	convoImageKernel<<<gridSize, blockSize>>>(d_outGrayScale, width, height, d_ySombelFilter, filterWidth, d_ySombelOut);

	// energy
	energyCalcKernel<<<gridSize, blockSize>>>(d_xSombelOut, d_ySombelOut, width, height, d_energy);

	cudaDeviceSynchronize();
	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time of device: %f ms\n\n", time);

	// copy result back
	int *tmp = (int*)malloc(width * height * sizeof(int));
	CHECK(cudaMemcpy(tmp, d_energy, width * height * sizeof(int), cudaMemcpyDeviceToHost));
	for(int i = 0 ; i < width * height; i++){
		outPixels[i] = tmp[i];
	}

	free(tmp);
	CHECK(cudaFree(d_inPixels));
	CHECK(cudaFree(d_outGrayScale));
	CHECK(cudaFree(d_xSombelOut));
	CHECK(cudaFree(d_xSombelFilter));
	CHECK(cudaFree(d_ySombelOut));
	CHECK(cudaFree(d_ySombelFilter));
	CHECK(cudaFree(d_energy));
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
		if(costTable[idx + i] <=  currentMin){
			currentMin = costTable[idx + i];
			result = i;
		}
	}
	return result;
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

void findSeamOnHost(unsigned char* inPixels, int width, int height, int* seamPos){

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

double checkCorrect(unsigned char* out, unsigned char* out2, int width, int height){
	float err = 0;
	int n =  width * height;
	for (int i = 0; i < n; i++)
		err += abs(out[i] - out2[i]);
	err /= n;
	return err;
}

int main(int argc, char ** argv)
{

	// Read input image file
	int width, height;
	unsigned char * inPixels;
	readPnm(argv[1], width, height, inPixels);
	printf("\nImage size (width x height): %i x %i\n", width, height);

	int *hostSeamPos = (int*) malloc(height * 2 * sizeof(int));
	findSeamOnHost(inPixels, width, height, hostSeamPos);
	for(int i = 0; i < height; i++){
		printf("y: %i, x: %i\n", hostSeamPos[i*2], hostSeamPos[i*2+1]);
	}

	free(inPixels);
	free(hostSeamPos);
}
