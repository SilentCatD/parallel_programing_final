#include <stdio.h>
#include <stdint.h>

// - Use shared memory and constant memory in 2 convo steps
// - Use cuda stream to parallel 2 somber convo step

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


void seamOnDeivce(unsigned char* inPixels, int width, int height, unsigned char* outPixels, dim3 blockSize = dim3(1, 1)){

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
	dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);

	// grayScale
	unsigned char *d_outGrayScale;
	CHECK(cudaMalloc(&d_outGrayScale, nBytes));

	int s_inPixelsSize = ((blockSize.x + FILTER_WIDTH - 1) * (blockSize.y + FILTER_WIDTH - 1)) * sizeof(unsigned char);

	// xSombel
	int xSombelFilter[] = {1, 0, -1, 2, 0, -2, 1, 0, -1}; 
	int *d_xSombelOut;
    cudaStream_t xSombelConvoStream;
    CHECK(cudaStreamCreate(&xSombelConvoStream));
	CHECK(cudaMemcpyToSymbol(dc_xSombelFilter, xSombelFilter, sizeof(xSombelFilter)));
	CHECK(cudaMalloc(&d_xSombelOut, width * height * sizeof(int)));

	// ySombel	
	int ySombelFilter[] = {1, 2, 1, 0, 0, 0, -1, -2, -1}; 
	int * d_ySombelOut;
    cudaStream_t ySombelConvoStream;
    CHECK(cudaStreamCreate(&ySombelConvoStream));
	CHECK(cudaMemcpyToSymbol(dc_ySombelFilter, ySombelFilter, sizeof(ySombelFilter)));
	CHECK(cudaMalloc(&d_ySombelOut, width * height * sizeof(int)));

	// energy
	int* d_energy;
	CHECK(cudaMalloc(&d_energy, width * height * sizeof(int)));

	// Execute
	timer.Start();

    // grayScale
	convertRgb2GrayKernel<<<gridSize, blockSize>>>(d_inPixels, width, height, d_outGrayScale);

	cudaDeviceSynchronize();
	// xSombel
	convoImageKernel<<<gridSize, blockSize, s_inPixelsSize, xSombelConvoStream>>>(d_outGrayScale, width, height, FILTER_WIDTH, d_xSombelOut, true);

	// ySombel	
	convoImageKernel<<<gridSize, blockSize, s_inPixelsSize, ySombelConvoStream>>>(d_outGrayScale, width, height, FILTER_WIDTH, d_ySombelOut, false);
	cudaDeviceSynchronize();

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
	CHECK(cudaFree(d_ySombelOut));
	CHECK(cudaFree(d_energy));
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
void seamOnHost(unsigned char* inPixels, int width, int height, unsigned char* outPixels){

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
	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time of host: %f ms\n\n", time);

	// copy result out
	for(int i = 0; i < width * height; i++){
		outPixels[i] = energy[i];
	}

	free(grayScale);
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
	unsigned char * inPixels, *hostOutPixels, *deviceOutPixels;
	readPnm(argv[1], width, height, inPixels);
	printf("\nImage size (width x height): %i x %i\n", width, height);

	hostOutPixels = (unsigned char*) malloc(width * height);
	seamOnHost(inPixels, width, height, hostOutPixels);

	deviceOutPixels = (unsigned char*) malloc(width * height);
	seamOnDeivce(inPixels, width, height, deviceOutPixels, dim3(32, 32));


	double correct = checkCorrect(hostOutPixels, deviceOutPixels, width, height);
	printf("ERROR: %f", correct);

    char * outFileNameBase = strtok(argv[2], "."); // Get rid of extension
	writePnm(hostOutPixels, 1, width, height, concatStr(outFileNameBase, "_out_host.pnm"));
	writePnm(deviceOutPixels, 1, width, height, concatStr(outFileNameBase, "_out_device.pnm"));
	free(inPixels);
	free(hostOutPixels);
	free(deviceOutPixels);
}
