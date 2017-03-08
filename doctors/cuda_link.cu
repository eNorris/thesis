#include "cuda_link.h"

float **init_gpu(int nx, int ny, float *cpu_data)
{
    if(nx <= 0 || ny <= 0)
        return NULL;

    std::cout << "Initializing GPU resources" << std::endl;

    // Check the number of GPU resources
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    for(unsigned int i = 0; i < nDevices; i++)
    {
        // Find a gpu
        cudaDeviceProp props;
        checkCudaErrors(cudaGetDeviceProperties(&props, i));

        std::cout << "Device " << i << ": " << props.name << " with compute "
             << props.major << "." << props.minor << " capability" << std::endl;
        std::cout << "Max threads per block: " << props.maxThreadsPerBlock << std::endl;
        std::cout << "Max grid size: " << props.maxGridSize[0] << " x " << props.maxGridSize[1] << " x " << props.maxGridSize[2] << std::endl;
        std::cout << "Memory Clock Rate (KHz): " << props.memoryClockRate << std::endl;
        std::cout << "Memory Bus Width (bits): " << props.memoryBusWidth << std::endl;
        std::cout << "Peak Memory Bandwidth (GB/s): " << (2.0*props.memoryClockRate*(props.memoryBusWidth/8)/1.0e6) << '\n' << std::endl;
    }

    std::clock_t start = std::clock();
    float *gpu_data1;
    float *gpu_data2;
    cudaMalloc(&gpu_data1, nx*ny*sizeof(float));
    cudaMemcpyAsync(gpu_data1, cpu_data, nx*ny*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&gpu_data2, nx*ny*sizeof(float));
    std::cout << "Copy Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC/1000)
         << " ms" << std::endl;

    float **gpu_datas = new float*[2];
    gpu_datas[0] = gpu_data1;
    gpu_datas[1] = gpu_data2;

    return gpu_datas;
}

void updateCpuData(float *data_cpu, float *data_gpu, int nx, int ny)
{
    if(cudaMemcpyAsync(data_cpu, data_gpu, nx*ny*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        printf("updateCpuData: Cuda Error!");
}

int launch_diffusionKernel(int nx, int ny, float *prevdata, float *data)
{
    if(nx > 1024 || ny > 2014)
        printf("launch_diffusionKernel() GPU dimension exceeded!!!!");

    dim3 dimGrid(nx);
    dim3 dimBlock(ny);

    testKernel4<<<dimGrid, dimBlock>>>(prevdata, data, nx, ny);
    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
}
