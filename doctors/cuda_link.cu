#include "cuda_link.h"

#include "quadrature.h"
#include "mesh.h"
#include "xsection.h"
#include "sourceparams.h"

float **init_gpu(int groups, int voxels, int angles)
{
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

    float **gpu_datas = new float*[nDevices+1];
    *(gpu_datas[0]) = nDevices;

    for(unsigned int i = 0; i < nDevices; i++)
    {
        if(cudaMalloc(&gpu_datas[i+1], groups*voxels*angles*sizeof(float)) != cudaSuccess)
            std::cout << "init_gpu threw an error while allocating CUDA memory" << std::endl;
    }

    std::cout << "Alloc Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC/1000)
         << " ms" << std::endl;

    return gpu_datas;
}

void updateCpuData(float *data_cpu, float *data_gpu, size_t elements)
{
    if(cudaMemcpyAsync(data_cpu, data_gpu, elements*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        printf("updateCpuData: Cuda Error!");
}

int launch_isoRayKernel(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const std::vector<RAY_T> *uflux, const SourceParams *params)
{
    dim3 dimGrid(5);
    dim3 dimBlock(5);

    isoRayKernel<<<dimGrid, dimBlock>>>(NULL, NULL, 1, 2);
    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
}
