#include "cuda_link.h"

#include "quadrature.h"
#include "mesh.h"
#include "xsection.h"
#include "sourceparams.h"
#include "solverparams.h"

void reportGpuData()
{
    std::cout << "Reporting GPU resources" << std::endl;

    // Check the number of GPU resources
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    std::cout << "Found " << nDevices << " CUDA devices" << std::endl;

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
}

int *alloc_gpuInt(int gpuId, int elements, int *data)
{
    if(cudaSetDevice(gpuId) != cudaSuccess)
        std::cout << "alloc_gpu failed to set the device" << std::endl;

    //for(unsigned int i = 0; i < nDevices; i++)
    //{
    int *gpu_data;
    if(cudaMalloc(&gpu_data, elements*sizeof(int)) != cudaSuccess)
        std::cout << "alloc_gpuInt threw an error while allocating CUDA memory" << std::endl;

    if(data != NULL)
    {
        if(cudaMemcpy(gpu_data, data, elements*sizeof(int), HOST_TO_DEVICE) != cudaSuccess)
            std::cout << "alloc_gpuInt failed while copying data" << std::endl;
    }
    //}

    return gpu_data;
}

float *alloc_gpuFloat(int gpuId, int elements)
{
    if(cudaSetDevice(gpuId) != cudaSuccess)
        std::cout << "alloc_gpu failed to set the device" << std::endl;

    int *gpu_data;
    if(cudaMalloc(&gpu_data, elements*sizeof(float)) != cudaSuccess)
        std::cout << "alloc_gpuFloat threw an error while allocating CUDA memory" << std::endl;

    if(data != NULL)
    {
        if(cudaMemcpy(gpu_data, data, elements*sizeof(float), HOST_TO_DEVICE) != cudaSuccess)
            std::cout << "alloc_gpuFloat failed while copying data" << std::endl;
    }

    return gpu_data;
}

void release_gpu(float *gpu_data)
{
    //int nGpu = (*gpus[0]);
    //for(int i = 0; i < nGpu; i++)
    //{
    if(cudaFree(&gpu_data) != cudaSuccess)
        std::cout << "relase_gpu threw an error while deallocating CUDA memory" << std::endl;
    //}
    //int **gpu_datas = new int*[nDevices+1];
    //(*gpu_datas[0]) = nDevices;
    //memcpy(gpu_datas[0], &nDevices, sizeof(int));
    //(*gpu_datas[0]) = nDevices;  // Turns nDevices into an address of a float

    //for(unsigned int i = 0; i < nDevices; i++)
    //{
    //    if(cudaMalloc(&gpu_datas[i+1], elements/2*sizeof(float)) != cudaSuccess)
    //        std::cout << "init_gpu threw an error while allocating CUDA memory" << std::endl;
    //}
}

void updateCpuData(float *data_cpu, float *data_gpu, size_t elements)
{
    if(cudaMemcpyAsync(data_cpu, data_gpu, elements*sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        printf("updateCpuData: Cuda Error!");
}

int launch_isoRayKernel(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const SolverParams *solPar, const SourceParams *srcPar, const std::vector<RAY_T> *uflux)
{
    dim3 dimGrid(5);
    dim3 dimBlock(5);

    /*
    float *uflux,
    float *xNodes, float *yNodes, float zNodes,
    float *dx, float *dy, float *dz,
    int *zoneId,
    float *atomDensity,
    float *tot1d,
    float *srcStrength
    int groups,
    flost sx, float sy, float sz,
    srcIndxX, int srcIndxY, int srcIndxZ,
    */

    int gpuId = 0;

    // Allocate memory space for the solution vector
    float *gpuUflux = alloc_gpuFloat(gpuId, mesh->voxelCount() * xs->groupCount());

    // Copy the xyzNode values
    float gpuXNodes = alloc_gpuFloat(gpuId, mesh->xNodes.size(), &mesh->xNodes[0]);
    float gpuYNodes = alloc_gpuFloat(gpuId, mesh->xNodes.size(), &mesh->yNodes[0]);
    float gpuZNodes = alloc_gpuFloat(gpuId, mesh->xNodes.size(), &mesh->zNodes[0]);

    // Copy the dxyz values
    float gpuDx = alloc_gpuFloat(gpuId, mesh->dx.size(), &mesh->dx[0]);
    float gpuDy = alloc_gpuFloat(gpuId, mesh->dy.size(), &mesh->dy[0]);
    float gpuDz = alloc_gpuFloat(gpuId, mesh->dz.size(), &mesh->dz[0]);

    // Copy the zone id number
    int gpuZoneId = alloc_gpuInt(gpuId, mesh->zoneId.size(), &mesh->zoneId[0]);

    // Copy the atom density
    float gpuAtomDensity = alloc_gpuFloat(gpuId, mesh->zoneId.size(), &mesh->zoneId[0]);

    // Copy the xs data
    float gpuTot1d = alloc_gpuFloat(gpuId, xs->m_tot1d.size(), &xs->m_tot1d[0]);

    // Copy the source strength
    float gpuSrcStrength = alloc_gpuFloat(gpuId, srcPar->spectraIntensity.size(), &srcPar->spectraIntensity[0]);

    int ixSrc, iySrc, izSrc;

    ixSrc = 5;
    iySrc = 5;
    izSrc = 5;

    isoRayKernel<<<dimGrid, dimBlock>>>(
                gpuUflux,
                gpuXNodes, gpuYNodes, gpuZNodes,
                gpuDx, gpuDy, gpuDz,
                gpuZoneId,
                gpuAtomDensity,
                gpuTot1d,
                gpuSrcStrength,
                xs->groupCount(),
                srcPar->sourceX, srcPar->sourceY, srcPar->sourceZ,
                ixSrc, iySrc, izSrc);
    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
}
