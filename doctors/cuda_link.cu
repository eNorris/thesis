#include "cuda_link.h"

#include "quadrature.h"
#include "mesh.h"
#include "xsection.h"
#include "sourceparams.h"
#include "solverparams.h"

#include "outwriter.h"
//#include <string>
#include <stdio.h>

#include <cuda_profiler_api.h>

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
        std::cout << "Peak Memory Bandwidth (GB/s): " << (2.0*props.memoryClockRate*(props.memoryBusWidth/8)/1.0e6) << std::endl;

        int cores = 0;
        int mp = props.multiProcessorCount;
        switch (props.major){
        case 2: // Fermi
            if (props.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (props.minor == 1) cores = mp * 128;
            else if (props.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
        }

        std::cout << "SMs: " << mp << std::endl;
        std::cout << "CUDA Cores: " << cores << '\n' << std::endl;

    }
}

int *alloc_gpuInt(const int gpuId, const int elements, const int *data)
{
    cudaError_t cudaerr;
    if((cudaerr = cudaSetDevice(gpuId)) != cudaSuccess)
        std::cout << "alloc_gpuInt failed to set the device with error code: " << cudaerr << ": " << cudaGetErrorString(cudaerr) << std::endl;

    int *gpu_data;
    if((cudaerr = cudaMalloc(&gpu_data, elements*sizeof(int))) != cudaSuccess)
        std::cout << "alloc_gpuInt threw an error while allocating CUDA memory with error code: " << cudaerr << ": " << cudaGetErrorString(cudaerr) << std::endl;

    if(data != NULL)
    {
        if((cudaerr = cudaMemcpyAsync(gpu_data, data, elements*sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess)
            std::cout << "alloc_gpuInt failed while copying data with error code: " << cudaerr << ": " << cudaGetErrorString(cudaerr) << std::endl;
    }

    return gpu_data;
}
/*
RAY_T *alloc_gpuFloat(const int gpuId, const int elements, const RAY_T *cpuData)
{
    cudaError_t cudaerr;
    if((cudaerr = cudaSetDevice(gpuId)) != cudaSuccess)
        std::cout << "alloc_gpuFloat failed to set the device with error code: " << cudaerr << ": " << cudaGetErrorString(cudaerr) << std::endl;

    RAY_T *gpuData;
    if((cudaerr = cudaMalloc(&gpuData, elements*sizeof(RAY_T))) != cudaSuccess)
        std::cout << "alloc_gpuFloat threw an error while allocating CUDA memory with error code: " << cudaerr << ": " << cudaGetErrorString(cudaerr) << std::endl;

    if(cpuData != NULL)
    {
        if((cudaerr = cudaMemcpyAsync(gpuData, cpuData, elements*sizeof(RAY_T), cudaMemcpyHostToDevice)) != cudaSuccess)
            std::cout << "alloc_gpuFloat failed while copying data with error code: " << cudaerr << ": " << cudaGetErrorString(cudaerr) << std::endl;
    }

    return gpuData;
}
*/

/*
void release_gpu(int gpuId, float *gpu_data)
{
    cudaError_t cudaerr;
    if((cudaerr = cudaSetDevice(gpuId)) != cudaSuccess)
        std::cout << "release_gpu (float) failed to set the device with error code: " << cudaerr << ": " << cudaGetErrorString(cudaerr) << std::endl;

    if((cudaerr = cudaFree(gpu_data)) != cudaSuccess)
        std::cout << "relase_gpu (float) threw an error while deallocating CUDA memory with error code: " << cudaerr << ": " << cudaGetErrorString(cudaerr) << std::endl;
}

void release_gpu(int gpuId, int *gpu_data)
{
    cudaError_t cudaerr;
    if((cudaerr = cudaSetDevice(gpuId)) != cudaSuccess)
        std::cout << "release_gpu (int) failed to set the device with error code: " << cudaerr << ": " << cudaGetErrorString(cudaerr) << std::endl;

    if((cudaerr = cudaFree(gpu_data)) != cudaSuccess)
        std::cout << "relase_gpu (int) threw an error while deallocating int CUDA memory with error code: " << cudaerr << ": " << cudaGetErrorString(cudaerr) << std::endl;
}
*/

/*
void updateCpuData(int gpuId, float *cpuData, float *gpuData, size_t elements, int cpuOffset)
{
    cudaError_t cudaerr;
    if((cudaerr = cudaSetDevice(gpuId)) != cudaSuccess)
        std::cout << "updateCpuData (float) failed to set the device with error code: " << cudaerr << ": " << cudaGetErrorString(cudaerr) << std::endl;

    if((cudaerr = cudaMemcpyAsync(cpuData+cpuOffset, gpuData, elements*sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess)
        std::cout << "updateCpuData (float) MemcpyAsync failed with error code: " << cudaerr << ": " << cudaGetErrorString(cudaerr) << std::endl;
}

void updateCpuData(int gpuId, int *cpuData, int *gpuData, size_t elements, int cpuOffset)
{
    cudaError_t cudaerr;
    if((cudaerr = cudaSetDevice(gpuId)) != cudaSuccess)
        std::cout << "updateCpuData (int) failed to set the device with error code: " << cudaerr << ": " << cudaGetErrorString(cudaerr) << std::endl;

    if((cudaerr = cudaMemcpyAsync(cpuData+cpuOffset, gpuData, elements*sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess)
        std::cout << "updateCpuData (int) MemcpyAsync failed with error code: " << cudaerr << ": " << cudaGetErrorString(cudaerr) << std::endl;
}
*/

int launch_isoRayKernel(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const SolverParams *solPar, const SourceParams *srcPar, std::vector<RAY_T> *uflux)
{
    reportGpuData();

    if(uflux == NULL)
    {
        std::cout << "STOP!" << std::endl;
        return -1;
    }

    int gpuId = 0;

    // Allocate memory space for the solution vector
    //std::cout << "Allocating uflux" << std::endl;
    RAY_T *gpuUflux = alloc_gpuFloat<RAY_T>(gpuId, mesh->voxelCount() * xs->groupCount(), NULL);

    // Copy the xyzNode values
    float *gpuXNodes = alloc_gpuFloat<float>(gpuId, mesh->xNodes.size(), &mesh->xNodes[0]);
    float *gpuYNodes = alloc_gpuFloat<float>(gpuId, mesh->xNodes.size(), &mesh->yNodes[0]);
    float *gpuZNodes = alloc_gpuFloat<float>(gpuId, mesh->xNodes.size(), &mesh->zNodes[0]);

    // Copy the dxyz values
    float *gpuDx = alloc_gpuFloat<float>(gpuId, mesh->dx.size(), &mesh->dx[0]);
    float *gpuDy = alloc_gpuFloat<float>(gpuId, mesh->dy.size(), &mesh->dy[0]);
    float *gpuDz = alloc_gpuFloat<float>(gpuId, mesh->dz.size(), &mesh->dz[0]);

    // Copy the zone id number
    int *gpuZoneId = alloc_gpuInt(gpuId, mesh->zoneId.size(), &mesh->zoneId[0]);

    // Copy the atom density
    float *gpuAtomDensity = alloc_gpuFloat<float>(gpuId, mesh->atomDensity.size(), &mesh->atomDensity[0]);

    // Copy the xs data
    float *gpuTot1d = alloc_gpuFloat<float>(gpuId, xs->m_tot1d.size(), &xs->m_tot1d[0]);

    // Copy the source strength
    //std::cout << "Allocating source strength" << std::endl;
    float *gpuSrcStrength = alloc_gpuFloat<float>(gpuId, srcPar->spectraIntensity.size(), &srcPar->spectraIntensity[0]);

    //int ixSrc, iySrc, izSrc;

    unsigned int ixSrc = 0;
    unsigned int iySrc = 0;
    unsigned int izSrc = 0;

    while(mesh->xNodes[ixSrc+1] < srcPar->sourceX)
        ixSrc++;

    while(mesh->yNodes[iySrc+1] < srcPar->sourceY)
        iySrc++;

    while(mesh->zNodes[izSrc+1] < srcPar->sourceZ)
        izSrc++;

    dim3 dimGrid(mesh->xElemCt, mesh->yElemCt);
    dim3 dimBlock(mesh->zElemCt);

    //std::cout << "Grid: " << dimGrid.x << "x" << dimGrid.y << ",   Block: " << dimBlock.x << "x" << dimBlock.y << std::endl;

    isoRayKernel<<<dimGrid, dimBlock>>>(
                gpuUflux,
                gpuXNodes, gpuYNodes, gpuZNodes,
                gpuDx, gpuDy, gpuDz,
                gpuZoneId,
                gpuAtomDensity,
                gpuTot1d,
                gpuSrcStrength,
                xs->groupCount(),
                mesh->xElemCt, mesh->yElemCt, mesh->zElemCt,
                srcPar->sourceX, srcPar->sourceY, srcPar->sourceZ,
                ixSrc, iySrc, izSrc);

    size_t elements = mesh->voxelCount() * xs->groupCount();
    //uflux = new RAY_T[elements];
    uflux->resize(elements);
    //cudaDeviceSynchronize();

    updateCpuDataBlocking(gpuId, &(*uflux)[0], gpuUflux, elements);
    //OutWriter::writeArray("uflux.dat", *uflux);

    release_gpu(gpuId, gpuUflux);
    release_gpu(gpuId, gpuXNodes);
    release_gpu(gpuId, gpuYNodes);
    release_gpu(gpuId, gpuZNodes);
    release_gpu(gpuId, gpuDx);
    release_gpu(gpuId, gpuDy);
    release_gpu(gpuId, gpuDz);
    release_gpu(gpuId, gpuZoneId);
    release_gpu(gpuId, gpuAtomDensity);
    release_gpu(gpuId, gpuTot1d);
    release_gpu(gpuId, gpuSrcStrength);

    std::cout << "Most recent CUDA Error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    //if(cudaFree(gpu_data) != cudaSuccess)
    //    std::cout << "alloc_gpuInt failed while copying data" << std::endl;

    return EXIT_SUCCESS;
}

int launch_isoSolKernel(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const SolverParams *solPar, const SourceParams *srcPar, const std::vector<RAY_T> *cpuUFlux, std::vector<SOL_T> *cpuCFlux)
{
    //std::cout << "Launching solver kernel" << std::endl;
    if(cpuUFlux == NULL)
    {
        std::cout << "STOP!" << std::endl;
        return -1;
    }

    if(cpuCFlux == NULL)
    {
        std::cout << "STOP!" << std::endl;
        return -2;
    }

    int gpuId = 0;

    std::vector<SOL_T> errMaxList;
    std::vector<std::vector<SOL_T> > errList;
    std::vector<std::vector<SOL_T> > errIntList;
    std::vector<int> converganceIters;
    std::vector<SOL_T> converganceTracker;

    errMaxList.resize(xs->groupCount());
    errList.resize(xs->groupCount());
    errIntList.resize(xs->groupCount());
    converganceIters.resize(xs->groupCount());
    converganceTracker.resize(xs->groupCount());

    std::clock_t startTime = std::clock();

    const int maxIterations = 25;
    const SOL_T epsilon = static_cast<SOL_T>(0.001);

    cpuCFlux->resize(xs->groupCount() * mesh->voxelCount());
    std::vector<SOL_T> cpuCFluxTmp(mesh->voxelCount(), static_cast<SOL_T>(0.0));

    //std::vector<SOL_T> errMaxList;
    //errMaxList.resize(xs->groupCount());

    if(cpuUFlux == NULL && srcPar == NULL)
    {
        std::cout << "uFlux and srcPar cannot both be NULL" << std::endl;
        return 55;
    }

    // Computed the highest energy group actually used
    bool noDownscatterYet = true;
    unsigned int highestEnergy = 0;

    //std::cout << "About to do high check" << std::endl;

    while(noDownscatterYet)
    {
        SOL_T dmax = 0.0;
        unsigned int vc = mesh->voxelCount();
        for(unsigned int ir = 0; ir < vc; ir++)
        {
            dmax = (dmax > (*cpuUFlux)[highestEnergy*vc + ir]) ? dmax : (*cpuUFlux)[highestEnergy*vc + ir];
        }
        if(dmax <= 0.0)
        {
            std::cout << "No external source or downscatter, skipping energy group " << highestEnergy << std::endl;
            highestEnergy++;
        }
        else
        {
            noDownscatterYet = false;
        }

        if(highestEnergy >= xs->groupCount())
        {
            std::cout << "Zero flux everywhere from the raytracer" << std::endl;
            return 57;
        }
    }

    // Allocate GPU resources for the external source computation
    RAY_T *gpuUFlux = alloc_gpuFloat<RAY_T>(gpuId, cpuUFlux->size(), &(*cpuUFlux)[0]);
    SOL_T *gpuColFlux = alloc_gpuFloat<SOL_T>(gpuId, cpuCFlux->size(), NULL);

    float *gpuVol = alloc_gpuFloat<float>(gpuId, mesh->vol.size(), &mesh->vol[0]);
    float *gpuAtomDensity = alloc_gpuFloat<float>(gpuId, mesh->atomDensity.size(), &mesh->atomDensity[0]);
    int   *gpuZoneId = alloc_gpuInt(gpuId, mesh->zoneId.size(), &mesh->zoneId[0]);

    float *gpuScatXs2d = alloc_gpuFloat<float>(gpuId, xs->m_scat2d.size(), &xs->m_scat2d[0]);

    // Allocate additional GPU resources for the solver
    SOL_T *gpuTempFlux = alloc_gpuFloat<SOL_T>(gpuId, mesh->voxelCount(), NULL);
    SOL_T *gpu1stSource = alloc_gpuFloat<SOL_T>(gpuId, mesh->voxelCount(), NULL);
    SOL_T *gpuTotalSource = alloc_gpuFloat<SOL_T>(gpuId, mesh->voxelCount(), NULL);

    //unsigned int anglesPerOctant = quad->angleCount()/8;
    //SOL_T *gpuOutboundFluxX = alloc_gpuFloat<SOL_T>(gpuId, mesh->yElemCt * mesh->zElemCt * anglesPerOctant, NULL);
    //SOL_T *gpuOutboundFluxY = alloc_gpuFloat<SOL_T>(gpuId, mesh->yElemCt * mesh->zElemCt * anglesPerOctant, NULL);
    //SOL_T *gpuOutboundFluxZ = alloc_gpuFloat<SOL_T>(gpuId, mesh->yElemCt * mesh->zElemCt * anglesPerOctant, NULL);
    SOL_T *gpuOutboundFluxX = alloc_gpuFloat<SOL_T>(gpuId, mesh->voxelCount(), NULL);
    SOL_T *gpuOutboundFluxY = alloc_gpuFloat<SOL_T>(gpuId, mesh->voxelCount(), NULL);
    SOL_T *gpuOutboundFluxZ = alloc_gpuFloat<SOL_T>(gpuId, mesh->voxelCount(), NULL);

    float *gpuAxy = alloc_gpuFloat<float>(gpuId, mesh->Axy.size(), &mesh->Axy[0]);
    float *gpuAxz = alloc_gpuFloat<float>(gpuId, mesh->Axz.size(), &mesh->Axz[0]);
    float *gpuAyz = alloc_gpuFloat<float>(gpuId, mesh->Ayz.size(), &mesh->Ayz[0]);

    float *gpuMu = alloc_gpuFloat<float>(gpuId, quad->mu.size(), &quad->mu[0]);
    float *gpuEta = alloc_gpuFloat<float>(gpuId, quad->eta.size(), &quad->eta[0]);
    float *gpuXi = alloc_gpuFloat<float>(gpuId, quad->zi.size(), &quad->zi[0]);
    float *gpuWt = alloc_gpuFloat<float>(gpuId, quad->wt.size(), &quad->wt[0]);

    float *gpuTotXs1d = alloc_gpuFloat<float>(gpuId, xs->m_tot1d.size(), &xs->m_tot1d[0]);

    // Zero the scalar flux
    int erblocks = 64;
    int ergrids = cpuCFlux->size() / erblocks;
    if(cpuCFlux->size() % erblocks != 0)
        ergrids += 1;  // Account for lengths not divisible by 64

    zeroKernel<<<dim3(ergrids), dim3(erblocks)>>>(cpuCFlux->size(), gpuColFlux);

    // Generate the sweep index block
    int totalSubsweeps = mesh->xElemCt + mesh->yElemCt + mesh->zElemCt - 2;
    std::vector<int> threadIndexToGlobalIndex(mesh->voxelCount());
    std::vector<int> subSweepStartIndex(totalSubsweeps);
    std::vector<int> subSweepVoxelCount(totalSubsweeps);

    // Trivial edge cases that aren't computed during the loop
    subSweepStartIndex[0] = 0;
    threadIndexToGlobalIndex[0] = 0;
    subSweepVoxelCount[totalSubsweeps-1] = 1;

    for(unsigned int iSubSweep = 0; iSubSweep < totalSubsweeps; iSubSweep++)
    {
        int iSubSweepPrev = iSubSweep - 1;
        int C = (iSubSweepPrev+1) * (iSubSweepPrev+2) / 2;

        int dx = max(iSubSweepPrev+1 - (signed)mesh->xElemCt, 0);
        int dy = max(iSubSweepPrev+1 - (signed)mesh->yElemCt, 0);
        int dz = max(iSubSweepPrev+1 - (signed)mesh->zElemCt, 0);
        int dxy = max(iSubSweepPrev+1 - (signed)mesh->xElemCt - (signed)mesh->yElemCt, 0);
        int dxz = max(iSubSweepPrev+1 - (signed)mesh->xElemCt - (signed)mesh->zElemCt, 0);
        int dyz = max(iSubSweepPrev+1 - (signed)mesh->yElemCt - (signed)mesh->zElemCt, 0);

        int Lx = dx * (dx + 1) / 2;
        int Ly = dy * (dy + 1) / 2;
        int Lz = dz * (dz + 1) / 2;

        int Gxy = dxy * (dxy + 1) / 2;
        int Gxz = dxz * (dxz + 1) / 2;
        int Gyz = dyz * (dyz + 1) / 2;

        int voxPrevSubSweep = C - Lx - Ly - Lz + Gxy + Gxz + Gyz;
        subSweepStartIndex[iSubSweep] = subSweepStartIndex[iSubSweepPrev] + voxPrevSubSweep;
        subSweepVoxelCount[iSubSweepPrev] = voxPrevSubSweep;

        int voxelsSoFar = 0;
        for(int ix = 0; ix <= min(mesh->xElemCt-1, iSubSweep); ix++)
            for(int iy = 0; iy <= min(mesh->yElemCt-1, iSubSweep-ix); iy++)
            {
                int iz = iSubSweep - ix - iy;
                if(iz >= mesh->zElemCt)
                    continue;

                int ir = ix*mesh->yElemCt*mesh->zElemCt + iy*mesh->zElemCt + iz;

                threadIndexToGlobalIndex[subSweepStartIndex[iSubSweep] + voxelsSoFar] = ir;
                voxelsSoFar++;
            }
    }

    int *gpuThreadIndexToGlobalIndex = alloc_gpuInt(gpuId, threadIndexToGlobalIndex.size(), &threadIndexToGlobalIndex[0]);

    dim3 dimGrid(mesh->xElemCt, mesh->yElemCt);
    dim3 dimBlock(mesh->zElemCt);

    for(unsigned int ie = highestEnergy; ie < xs->groupCount(); ie++)  // for every energy group
    {
        std::cout << "ie=" << ie << std::endl;
        int iterNum = 1;
        SOL_T maxDiff = 1.0;
        SOL_T totDiff = 1.0E10;    // Should be very large
        SOL_T totDiffPre = 1.0E11; // Should be larger than totDiff

        // Needs to be done before the first clearSweepKernel<<<>>> call
        int rblocks = 64;
        int rgrids = mesh->voxelCount() / rblocks;
        if(mesh->voxelCount() % rblocks != 0)
            rgrids += 1;  // Account for lengths not divisible by 64
        zeroKernel<<<dim3(rgrids), dim3(rblocks)>>>(mesh->voxelCount(), gpuTempFlux);
        //zeroKernel<<<dim3(rgrids), dim3(rblocks)>>>(mesh->voxelCount(), gpu1stSource);

        isoSrcKernel<<<dimGrid, dimBlock>>>(
                                          gpuUFlux,
                                          gpu1stSource,
                                          gpuVol, gpuAtomDensity, gpuZoneId,
                                          gpuScatXs2d,
                                          mesh->voxelCount(), xs->groupCount(), solPar->pn+1, highestEnergy, ie,
                                          mesh->xElemCt, mesh->yElemCt, mesh->zElemCt);

        //cudaDeviceSynchronize();


        //std::cout << "About to write the source results" << std::endl;
        //std::vector<float> cpuExtSrc;
        //cpuExtSrc.resize(mesh->voxelCount());
        //updateCpuDataBlocking(gpuId, &cpuExtSrc[0], gpu1stSource, mesh->voxelCount());
        //char ieString[256];
        //sprintf(ieString, "%d", ie);
        //OutWriter::writeArray(std::string("gpuExtSrc") + ieString + ".dat", cpuExtSrc);

        downscatterKernel<<<dimGrid, dimBlock>>>(
                gpuTotalSource,
                highestEnergy, ie,
                mesh->xElemCt, mesh->yElemCt, mesh->zElemCt, xs->groupCount(), solPar->pn+1,
                gpuZoneId,
                gpuColFlux,
                gpuScatXs2d,
                gpuAtomDensity, gpuVol,
                gpu1stSource);

        for(unsigned int i = 0; i < cpuCFluxTmp.size(); i++)
            cpuCFluxTmp[i] = 0.0f;

        //printf("Begin Total crit=%e\n", totDiff/totDiffPre);

        while(iterNum <= maxIterations && maxDiff > epsilon && totDiff/totDiffPre < 1.0)  // while not converged
        {
            //std::cout << "it: " << iterNum << std::endl;
            clearSweepKernel<<<dimGrid, dimBlock>>>(
                    gpuColFlux, gpuTempFlux,
                    mesh->xElemCt, mesh->yElemCt, mesh->zElemCt, ie);

            for(unsigned int i = 0; i < cpuCFluxTmp.size(); i++)
            {
                //cpuCFlux[ie*mesh->voxelCount() + i] = cpuCFluxTmp[i];
                cpuCFluxTmp[i] = 0.0f;
            }

            for(unsigned int iang = 0; iang < quad->angleCount(); iang++)  // for every angle
            //for(unsigned int io = 0; io < 8; io++)  // for every octant
            {
                // Find the correct direction to sweep
                int diz = 1;                      // Sweep direction
                if(quad->eta[iang] < 0)           // Condition to sweep backward
                {
                    diz = -1;                     // Sweep toward zero
                }

                int diy = 1;
                if(quad->zi[iang] < 0)
                {
                    diy = -1;
                }

                int dix = 1;
                if(quad->mu[iang] < 0)
                {
                    dix = -1;
                }

                //int dix = io / 4 == 0 ? 1 : -1;  // + x x, positive are first
                //int diy = (io/2) % 2 == 0 ? 1 : -1;  // x + x, alternate every other octant
                //int diz = io % 2 == 0 ? 1 : -1;  // x x + alternate every octant

                //std::cout << "io = " << io << ", dix = " << dix << ", diy = " << diy << ", diz = " << diz << std::endl;

                for(unsigned int subSweepId = 0; subSweepId < totalSubsweeps; subSweepId++)
                {

                    //std::cout << "Subsweep " << subSweepId << std::endl;

                    int raise = subSweepVoxelCount[subSweepId] % 64 == 0 ? 0 : 1;
                    dim3 dimGridS(subSweepVoxelCount[subSweepId] / 64 + raise);
                    dim3 dimBlockS(64);

                    isoSolKernel<<<dimGridS, dimBlockS>>>(
                          gpuColFlux, gpuTempFlux,
                          gpuTotalSource,
                          gpuTotXs1d, gpuScatXs2d,
                          gpuAxy, gpuAxz, gpuAyz,
                          gpuZoneId, gpuAtomDensity, gpuVol,
                          gpuMu, gpuEta, gpuXi, gpuWt,
                          gpuOutboundFluxX, gpuOutboundFluxY, gpuOutboundFluxZ,
                          ie, iang,
                          mesh->xElemCt, mesh->yElemCt, mesh->zElemCt, xs->groupCount(), quad->angleCount(), solPar->pn+1, // Send pn+1 to send number of pn groups
                          dix, diy, diz,
                          subSweepStartIndex[subSweepId], subSweepVoxelCount[subSweepId], gpuThreadIndexToGlobalIndex);

                    //cudaDeviceSynchronize();

                    //std::cout << "Launched subSweepId=" << subSweepId <<  "(" << dimGridS.x << ", " << dimGridS.y << " : " << dimBlockS.x << ", " << dimBlockS.y << " )" << std::endl;
                    //std::cin.ignore(1024, '\n');
                    //std::cout << "Ran angle " << iang << std::endl;
                    //std::cin.get();
                }

                updateCpuData(gpuId, &cpuCFluxTmp[0], gpuTempFlux, mesh->voxelCount());
            } // end of all angles

            //updateCpuDataBlocking(gpuId, &cpuCFluxTmp[0], gpuTempFlux, mesh->voxelCount());

            //char iterString[3];  // 2 digits + NULL
            //char ieString[3];  // 2 digits + NULL
            //sprintf(iterString, "%d", iterNum);
            //sprintf(ieString, "%d", ie);
            //OutWriter::writeArray(std::string("gpuScalarFlux_") + std::string(ieString) + "_" + std::string(iterString) + ".dat", cpuCFluxTmp);

            // Make sure all kernels and data transfers finish before advancing
            cudaDeviceSynchronize();

            maxDiff = -1.0e35f;
            totDiffPre = totDiff;
            totDiff = 0.0f;
            //float totDiff = 0.0f;
            for(unsigned int i = 0; i < mesh->voxelCount(); i++)
            {
                maxDiff = max( abs((cpuCFluxTmp[i]-(*cpuCFlux)[ie*mesh->voxelCount() + i])/cpuCFluxTmp[i]), maxDiff);
                totDiff += abs(cpuCFluxTmp[i]-(*cpuCFlux)[ie*mesh->voxelCount() + i]);
            }

            for(unsigned int i = 0; i < cpuCFluxTmp.size(); i++)
            {
                (*cpuCFlux)[ie*mesh->voxelCount() + i] = cpuCFluxTmp[i];
            }

            //printf("Diff=%e, tot=%e, pre=%e, condition=%e\n", maxDiff, totDiff, totDiffPre, totDiff/totDiffPre);

            errList[ie].push_back(maxDiff);
            errIntList[ie].push_back(totDiff);
            errMaxList[ie] = maxDiff;
            converganceIters[ie] = iterNum;

            //for(unsigned int i = 0; i < mesh->voxelCount(); i++)
            //    cpuCFluxTmp[i] = (*cpuCFlux)[ie*mesh->voxelCount() + i];

            //std::cout << "Max diff = " << maxDiff << std::endl;

            iterNum++;
        } // end not converged

        if(!(iterNum <= maxIterations))
        {
            std::cout << "Max iterations hit" << std::endl;
        }
        else if(!(maxDiff > epsilon))
        {
            std::cout << "Converged on relative error" << std::endl;
        }
        else
        {
            std::cout << "Converged on precsion bound" << std::endl;
        }
        //iterNum <= maxIterations && maxDiff > epsilon && totDiff/totDiffPre < 1.0

        //printf("End Total crit=%e\n", totDiff/totDiffPre);

    }  // end each energy group

    std::cout << "Time to complete: " << (std::clock() - startTime)/(double)(CLOCKS_PER_SEC/1000) << " ms" << std::endl;

    // Release the GPU resources
    release_gpu(gpuId, gpuUFlux);
    release_gpu(gpuId, gpuZoneId);
    release_gpu(gpuId, gpuAtomDensity);

    release_gpu(gpuId, gpuAxy);
    release_gpu(gpuId, gpuAxz);
    release_gpu(gpuId, gpuAyz);

    release_gpu(gpuId, gpuMu);
    release_gpu(gpuId, gpuEta);
    release_gpu(gpuId, gpuXi);
    release_gpu(gpuId, gpuWt);

    release_gpu(gpuId, gpuTotXs1d);
    release_gpu(gpuId, gpuScatXs2d);

    for(unsigned int i = 0; i < errList.size(); i++)
    {
        std::cout << "%Group: " << i << "   maxDiff: " << errMaxList[i] << "   Iterations: " << converganceIters[i] << '\n';
        std::cout << "gpu" << i << " = [";
        for(unsigned int j = 0; j < errList[i].size(); j++)
            std::cout << errList[i][j] << ",\t";
        std::cout << "];\ngpu" << i << "i = [";
        for(unsigned int j = 0; j < errIntList[i].size(); j++)
            std::cout << errIntList[i][j] << ",\t";
        std::cout << "];" << std::endl;
    }

    std::cout << "Most recent CUDA Error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;

    cudaDeviceReset();
    cudaProfilerStop();

    return EXIT_SUCCESS;
}

/*
template <class T>
void reduce(int size, int threads, int blocks, T *d_idata, T *d_odata)
{
    int numBlocks = 0;
    int numThreads = 0;
    int maxBlocks = 64;
    int maxThreads = 256;
    //getNumBlocksAndThreads(0, size, maxBlocks, maxThreads, numBlocks, numThreads);

    cudaDeviceProp prop;
    //int device;
   // checkCudaErrors(cudaGetDevice(&gpuId));
    checkCudaErrors(cudaGetDeviceProperties(&prop, gpuId));

    numThreads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
    numBlocks = (n + (numThreads * 2 - 1)) / (numThreads * 2);

    if ((float)numThreads*numBlocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
    {
        printf("n is too large, please choose a smaller number!\n");
    }

    if (numBlocks > prop.maxGridSize[0])
    {
        printf("Grid size <%d> excceeds the device capability <%d>, set block size as %d (original %d)\n",
               numBlocks, prop.maxGridSize[0], numThreads*2, numThreads);

        numBlocks /= 2;
        numThreads *= 2;
    }

    numBlocks = MIN(maxBlocks, numBlocks);

    // allocate mem for the result on host side
    T *h_odata = (T *) malloc(numBlocks*sizeof(T));

    printf("%d blocks\n\n", numBlocks);

    // allocate device memory and data
    T *d_idata = NULL;
    T *d_odata = NULL;

    checkCudaErrors(cudaMalloc((void **) &d_idata, bytes));
    checkCudaErrors(cudaMalloc((void **) &d_odata, numBlocks*sizeof(T)));

    // copy data directly to device memory
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_odata, h_idata, numBlocks*sizeof(T), cudaMemcpyHostToDevice));

    // warm-up
    //reduce<T>(size, numThreads, numBlocks, whichKernel, d_idata, d_odata);

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    if (isPow2(size))
    {
        switch (threads)
        {
            case 512:
                reduce6<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 256:
                reduce6<T, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 128:
                reduce6<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 64:
                reduce6<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 32:
                reduce6<T,  32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 16:
                reduce6<T,  16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  8:
                reduce6<T,   8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  4:
                reduce6<T,   4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  2:
                reduce6<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  1:
                reduce6<T,   1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;
        }
    }
    else
    {
        switch (threads)
        {
            case 512:
                reduce6<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 256:
                reduce6<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 128:
                reduce6<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 64:
                reduce6<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 32:
                reduce6<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 16:
                reduce6<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  8:
                reduce6<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  4:
                reduce6<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  2:
                reduce6<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  1:
                reduce6<T,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;
        }
    }
}
*/
/*
void getNumBlocksAndThreads(int gpuId, int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
    //get device capability, to avoid block/grid size excceed the upbound
    cudaDeviceProp prop;
    //int device;
   // checkCudaErrors(cudaGetDevice(&gpuId));
    checkCudaErrors(cudaGetDeviceProperties(&prop, gpuId));

    threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);

    if ((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
    {
        printf("n is too large, please choose a smaller number!\n");
    }

    if (blocks > prop.maxGridSize[0])
    {
        printf("Grid size <%d> excceeds the device capability <%d>, set block size as %d (original %d)\n",
               blocks, prop.maxGridSize[0], threads*2, threads);

        blocks /= 2;
        threads *= 2;
    }

    blocks = MIN(maxBlocks, blocks);
}
*/
