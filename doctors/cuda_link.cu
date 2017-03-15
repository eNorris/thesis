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

int *alloc_gpuInt(const int gpuId, const int elements, const int *data)
{
    int cudaerr;
    if((cudaerr = cudaSetDevice(gpuId)) != cudaSuccess)
        std::cout << "alloc_gpuInt failed to set the device with error code: " << cudaerr << std::endl;

    int *gpu_data;
    if((cudaerr = cudaMalloc(&gpu_data, elements*sizeof(int))) != cudaSuccess)
        std::cout << "alloc_gpuInt threw an error while allocating CUDA memory with error code: " << cudaerr << std::endl;

    if(data != NULL)
    {
        if((cudaerr = cudaMemcpyAsync(gpu_data, data, elements*sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess)
            std::cout << "alloc_gpuInt failed while copying data with error code: " << cudaerr << std::endl;
    }

    return gpu_data;
}

float *alloc_gpuFloat(const int gpuId, const int elements, const float *cpuData)
{
    int cudaerr;
    if((cudaerr = cudaSetDevice(gpuId)) != cudaSuccess)
        std::cout << "alloc_gpuFloat failed to set the device with error code: " << cudaerr << std::endl;

    float *gpuData;
    if((cudaerr = cudaMalloc(&gpuData, elements*sizeof(float))) != cudaSuccess)
        std::cout << "alloc_gpuFloat threw an error while allocating CUDA memory with error code: " << cudaerr << std::endl;

    if(cpuData != NULL)
    {
        if((cudaerr = cudaMemcpyAsync(gpuData, cpuData, elements*sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess)
            std::cout << "alloc_gpuFloat failed while copying data with error code: " << cudaerr << std::endl;
    }

    return gpuData;
}

void release_gpu(int gpuId, float *gpu_data)
{
    int cudaerr;
    if((cudaerr = cudaSetDevice(gpuId)) != cudaSuccess)
        std::cout << "release_gpu (float) failed to set the device with error code: " << cudaerr << std::endl;

    if((cudaerr = cudaFree(gpu_data)) != cudaSuccess)
        std::cout << "relase_gpu (float) threw an error while deallocating CUDA memory with error code: " << cudaerr << std::endl;
}

void release_gpu(int gpuId, int *gpu_data)
{
    int cudaerr;
    if((cudaerr = cudaSetDevice(gpuId)) != cudaSuccess)
        std::cout << "release_gpu (int) failed to set the device with error code: " << cudaerr << std::endl;

    if((cudaerr = cudaFree(gpu_data)) != cudaSuccess)
        std::cout << "relase_gpu (int) threw an error while deallocating int CUDA memory with error code: " << cudaerr << std::endl;
}

void updateCpuData(int gpuId, float *cpuData, float *gpuData, size_t elements, int cpuOffset)
{
    int cudaerr;
    if((cudaerr = cudaSetDevice(gpuId)) != cudaSuccess)
        std::cout << "updateCpuData (float) failed to set the device with error code: " << cudaerr << std::endl;

    if((cudaerr = cudaMemcpyAsync(cpuData+cpuOffset, gpuData, elements*sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess)
        std::cout << "updateCpuData (float) MemcpyAsync failed with error code: " << cudaerr << std::endl;
}

void updateCpuData(int gpuId, int *cpuData, int *gpuData, size_t elements, int cpuOffset)
{
    int cudaerr;
    if((cudaerr = cudaSetDevice(gpuId)) != cudaSuccess)
        std::cout << "updateCpuData (int) failed to set the device with error code: " << cudaerr << std::endl;

    if((cudaerr = cudaMemcpyAsync(cpuData+cpuOffset, gpuData, elements*sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess)
        std::cout << "updateCpuData (int) MemcpyAsync failed with error code: " << cudaerr << std::endl;
}

int launch_isoRayKernel(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const SolverParams *solPar, const SourceParams *srcPar, std::vector<RAY_T> *uflux)
{
    if(uflux == NULL)
    {
        std::cout << "STOP!" << std::endl;
        return -1;
    }

    int gpuId = 0;

    // Allocate memory space for the solution vector
    float *gpuUflux = alloc_gpuFloat(gpuId, mesh->voxelCount() * xs->groupCount(), NULL);

    // Copy the xyzNode values
    float *gpuXNodes = alloc_gpuFloat(gpuId, mesh->xNodes.size(), &mesh->xNodes[0]);
    float *gpuYNodes = alloc_gpuFloat(gpuId, mesh->xNodes.size(), &mesh->yNodes[0]);
    float *gpuZNodes = alloc_gpuFloat(gpuId, mesh->xNodes.size(), &mesh->zNodes[0]);

    // Copy the dxyz values
    float *gpuDx = alloc_gpuFloat(gpuId, mesh->dx.size(), &mesh->dx[0]);
    float *gpuDy = alloc_gpuFloat(gpuId, mesh->dy.size(), &mesh->dy[0]);
    float *gpuDz = alloc_gpuFloat(gpuId, mesh->dz.size(), &mesh->dz[0]);

    // Copy the zone id number
    int *gpuZoneId = alloc_gpuInt(gpuId, mesh->zoneId.size(), &mesh->zoneId[0]);

    // Copy the atom density
    float *gpuAtomDensity = alloc_gpuFloat(gpuId, mesh->atomDensity.size(), &mesh->atomDensity[0]);

    // Copy the xs data
    float *gpuTot1d = alloc_gpuFloat(gpuId, xs->m_tot1d.size(), &xs->m_tot1d[0]);

    // Copy the source strength
    float *gpuSrcStrength = alloc_gpuFloat(gpuId, srcPar->spectraIntensity.size(), &srcPar->spectraIntensity[0]);

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

    std::cout << "Grid: " << dimGrid.x << "x" << dimGrid.y << ",   Block: " << dimBlock.x << "x" << dimBlock.y << std::endl;

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

    updateCpuData(gpuId, &(*uflux)[0], gpuUflux, elements);
    //int cudaerr;
    //if((cudaerr = cudaMemcpy(gpuUflux, &(*uflux)[0], elements*sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess)
    //    std::cout << "launch_isoRayKernel failed while copying flux from GPU to CPU with error code "<< cudaerr << std::endl;

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

int launch_isoSolKernel(const Quadrature *quad, const Mesh *mesh, const XSection *xs, const SolverParams *solPar, const SourceParams *srcPar, std::vector<RAY_T> *uFlux)
{
    if(uflux == NULL)
    {
        std::cout << "STOP!" << std::endl;
        return -1;
    }

    int gpuId = 0;

    /*
    // Allocate memory space for the solution vector
    float *gpuUflux = alloc_gpuFloat(gpuId, mesh->voxelCount() * xs->groupCount(), NULL);

    // Copy the xyzNode values
    float *gpuXNodes = alloc_gpuFloat(gpuId, mesh->xNodes.size(), &mesh->xNodes[0]);
    float *gpuYNodes = alloc_gpuFloat(gpuId, mesh->xNodes.size(), &mesh->yNodes[0]);
    float *gpuZNodes = alloc_gpuFloat(gpuId, mesh->xNodes.size(), &mesh->zNodes[0]);

    // Copy the dxyz values
    float *gpuDx = alloc_gpuFloat(gpuId, mesh->dx.size(), &mesh->dx[0]);
    float *gpuDy = alloc_gpuFloat(gpuId, mesh->dy.size(), &mesh->dy[0]);
    float *gpuDz = alloc_gpuFloat(gpuId, mesh->dz.size(), &mesh->dz[0]);

    // Copy the zone id number
    int *gpuZoneId = alloc_gpuInt(gpuId, mesh->zoneId.size(), &mesh->zoneId[0]);

    // Copy the atom density
    float *gpuAtomDensity = alloc_gpuFloat(gpuId, mesh->atomDensity.size(), &mesh->atomDensity[0]);

    // Copy the xs data
    float *gpuTot1d = alloc_gpuFloat(gpuId, xs->m_tot1d.size(), &xs->m_tot1d[0]);

    // Copy the source strength
    float *gpuSrcStrength = alloc_gpuFloat(gpuId, srcPar->spectraIntensity.size(), &srcPar->spectraIntensity[0]);
    */

    //int ixSrc, iySrc, izSrc;

    std::clock_t startTime = std::clock();

    const int maxIterations = 25;
    const SOL_T epsilon = 0.01f;

    std::vector<SOL_T> *scalarFlux = new std::vector<SOL_T>(xs->groupCount() * mesh->voxelCount(), 0.0f);
    std::vector<SOL_T> tempFlux(mesh->voxelCount(), 0.0f);
    std::vector<SOL_T> preFlux(mesh->voxelCount(), -100.0f);
    std::vector<SOL_T> totalSource(mesh->voxelCount(), -100.0f);
    std::vector<SOL_T> outboundFluxX(mesh->voxelCount(), -100.0f);
    std::vector<SOL_T> outboundFluxY(mesh->voxelCount(), -100.0f);
    std::vector<SOL_T> outboundFluxZ(mesh->voxelCount(), -100.0f);
    std::vector<SOL_T> extSource(xs->groupCount() * mesh->voxelCount(), 0.0f);

    std::vector<SOL_T> errMaxList;
    std::vector<std::vector<SOL_T> > errList;
    std::vector<int> converganceIters;
    std::vector<SOL_T> converganceTracker;

    errMaxList.resize(xs->groupCount());
    errList.resize(xs->groupCount());
    converganceIters.resize(xs->groupCount());
    converganceTracker.resize(xs->groupCount());

    //SOL_T influxX = 0.0f;
    //SOL_T influxY = 0.0f;
    //SOL_T influxZ = 0.0f;

    //int ejmp = mesh->voxelCount() * quad->angleCount();
    //int ajmp = mesh->voxelCount();
    //int xjmp = mesh->xjmp();
    //int yjmp = mesh->yjmp();

    //bool downscatterFlag = false;

    if(uFlux == NULL && srcPar == NULL)
    {
        std::cout << "uFlux and srcPar cannot both be NULL" << std::endl;
        return;
        //qDebug() << "uFlux and params cannot both be NULL";
    }

    bool noDownscatterYet = true;
    unsigned int highestEnergy = 0;

    while(noDownscatterYet)
    {
        SOL_T dmax = 0.0;
        unsigned int vc = mesh->voxelCount();
        for(unsigned int ira = 0; ira < vc; ira++)
        {
            dmax = (dmax > (*uFlux)[highestEnergy*vc + ira]) ? dmax : (*uFlux)[highestEnergy*vc + ira];
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
            return;
        }
    }

    if(uFlux != NULL)
    {
        std::cout << "Loading uncollided flux into external source" << std::endl;
        // If there is an uncollided flux provided, use it, otherwise, calculate the external source
        //unsigned int xx= xs->groupCount();
        for(unsigned int ie = highestEnergy; ie < xs->groupCount(); ie++)  // Sink energy
            for(unsigned int ri = 0; ri < mesh->voxelCount(); ri++)
                for(unsigned int iep = highestEnergy; iep <= ie; iep++) // Source energy
                    //                               [#]   =                        [#/cm^2]      * [cm^3]        *  [b]                               * [1/b-cm]
                    extSource[ie*mesh->voxelCount() + ri] += (*uFlux)[iep*mesh->voxelCount() + ri] * mesh->vol[ri] * xs->scatxs2d(mesh->zoneId[ri], iep, ie, 0) * mesh->atomDensity[ri];

        OutWriter::writeArray("externalSrc.dat", extSource);
    }
    else
    {
        std::cout << "Building external source" << std::endl;
        int srcIndxE = xs->groupCount() - 1;
        int srcIndxX = 32;
        int srcIndxY = 4;  //mesh->yElemCt/2;
        int srcIndxZ = 8;
        //                                                                              [#] = [#]
        extSource[srcIndxE * mesh->voxelCount() + srcIndxX*xjmp + srcIndxY*yjmp + srcIndxZ] = 1.0;
    }

    std::cout << "Solving " << mesh->voxelCount() * quad->angleCount() * xs->groupCount() << " elements in phase space" << std::endl;


    for(unsigned int ie = highestEnergy; ie < xs->groupCount(); ie++)  // for every energy group
    {
        int iterNum = 1;
        SOL_T maxDiff = 1.0;

        for(unsigned int i = 0; i < totalSource.size(); i++)
            totalSource[i] = 0;

        // Calculate the down-scattering source
        for(unsigned int iie = highestEnergy; iie < ie; iie++)
            for(unsigned int ir = 0; ir < mesh->voxelCount(); ir++)
            {
                int zidIndx = mesh->zoneId[ir];
                //         [#]    +=  [#]          *      [#/cm^2]                               * [b]                          * [1/b-cm]                * [cm^3]
                totalSource[ir] += m_4pi_inv*(*scalarFlux)[iie*mesh->voxelCount() + ir] * xs->scatxs2d(zidIndx, iie, ie, 0) * mesh->atomDensity[ir] * mesh->vol[ir]; //xsref(ie-1, zidIndx, 0, iie));
            }

        // Add the external source
        for(unsigned int ri = 0; ri < mesh->voxelCount(); ri++)
        {
            //  [#]         +=  [#]
            totalSource[ri] += extSource[ie*mesh->voxelCount() + ri];
        }

        while(iterNum <= maxIterations && maxDiff > epsilon)  // while not converged
        {
            //qDebug() << "Iteration #" << iterNum;

            preFlux = tempFlux;  // Store flux for previous iteration


            // Clear for a new sweep
            for(unsigned int i = 0; i < tempFlux.size(); i++)
                tempFlux[i] = 0;

            for(unsigned int iang = 0; iang < quad->angleCount(); iang++)  // for every angle
            {
                //qDebug() << "Angle #" << iang;

                // Find the correct direction to sweep
                int izStart = 0;                  // Sweep start index
                int diz = 1;                      // Sweep direction
                if(quad->eta[iang] < 0)           // Condition to sweep backward
                {
                    izStart = mesh->zElemCt - 1;  // Start at the far end
                    diz = -1;                     // Sweep toward zero
                }

                int iyStart = 0;
                int diy = 1;
                if(quad->zi[iang] < 0)
                {
                    iyStart = mesh->yElemCt - 1;
                    diy = -1;
                }

                int ixStart = 0;
                int dix = 1;
                if(quad->mu[iang] < 0)
                {
                    ixStart = mesh->xElemCt - 1;
                    dix = -1;
                }

                dim3 dimGrid(mesh->xElemCt, mesh->yElemCt);
                dim3 dimBlock(mesh->zElemCt);
                std::cout << "Grid: " << dimGrid.x << "x" << dimGrid.y << ",   Block: " << dimBlock.x << "x" << dimBlock.y << std::endl;

                isoSolKernel<<<dimGrid, dimBlock>>>(
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

                for(unsigned int i = 0; i < tempFlux.size(); i++)
                {
                    (*scalarFlux)[ie*mesh->voxelCount() + i] = tempFlux[i];
                }

                emit signalNewSolverIteration(scalarFlux);

                unsigned int xTracked = mesh->xElemCt/2;
                unsigned int yTracked = mesh->yElemCt/2;
                unsigned int zTracked = mesh->zElemCt/2;
                converganceTracker.push_back((*scalarFlux)[ie*mesh->voxelCount() + xTracked*xjmp + yTracked*yjmp + zTracked]);

            } // end of all angles

            maxDiff = -1.0E35f;
            for(unsigned int i = 0; i < tempFlux.size(); i++)
            {
                //float z = qAbs((tempFlux[i] - preFlux[i])/tempFlux[i]);
                maxDiff = qMax(maxDiff, qAbs((tempFlux[i] - preFlux[i])/tempFlux[i]));

                if(std::isnan(maxDiff))
                    qDebug() << "Found a diff nan!";
            }
            qDebug() << "Max diff = " << maxDiff;

            errList[ie].push_back(maxDiff);
            errMaxList[ie] = maxDiff;
            converganceIters[ie] = iterNum;

            for(unsigned int i = 0; i < tempFlux.size(); i++)
            {
                (*scalarFlux)[ie*mesh->voxelCount() + i] = tempFlux[i];
            }

            iterNum++;
        } // end not converged
    }  // end each energy group

    qDebug() << "Time to complete: " << (std::clock() - startTime)/(double)(CLOCKS_PER_SEC/1000) << " ms";

    for(unsigned int i = 0; i < errList.size(); i++)
    {
        std::cout << "Group: " << i << "   maxDiff: " << errMaxList[i] << "   Iterations: " << converganceIters[i] << '\n';
        for(unsigned int j = 0; j < errList[i].size(); j++)
            std::cout << errList[i][j] << '\t';
        std::cout << std::endl;
    }

    emit signalNewSolverIteration(scalarFlux);
    emit signalSolverFinished(scalarFlux);

    size_t elements = mesh->voxelCount() * xs->groupCount();
    //uflux = new RAY_T[elements];
    uflux->resize(elements);
    //cudaDeviceSynchronize();

    updateCpuData(gpuId, &(*uflux)[0], gpuUflux, elements);
    //int cudaerr;
    //if((cudaerr = cudaMemcpy(gpuUflux, &(*uflux)[0], elements*sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess)
    //    std::cout << "launch_isoRayKernel failed while copying flux from GPU to CPU with error code "<< cudaerr << std::endl;

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
