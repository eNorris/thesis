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

    std::clock_t startTime = std::clock();

    const int maxIterations = 25;
    const SOL_T epsilon = 0.01f;

    std::vector<SOL_T> *scalarFlux = new std::vector<SOL_T>(xs->groupCount() * mesh->voxelCount());

    std::vector<SOL_T> errMaxList;
    std::vector<std::vector<SOL_T> > errList;
    std::vector<int> converganceIters;
    std::vector<SOL_T> converganceTracker;

    errMaxList.resize(xs->groupCount());
    errList.resize(xs->groupCount());
    converganceIters.resize(xs->groupCount());
    converganceTracker.resize(xs->groupCount());

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
        for(unsigned int ir = 0; ir < vc; ir++)
        {
            dmax = (dmax > (*uFlux)[highestEnergy*vc + ir]) ? dmax : (*uFlux)[highestEnergy*vc + ir];
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

    // Allocate GPU resources for the external source computation

    float *gpuUFlux = alloc_gpuFloat(gpuId, mesh->voxelCount(), &(*uFlux)[0]);
    float *gpuExtSource = alloc_gpuFloat(gpuId, mesh->voxelCount() * xs->groupCount(), NULL);

    float *gpuVol = alloc_gpuFloat(gpuId, mesh->vol.size(), &mesh->vol[0]);
    float *gpuAtomDensity = alloc_gpuFloat(gpuId, mesh->atomDensity.size(), &mesh->atomDensity[0]);
    int   *gpuZoneId = alloc_gpuInt(gpuId, mesh->zoneId.size(), &mesh->zoneId[0]);

    float *gpuScatXs2d = alloc_gpuFloat(gpuId, xs->m_scat2d.size(), &xs->m_scat2d[0]);

    if(uFlux != NULL)
    {
        //std::cout << "Loading uncollided flux into external source" << std::endl;

        for(unsigned int iSink = highestEnergy; iSink < xs->groupCount(); iSink++)
            isoSrcKernel<<<dimGrid, dimBlock>>>(
                                              gpuUFlux,
                                              gpuExtSource,
                                              gpuVol, gpuAtomDensity, gpuZoneId,
                                              gpuScatXs2d,
                                              mesh->voxelCount(), xs->groupCount(), solPar->pn, highestEnergy, iSink);

        cudaDeviceSynchronize();
        release_gpu(gpuId, gpuUFlux);

        OutWriter::writeArray("externalSrc.dat", extSource);
    }
    else
    {
        abort();
        //std::cout << "Building external source" << std::endl;
        //int srcIndxE = xs->groupCount() - 1;
        //int srcIndxX = 32;
        //int srcIndxY = 4;  //mesh->yElemCt/2;
        //int srcIndxZ = 8;
        //                                                                              [#] = [#]
        //extSource[srcIndxE * mesh->voxelCount() + srcIndxX*xjmp + srcIndxY*yjmp + srcIndxZ] = 1.0;
    }

    std::cout << "Solving " << mesh->voxelCount() * quad->angleCount() * xs->groupCount() << " elements in phase space" << std::endl;

    float *gpuTempFlux = alloc_gpuFloat(gpuId, mesh->voxelCount(), NULL);
    float *gpuPreFlux = alloc_gpuFloat(gpuId, mesh->voxelCount(), NULL);
    float *gpuTotalSource = alloc_gpuFloat(gpuId, mesh->voxelCount(), NULL);
    float *gpuOutboundFluxX = alloc_gpuFloat(gpuId, mesh->voxelCount(), NULL);
    float *gpuOutboundFluxY = alloc_gpuFloat(gpuId, mesh->voxelCount(), NULL);
    float *gpuOutboundFluxZ = alloc_gpuFloat(gpuId, mesh->voxelCount(), NULL);

    dim3 dimGrid(mesh->xElemCt, mesh->yElemCt);
    dim3 dimBlock(mesh->zElemCt);
    std::cout << "Grid: " << dimGrid.x << "x" << dimGrid.y << ",   Block: " << dimBlock.x << "x" << dimBlock.y << std::endl;

    for(unsigned int ie = highestEnergy; ie < xs->groupCount(); ie++)  // for every energy group
    {
        int iterNum = 1;
        SOL_T maxDiff = 1.0;

        zeroKernel<<<dimGrid, dimBlock>>>(mesh->xElemCt, mesh->yElemCt, mesh->zElemCt, gpuTotalSource);

        //for(unsigned int i = 0; i < totalSource.size(); i++)
        //    totalSource[i] = 0;

        // Calculate the down-scattering source
        downscatterKernel<<<dimGrid, dimBlock>>>(
                gpuTotalSource,
                highestEnergy, SINK_GRP,
                mesh->xElemCt, mesh->yElemCt, mesh->zElemCt, xs->groupCount(), solPar->pn,
                gpuZoneId,
                gpuSc,
                gpuScatXs2d,
                gpuAtomDensity, gpuVol,
                gpuExtSource);
        //for(unsigned int iie = highestEnergy; iie < ie; iie++)
        //    for(unsigned int ir = 0; ir < mesh->voxelCount(); ir++)
        //    {
        //        int zidIndx = mesh->zoneId[ir];
        //        //         [#]    +=  [#]          *      [#/cm^2]                               * [b]                          * [1/b-cm]                * [cm^3]
        //        totalSource[ir] += m_4pi_inv*(*scalarFlux)[iie*mesh->voxelCount() + ir] * xs->scatxs2d(zidIndx, iie, ie, 0) * mesh->atomDensity[ir] * mesh->vol[ir]; //xsref(ie-1, zidIndx, 0, iie));
        //    }

        // Add the external source
        //for(unsigned int ri = 0; ri < mesh->voxelCount(); ri++)
        //{
            //  [#]         +=  [#]
        //    totalSource[ri] += extSource[ie*mesh->voxelCount() + ri];
        //}

        while(iterNum <= maxIterations && maxDiff > epsilon)  // while not converged
        {
            //qDebug() << "Iteration #" << iterNum;

            //preFlux = tempFlux;  // Store flux for previous iteration

            clearSweepKernel<<<dimGrid, dimBlock>>>(
                    gpuPreFlux, gpuTempFlux,
                    mesh->xElemCt, mesh->yElemCt, mesh->zElemCt);

            // Clear for a new sweep
            //for(unsigned int i = 0; i < tempFlux.size(); i++)
            //    tempFlux[i] = 0;

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

                isoSolKernel<<<dimGrid, dimBlock>>>(
                      gpuScalarFlux, gpuTempFlux,
                      gpuTotalSource,
                      gpuTotXs1d, gpuScatxs2d,
                      gpuAxy, gpuAxz, gpuAyz,
                      gpuZoneId, gpuAtomDensity, gpuVol,
                      gpuMu, gpuEta, gpuXi, gpuWt,
                      gpuOutboundFluxX, gpuOutboundFluxY, gpuOutboundFluxZ,
                      ie, iang,
                      mesh->xElemCt, mesh->yElemCt, mesh->zElemCt, quad->angleCount());

                // TODO: Why is this done twice?
                //for(unsigned int i = 0; i < tempFlux.size(); i++)
                //{
                //    (*scalarFlux)[ie*mesh->voxelCount() + i] = tempFlux[i];
                //}

                updateCpuData(gpuId, scalarFlux, gpuTempFlux, mesh->voxelCount(), ie*mesh->voxelCount());
                //cudaMemCpy(gpuTempFlux, scalarFlux+ie*mesh->voxelCount(), Cuda)

                // TODO: launch gpu copy kernel
                // TODO: launch async memcpy

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

            // TODO shouldn't something involving preflux and tempflux be here?

            // It's done again here...
            //for(unsigned int i = 0; i < tempFlux.size(); i++)
            //{
            //    (*scalarFlux)[ie*mesh->voxelCount() + i] = tempFlux[i];
            //}

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

    //size_t elements = mesh->voxelCount() * xs->groupCount();
    //uflux = new RAY_T[elements];
    //uflux->resize(elements);
    //cudaDeviceSynchronize();

    //updateCpuData(gpuId, &(*uflux)[0], gpuUflux, elements);
    //int cudaerr;
    //if((cudaerr = cudaMemcpy(gpuUflux, &(*uflux)[0], elements*sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess)
    //    std::cout << "launch_isoRayKernel failed while copying flux from GPU to CPU with error code "<< cudaerr << std::endl;

    // Release the GPU resources
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
