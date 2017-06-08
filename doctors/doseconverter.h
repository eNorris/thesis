#ifndef DOSECONVERTER_H
#define DOSECONVERTER_H

#include <vector>
#include <QDebug>

#include <string>
#include "xsection.h"
#include "mesh.h"

#define MYMIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MYMAX(X, Y) (((X) > (Y)) ? (X) : (Y))

class DoseConverter
{
public:
    DoseConverter();
    ~DoseConverter();

    template<typename T>
    static std::vector<T> doseIcrp(const XSection &xs, const Mesh &mesh, const std::vector<T> &uflux, const std::vector<T> &cflux);

    template<typename T>
    static std::vector<T> doseEdep(const XSection &xs, const Mesh &mesh, const std::vector<T> &uflux, const std::vector<T> &cflux);
};

template<typename T>
std::vector<T> DoseConverter::doseIcrp(const XSection &xs, const Mesh &mesh, const std::vector<T> &uflux, const std::vector<T> &cflux)
{
    std::vector<T> icrpDose(mesh.voxelCount(), 0.0);

    if(uflux.size() != mesh.voxelCount()*xs.groupCount() || cflux.size() != mesh.voxelCount()*xs.groupCount())
    {
        qDebug() << "Incorrect mesh size in ICRP conversion! uflux: " << uflux.size() << "  cflux: " << cflux.size() << "  target: " << mesh.voxelCount()*xs.groupCount();
        for(unsigned int i = 0; i < icrpDose.size(); i++)
            icrpDose[i] = -1;
        return icrpDose;
    }

    // From ICRP 116, Table A.1 pg 126 for the ROT geometry
    const std::vector<float> icrpDe = {0.01,   0.015,  0.02,   0.03,  0.04,
                                 0.05,   0.06,   0.07,   0.08,  0.10,
                                 0.15,   0.2,    0.3,    0.4,   0.5,
                                 0.511,  0.6,    0.662,  0.8,   1.0,
                                 1.117,  1.33,   1.5,    2.0,   3.0,
                                 4.0,    5.0,    6.0,    6.129, 8.0,
                                 10.0,   15.0,   20.0,   30.0, 40.0};
    const std::vector<float> icrpDf = {0.0337, 0.0664, 0.0986, 0.158, 0.199,
                                0.226,   0.248,  0.273,  0.297, 0.355,
                                0.528,   0.721,  1.12,   1.52,  1.92,
                                1.96,    2.3,    2.54,   3.04,  3.72,
                                4.10,    4.75,   5.24,   6.55,  8.84,
                                10.8,    12.7,   14.4,   14.6,  17.6,
                                20.6,    27.7,   34.4,   46.1,  56.0};

    std::vector<float> conversionFactor;
    conversionFactor.resize(xs.groupCount(), 0.0);

    for(unsigned int ie = 0; ie < xs.groupCount(); ie++)
    {
        int iStart = 0;
        int iEnd = 0;

        while(icrpDe[iStart+1] <= (xs.gbounds[ie+1]/1e6))
        {
            float q = icrpDe[iStart+1];
            float v = xs.gbounds[ie+1]/1e6;
            iStart++;
        }

        while(icrpDe[iEnd+1] < xs.gbounds[ie]/1e6)
            iEnd++;

        float sumDose = 0.0f;
        for(unsigned int i = iStart; i <= iEnd; i++)
        {
            float leftE = MYMAX(xs.gbounds[ie+1]/1e6, icrpDe[i]);
            float rightE = MYMIN(xs.gbounds[ie]/1e6, icrpDe[i+1]);
            float dE = rightE - leftE;
            float slope = (icrpDf[i+1] - icrpDf[i])/(icrpDe[i+1] - icrpDe[i]);
            float leftDf = slope * (leftE - icrpDe[i]) + icrpDf[i];  // y = m(x - xi) + yi
            float rightDf = slope * (rightE - icrpDe[i]) + icrpDf[i];
            float dosePart = (leftDf + rightDf)/2;
            sumDose += dosePart * dE;
        }

        float sumE = (xs.gbounds[ie] - xs.gbounds[ie+1])/1e6;

        conversionFactor[ie] = sumDose/sumE;
    }

    for(unsigned int ix = 0; ix < mesh.xElemCt; ix++)
        for(unsigned int iy = 0; iy < mesh.yElemCt; iy++)
            for(unsigned int iz = 0; iz < mesh.zElemCt; iz++)
            {
                float dose = 0.0;
                int ir = ix*mesh.yElemCt*mesh.zElemCt + iy*mesh.zElemCt + iz;
                for(unsigned int ie = 0; ie < xs.groupCount(); ie++)
                    dose += (uflux[ie*mesh.voxelCount() + ir] + cflux[ie*mesh.voxelCount() + ir]) * conversionFactor[ie];
                icrpDose[ir] = dose;
            }

    return icrpDose;
}

template<typename T>
std::vector<T> DoseConverter::doseEdep(const XSection &xs, const Mesh &mesh, const std::vector<T> &uflux, const std::vector<T> &cflux)
{
    std::vector<T> depDose(mesh.voxelCount(), 0.0);

    if(uflux.size() != mesh.voxelCount()*xs.groupCount() || cflux.size() != mesh.voxelCount()*xs.groupCount())
    {
        qDebug() << "Incorrect mesh size in ICRP conversion! uflux: " << uflux.size() << "  cflux: " << cflux.size() << "  target: " << mesh.voxelCount()*xs.groupCount();
        for(unsigned int i = 0; i < depDose.size(); i++)
            depDose[i] = -1;
        return depDose;
    }

    T gramToKg = 0.001;
    T evToJ = 1.60218e-19;

    for(unsigned int ir = 0; ir < mesh.voxelCount(); ir++)
    {
        if(ir == 32*64*16 + 32*16 + 8)
        {
            qDebug() << "Hi";
        }
        int zid = mesh.zoneId[ir];
        for(unsigned int ie = 0; ie < xs.groupCount(); ie++)
        {
            //[eV/cm] =   Sigma_a = Sigma_t - Sigma-s [b]            * [1/b-cm]             * [eV]
            T sigE = (xs.totXs1d(zid, ie)-xs.scatXs1d(zid, ie)) * mesh.atomDensity[ir] * (xs.gbounds[ie] - xs.gbounds[ie+1]);

            // For every energy group lower compute g->h
            for(unsigned int iep = ie+1; iep < xs.groupCount(); iep++)
            {
                // [1/cm]=     [b]                          * [1/b-cm]
                T sig_gh = xs.scatxs2d(zid, ie, iep, 0) * mesh.atomDensity[ir];

                // [eV]
                T E0 = (xs.gbounds[ie] + xs.gbounds[ie+1])/2;   // Energy before collision
                T Ep = (xs.gbounds[iep] + xs.gbounds[iep+1])/2; // energy after collision

                //       [1/cm] * [eV]
                sigE += sig_gh * (E0 - Ep);
            }

            // [eV/cm^3]        1/cm^2]                                                          * [eV/cm]
            depDose[ir] += (uflux[ie*mesh.voxelCount() + ir] + cflux[ie*mesh.voxelCount() + ir]) * sigE;
        }

        // [J/kg]     [J/eV] / ([kg/g]  * [g/cm^3]) = [J-cm^3/eV-kg]
        depDose[ir] *= evToJ / (gramToKg * mesh.density[ir]);
    }

    return depDose;
}


#endif // DOSECONVERTER_H
