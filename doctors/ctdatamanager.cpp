#include "ctdatamanager.h"

#include <fstream>
#include <cstdint>
#include <QDebug>
#include <fstream>

#include "materialutils.h"

#include <iostream>
#include "histogram.h"

CtDataManager::CtDataManager() : m_valid(false)
{

}

CtDataManager::~CtDataManager()
{

}

Mesh *CtDataManager::parse16(int xbins, int ybins, int zbins, std::string filename)
{
    Mesh *m = new Mesh;
    long tbins = xbins * ybins * zbins;

    m->uniform(xbins, ybins, zbins, 50.0, 50.0, 12.5);
    m->initCtVariables();

    std::vector<U16_T> zoneIds;
    zoneIds.resize(tbins);

    std::ifstream fin(filename.c_str(), std::ios::binary);

    if(fin.good())
    {
        for(int i = 0; i < tbins; i++)
        {
            fin.read((char*)&zoneIds[i], 2);
        }
    }
    else
    {
        return NULL;
    }

    int gindx = 0;
    int YX = xbins * ybins;
    int X = xbins;
    for(int i = 0; i < xbins; i++)
        for(int j = 0; j < ybins; j++)
            for(int k = 0; k < zbins; k++)
            {
                m->ct[gindx] = zoneIds[k*YX + j*X + i];
                gindx++;
            }
    return ctNumberToHumanPhantom(m);
}

Mesh *CtDataManager::ctNumberToQuickCheck(Mesh *mesh)
{
    float maxi = -1;
    float mini = 1E10;
    for(unsigned int i = 0; i < mesh->voxelCount(); i++)
    {
        int ctv = mesh->ct[i];

        // Determine the density
        if(ctv <= 55)
            mesh->density[i] = 0.001 * (1.02 * ctv - 7.65);
        else
            mesh->density[i] = 0.001 * (0.58 * ctv + 467.79);

        if(mesh->density[i] > maxi)
            maxi = mesh->density[i];
        if(mesh->density[i] < mini)
            mini = mesh->density[i];

        // Determine the material
        mesh->zoneId[i] = mesh->density[i]*1.68;
        if(mesh->zoneId[i] > 63)
            mesh->zoneId[i] = 63;
    }

    qDebug() << "Max: " << maxi << "   Min: " << mini;

    return mesh;
}

Mesh *CtDataManager::ctNumberToHumanPhantom(Mesh *mesh)
{
    std::vector<float> atomPerG;

    std::vector<std::vector<float> > hounsfieldRangePhantom19Fractions;

    // Convert the weight fractions to atom fractions for each material
    for(unsigned int i = 0; i < MaterialUtils::hounsfieldRangePhantom19.size(); i++)
        hounsfieldRangePhantom19Fractions.push_back(MaterialUtils::weightFracToAtomFrac(MaterialUtils::hounsfieldRangePhantom19Elements, MaterialUtils::hounsfieldRangePhantom19Weights[i]));

    // Convert atom fraction to atom density for each material
    for(unsigned int i = 0; i < MaterialUtils::hounsfieldRangePhantom19.size(); i++)
        atomPerG.push_back(MaterialUtils::atomsPerGram(MaterialUtils::hounsfieldRangePhantom19Elements, hounsfieldRangePhantom19Fractions[i]));

    int offset = 1000;
    for(unsigned int i = 0; i < mesh->voxelCount(); i++)
    {
        // Raw data minus offset
        int ctv = mesh->ct[i] - offset;  // Underlying data is 0-2500 instead of -1000-1500

        // Determine the density (atom density is after the zoneId calculation)
        if(ctv <= 55)
        {
            if(ctv <= -1000)
            {
                mesh->density[i] = 0.001225f;
            }
            else
            {
                //mesh->density[i] = 0.001 * (ctv + offset);
                mesh->density[i] = 0.0010186f * ctv + 1.013812f;
            }
            //mesh->density[i] = 0.001 * (1.02 * ctv - 7.65);
        }
        else
        {
            mesh->density[i] = 0.000578402f * ctv + 1.103187f;
            //mesh->density[i] = 0.001 * (0.6 * ctv + offset);
            //mesh->density[i] = 0.001 * (0.58 * ctv + 467.79);
        }

        // Determine the material
        mesh->zoneId[i] = static_cast<U16_T>(MaterialUtils::hounsfieldRangePhantom19.size() - 1);  // Last bin is "illegal"
        for(unsigned int j = 0; j < MaterialUtils::hounsfieldRangePhantom19.size()-1; j++)
        {
            if(ctv < MaterialUtils::hounsfieldRangePhantom19[j])
            {
                mesh->zoneId[i] = j;
                break;
            }
        }

        if(mesh->zoneId[i] >= atomPerG.size())
        {
            qDebug() << "EXPLODE!";
        }

        if(mesh->zoneId[i] == 0)
        {
            mesh->density[i] = 0.001225f;  // force air density because it is very sensitive to miscalibrations
        }

        mesh->atomDensity[i] = mesh->density[i] * atomPerG[mesh->zoneId[i]] * 1.0E-24f;  // g/cc * @/g * cm^2/b = @/cm-b

        if(mesh->atomDensity[i] <= 1.0E-10f)
        {
            qDebug() << "Got zero density";
        }
    }

    return mesh;
}



