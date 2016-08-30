#include "ctdatamanager.h"

#include <fstream>
#include <cstdint>
#include <QDebug>
#include <fstream>

CtDataManager::CtDataManager() : m_valid(false)
{
    setup();
}

CtDataManager::~CtDataManager()
{

}

//Mesh CtDataReader::parse_256_256_64_16(std::string filename)
Mesh *CtDataManager::parse16(int xbins, int ybins, int zbins, std::string filename)
{
    Mesh *m = new Mesh;
    long tbins = xbins * ybins * zbins;

    m->uniform(xbins, ybins, zbins, 50.0, 50.0, 12.5);
    m->initCtVariables();

    std::vector<u_int16_t> zoneIds;
    zoneIds.resize(tbins);
    //zoneIds.resize(50);

    std::ifstream fin(filename.c_str(), std::ios::binary);

    if(fin.good())
    {
        int numx = fin.tellg()/2;
        qDebug() << "numx = " << numx;

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
    int YX = 256 * 256;
    int X = 256;
    for(int i = 0; i < xbins; i++)
        for(int j = 0; j < ybins; j++)
            for(int k = 0; k < zbins; k++)
            {
                m->ct[gindx] = zoneIds[k*YX + j*X + i];
                gindx++;
            }

    return ctNumberToQuickCheck(m);
    //return ctNumberToHumanPhantom(m);
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

    for(unsigned int i = 0; i < mesh->voxelCount(); i++)
    {
        int ctv = mesh->ct[i];

        // Determine the density
        if(ctv <= 55)
            mesh->density[i] = 0.001 * (1.02 * ctv - 7.65);
        else
            mesh->density[i] = 0.001 * (0.58 * ctv + 467.79);

        // Determine the material
        mesh->zoneId[i] = hounsfieldRangePhantom19.size() - 1;  // Last bin is "illegal"
        for(unsigned int j = 0; j < hounsfieldRangePhantom19.size(); j++)
        {
            //int ctvm = ctv - 1024;
            //int hsf = hounsfieldRangePhantom19[j];
            //bool truth = ctvm < hsf;
            if(ctv - 1024 < hounsfieldRangePhantom19[j])
            {
                mesh->zoneId[i] = j;
                break;
            }
        }
    }

    return mesh;
}

void CtDataManager::setup()
{
    hounsfieldRangePhantom19.push_back(-950); // 1 - air
    hounsfieldRangePhantom19.push_back(-100); // 2 - lung
    hounsfieldRangePhantom19.push_back(15);   // 3 - adipose/adrenal
    hounsfieldRangePhantom19.push_back(129);  // 4 - intestine/connective tissue
    hounsfieldRangePhantom19.push_back(200);  // 5 - bone
    hounsfieldRangePhantom19.push_back(300);
    hounsfieldRangePhantom19.push_back(400);
    hounsfieldRangePhantom19.push_back(500);
    hounsfieldRangePhantom19.push_back(600);
    hounsfieldRangePhantom19.push_back(700);  // 10
    hounsfieldRangePhantom19.push_back(800);
    hounsfieldRangePhantom19.push_back(900);
    hounsfieldRangePhantom19.push_back(1000);
    hounsfieldRangePhantom19.push_back(1100);
    hounsfieldRangePhantom19.push_back(1200);  // 15
    hounsfieldRangePhantom19.push_back(1300);
    hounsfieldRangePhantom19.push_back(1400);
    hounsfieldRangePhantom19.push_back(1500);
    hounsfieldRangePhantom19.push_back(3000);  // 19
    hounsfieldRangePhantom19.push_back(3000);  // 20 - Illegal
}


