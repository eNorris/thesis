#include "ctdatamanager.h"

#include <fstream>
#include <cstdint>
#include <QDebug>
#include <fstream>

#include "materialutils.h"
#include "ctmapdialog.h"

#include <iostream>
#include "histogram.h"

CtDataManager::CtDataManager() : m_valid(false), m_mesh(NULL), m_messageBox(), m_mapDialog(NULL)
{
    m_mapDialog = new CtMapDialog(NULL);

    connect(m_mapDialog, SIGNAL(water()), this, SLOT(onWater()));
    connect(m_mapDialog, SIGNAL(phantom19()), this, SLOT(onPhantom19()));
}

CtDataManager::~CtDataManager(){}

void CtDataManager::parse16(int xbins, int ybins, int zbins, QString filename)
{
    //Mesh *m = new Mesh;
    m_mesh = new Mesh;
    long tbins = xbins * ybins * zbins;

    m_mesh->uniform(xbins, ybins, zbins, 50.0, 50.0, 12.5);
    m_mesh->initCtVariables();

    std::vector<U16_T> zoneIds;
    zoneIds.resize(tbins);

    std::ifstream szChkFin(filename.toStdString().c_str(), std::ios::binary|std::ios::in|std::ios::ate);
    unsigned int szFin = szChkFin.tellg();
    if(szFin != 2*tbins)
    {
        QString errmsg = QString("The specified mesh size (");
                errmsg += QString::number(xbins) + ", " + QString::number(ybins) + ", " + QString::number(zbins) + ") ";
                errmsg += "requires " + QString::number(2*tbins) + " bytes of data to be populated. ";
                errmsg += "The binary file specified (" + filename +  ") only reported " + QString::number(szFin) + " bytes of data.";
                errmsg += "You may either abort further parsing of the data file or ignore this warning (which may result in corrupted geometry data).";
        int resp = QMessageBox::warning(NULL, "Data Size Mismatch", errmsg, QMessageBox::Abort | QMessageBox::Ignore);
        if(resp == QMessageBox::Abort)
            return;
    }
    szChkFin.close();

    std::ifstream fin(filename.toStdString().c_str(), std::ios::binary);

    if(fin.good())
    {
        for(int i = 0; i < tbins; i++)
        {
            if(i % 10000 == 0)
            {
                emit signalMeshUpdate(i);
            }
            fin.read((char*)&zoneIds[i], 2);
        }
    }
    else
    {
        return;
    }

    int gindx = 0;
    int YX = xbins * ybins;
    int X = xbins;
    for(int i = 0; i < xbins; i++)
        for(int j = 0; j < ybins; j++)
            for(int k = 0; k < zbins; k++)
            {
                m_mesh->ct[gindx] = zoneIds[k*YX + j*X + i];
                gindx++;
            }

    m_mapDialog->exec();

    //emit finishedMeshParsing(ctNumberToHumanPhantom(m));
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
    m_mesh->material = MaterialUtils::HOUNSFIELD19;

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

        if(mesh->zoneId[i] >= (signed) atomPerG.size())
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

Mesh *CtDataManager::ctNumberToWater(Mesh *mesh)
{
    m_mesh->material = MaterialUtils::WATER;

    std::vector<float> atomPerG;
    std::vector<std::vector<float> > waterFractions;

    // Check the size of the material vector
    if(MaterialUtils::water.size() != MaterialUtils::waterWeights.size())
    {
        qDebug() << "Water vectors mismatched: water: " << MaterialUtils::water.size() << ",  weight: " << MaterialUtils::waterWeights.size();
        return NULL;
    }

    for(unsigned int i = 0; i < MaterialUtils::waterWeights.size(); i++)
    {
        if(MaterialUtils::waterWeights[i].size() != MaterialUtils::waterElements.size())
        {
            qDebug() << "Water weight vector mismatched: weight[i]: " << MaterialUtils::waterWeights[i].size()
                     << ",  elements: " << MaterialUtils::waterElements.size() << " @ index " << i;
            return NULL;
        }
    }

    // Convert the weight fractions to atom fractions for each material
    for(unsigned int i = 0; i < MaterialUtils::waterElements.size(); i++)
        waterFractions.push_back(MaterialUtils::weightFracToAtomFrac(MaterialUtils::waterElements, MaterialUtils::waterWeights[i]));

    // Convert atom fraction to atom density for each material
    for(unsigned int i = 0; i < MaterialUtils::water.size(); i++)
        atomPerG.push_back(MaterialUtils::atomsPerGram(MaterialUtils::waterElements, waterFractions[i]));

    const int offset = 1024;
    for(unsigned int i = 0; i < mesh->voxelCount(); i++)
    {
        if(mesh->ct[i] > 65500) // Artifact
            mesh->ct[i] = 0;

        // Raw data minus offset
        int ctv = mesh->ct[i] - offset;  // Underlying data is 0-2500 instead of -1000-1500

        // Determine the density (atom density is after the zoneId calculation)
        if(ctv <= -66)
        {
            mesh->density[i] = 0.001225f;
        }
        else if(ctv <= 60)
        {
            mesh->density[i] = 1.0f;
        }
        else
        {
            mesh->density[i] = 1.1f;
        }

        // Determine the material
        mesh->zoneId[i] = static_cast<U16_T>(MaterialUtils::water.size() - 1);  // Last bin is "illegal"
        for(unsigned int j = 0; j < MaterialUtils::water.size()-1; j++)
        {
            if(ctv < MaterialUtils::water[j])
            {
                mesh->zoneId[i] = j;
                break;
            }
        }

        if(mesh->zoneId[i] >= (signed) atomPerG.size())
        {
            qDebug() << "EXPLODE!";
        }

        //float appg = atomPerG[mesh->zoneId[i]];
        mesh->atomDensity[i] = mesh->density[i] * atomPerG[mesh->zoneId[i]] * 1.0E-24f;  // g/cc * @/g * cm^2/b = @/cm-b

        //if(mesh->atomDensity[i] <= 1.0E-10f)
        //{
        //    qDebug() << "Got zero density";
        //}
    }

    return mesh;
}

void CtDataManager::onWater()
{
    emit finishedMeshParsing(ctNumberToWater(m_mesh));
}

void CtDataManager::onPhantom19()
{
    emit finishedMeshParsing(ctNumberToHumanPhantom(m_mesh));
}


