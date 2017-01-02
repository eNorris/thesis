#include "mcnpwriter.h"

#include <iostream>
#include <fstream>
#include <QDebug>

#include "mesh.h"
#include "materialutils.h"

McnpWriter::McnpWriter() : m_failFlag(false)
{

}

McnpWriter::~McnpWriter()
{

}

std::string McnpWriter::generateSurfaceString(Mesh *m)
{
    if(m->xNodeCt > MAX_BOUNDS)
    {
        std::cerr << "Too many x boundaries: " << m->xNodeCt << " requested, MAX_BOUNDS=" << MAX_BOUNDS << std::endl;
        m_failFlag = true;
        return "";
    }
    if(m->yNodeCt > MAX_BOUNDS)
    {
        std::cerr << "Too many y boundaries: " << m->yNodeCt << " requested, MAX_BOUNDS=" << MAX_BOUNDS << std::endl;
        m_failFlag = true;
        return "";
    }
    if(m->zNodeCt > MAX_BOUNDS)
    {
        std::cerr << "Too many z boundaries: " << m->zNodeCt << " requested, MAX_BOUNDS=" << MAX_BOUNDS << std::endl;
        m_failFlag = true;
        return "";
    }

    std::string surfString = "c ============================== SURFACE DECK ==============================\n";

    if(m->xNodeCt != m->xNodes.size())
    {
        std::cerr << "SIZE MISMATCH @ McnpWriter::generateSurfaceString(Mesh*): 43: node count mismatch" << std::endl;
    }

    for(unsigned int i = 0; i < m->xNodeCt; i++)
    {
        surfString += ( xsurf(i) + " px " + std::to_string(m->xNodes[i]) + '\n');
    }

    for(unsigned int i = 0; i < m->yNodeCt; i++)
    {
        surfString += (ysurf(i) + " py " + std::to_string(m->yNodes[i]) + '\n');
    }

    for(unsigned int i = 0; i < m->zNodeCt; i++)
    {
        surfString += (zsurf(i) + " pz " + std::to_string(m->zNodes[i]) + '\n');
    }

    return surfString;
}

std::string McnpWriter::generateCellString(Mesh *m, bool fineDensity)
{
    std::string allCellString = "c ============================== CELL DECK ==============================\n";

    std::vector<float> coarseDensity;
    if(!fineDensity)
    {
        // Calculate the average density for each material
        std::vector<int> matVoxels;
        coarseDensity.resize(MaterialUtils::hounsfieldRangePhantom19.size(), 0.0f);
        matVoxels.resize(MaterialUtils::hounsfieldRangePhantom19.size(), 0);

        for(int i = 0; i < m->atomDensity.size(); i++)
        {
            if(m->zoneId[i] > matVoxels.size())
            {
                qDebug() << "matVoxels wasn't big enough!";
            }
            matVoxels[m->zoneId[i]]++;
            coarseDensity[m->zoneId[i]] += m->atomDensity[i];
        }

        for(int i = 0; i < coarseDensity.size(); i++)
        {
            coarseDensity[i] /= matVoxels[i];
        }
    }

    for(unsigned int xi = 0; xi < m->xElemCt; xi++)
    {
        for(unsigned int yi = 0; yi < m->yElemCt; yi++)
        {
            for(unsigned int zi = 0; zi < m->zElemCt; zi++)
            {
                int mindx = xi * m->yElemCt * m->zElemCt + yi * m->zElemCt + zi;
                //std::string cellString = "";

                if(mindx == 208311)
                    qDebug() << "stop";

                if(mindx >= m->zoneId.size())
                {
                    std::cerr << "ERROR: mindx=" << mindx << " > zoneId.size=" << m->zoneId.size() << std::endl;
                    m_failFlag = true;
                    //cellString = "?????";
                    allCellString += "?????\n";
                    continue;
                }

                if(mindx >= m->atomDensity.size())
                {
                    std::cerr << "ERROR: mindx=" << mindx << " > atomDensity.size=" << m->atomDensity.size() << std::endl;
                    m_failFlag = true;
                    //cellString = "?????";
                    allCellString += "?????\n";
                    continue;
                }

                //if(mindx == 208311)
                //    qDebug() << "halt";

                std::string matstr = "";

                // Illegal cells are just void
                if(m->zoneId[mindx] == MaterialUtils::hounsfieldRangePhantom19.size()-1)
                {
                    matstr = "0";
                }
                else
                {
                    float atmden;
                    if(fineDensity)
                        atmden = m->atomDensity[mindx];
                    else
                        atmden = coarseDensity[m->zoneId[mindx]];
                    matstr = std::to_string(m->zoneId[mindx]+1) + " " + std::to_string(m->atomDensity[mindx]);
                }

                int importance = 1;
                float dx = m->xNodes[xi] + m->xNodes[xi+1];
                float dy = m->yNodes[yi] + m->yNodes[yi+1];
                float dz = m->zNodes[zi] + m->zNodes[zi+1];
                float dist = sqrt(dx*dx + dy*dy + dz*dz);
                float r2 = dist*dist;
                if(r2 < 1.0)
                    r2 = 1.0;
                importance = r2;

                // Increment zoneId by 1 because 0 is not legal for MCNP
                std::string cellString = padFourDigitsSpace(mindx+1) + " " + matstr + " " +
                        xsurf(xi) + " -" + xsurf(xi+1) + " " +
                        ysurf(yi) + " -" + ysurf(yi+1) + " " +
                        zsurf(zi) + " -" + zsurf(zi+1) + " " +
                        " imp:p=" + std::to_string(importance) + "  $ " + std::to_string(xi) + ", " + std::to_string(yi) + ", " + std::to_string(zi) + " \n";

                if(cellString.length() > 81)  // The newline char doesn't count in the 80 limit
                {
                    //std::cerr << "ERROR: Cell <" << xi << ", " << yi << ", " << zi << "> = " << mindx << " exceeded 80 characters" << std::endl;
                    m_failFlag = true;
                }

                allCellString += cellString;
            }
        }
    }

    allCellString += std::string("99999999 0 ") +
            " -" + xsurf(0) + ":" + xsurf(m->xNodeCt-1) +
            ":-" + ysurf(0) + ":" + ysurf(m->yNodeCt-1) +
            ":-" + zsurf(0) + ":" + zsurf(m->zNodeCt-1) +
            " imp:p=0  $ Outside\n";

    return allCellString;
}

std::string McnpWriter::generateDataCards(Mesh *m)
{
    std::string dataString = "c ==================== DATA DECK ====================\n";

    dataString += "nps 1E7\n";
    dataString += "mode p\n";

    // The offset prevents the source from landing on a surface plane which can cause particles to get lost
    float offset = 0.001f;

    //float maxerg = 0.01;

    dataString += "sdef par=p pos=" + std::to_string(m->xNodes[m->xNodeCt-1]/2 + offset) + " " + std::to_string(m->yNodes[m->yNodeCt-1]/2 + offset) + " " + std::to_string(m->zNodes[m->zNodeCt-1]/2 + offset) + " erg=0.030\n";


    return dataString;
}

std::string McnpWriter::generatePhantom19MaterialString()
{
    std::string allMatString = "c ---------- Materials ----------\n";

    for(int mid = 0; mid < MaterialUtils::hounsfieldRangePhantom19.size(); mid++)
    {
        std::vector<float> atomFractions = MaterialUtils::weightFracToAtomFrac(MaterialUtils::hounsfieldRangePhantom19Elements, MaterialUtils::hounsfieldRangePhantom19Weights[mid]);
        std::string matString = "m" + std::to_string(mid+1) + " \n";

        for(int zindx = 0; zindx < MaterialUtils::hounsfieldRangePhantom19Elements.size(); zindx++)
        {
            if(MaterialUtils::hounsfieldRangePhantom19Weights[mid][zindx] > 0)
                matString += "     " + std::to_string(MaterialUtils::hounsfieldRangePhantom19Elements[zindx]) + "000 " + std::to_string(atomFractions[zindx]) + "\n";
        }

        allMatString += matString;
    }

    return allMatString;
}

std::string McnpWriter::generateMeshTally(Mesh *m)
{
    std::string tallyString = std::string("FMESH4:p GEOM=REC ORIGIN=0 0 0\n     ") +
            "IMESH " + std::to_string(m->xNodes[m->xNodeCt-1]) + " IINTS " + std::to_string(m->xElemCt) + "\n     " +
            "JMESH " + std::to_string(m->yNodes[m->yNodeCt-1]) + " JINTS " + std::to_string(m->yElemCt) + "\n     " +
            "KMESH " + std::to_string(m->zNodes[m->zNodeCt-1]) + " KINTS " + std::to_string(m->zElemCt) + "\n     " +
            "EMESH .01 .045 .1 .2 EINTS 1 1 1 1\n";

    return tallyString;
}

void McnpWriter::writeMcnp(std::string filename, Mesh *m, bool fineDensity)
{
    std::ofstream fout;
    fout.open(filename.c_str());

    fout << "Automatically generated from CT Data\n";
    fout << "c MCNP input deck automatically generated by DOCTORS framework\n";
    fout << "c Authors: Edward Norris and Xin Liu, Missouri Univ. of Sci. and Tech.\n";
    fout << "c \n";
    fout << generateCellString(m, fineDensity);
    fout << "\n";
    fout << generateSurfaceString(m);
    fout << "\n";
    fout << generateDataCards(m);
    fout << generatePhantom19MaterialString();
    fout << generateMeshTally(m);

    fout.close();
}

std::string McnpWriter::padFourDigitsZero(int v)
{
    if(v > MAX_BOUNDS)
    {
        std::cerr << "Could not pad v=" << v << " because it was too large, MAX_BOUNDS=" << MAX_BOUNDS << std::endl;
        m_failFlag = true;
        return "";
    }

    std::string padded = "";

    int maxbounds = MAX_BOUNDS/100;

    while(maxbounds > 0)
    {
        if(maxbounds > v)
        {
            padded += "0";
        }
        maxbounds /= 10;
    }

    padded += std::to_string(v);

    return padded;
}

std::string McnpWriter::padFourDigitsSpace(int v)
{
    if(v == 1E6)
    {
        std::cerr << "WARNING: Exceeded 99,999 cells, MCNP6 is required" << std::endl;
    }

    if(v == 1E8)
    {
        std::cerr << "ERROR: MCNP6 can only handle cells up to 99,999,999" << std::endl;
        m_failFlag = true;
    }

    std::string padded = "";

    int maxbounds = 10000;

    while(maxbounds > 0)
    {
        if(maxbounds > v)
        {
            padded += " ";
        }
        maxbounds /= 10;
    }

    padded += std::to_string(v);

    return padded;
}

std::string McnpWriter::xsurf(int xindx)
{
    return std::string("1") + padFourDigitsZero(xindx+1);
}

std::string McnpWriter::ysurf(int yindx)
{
    return std::string("2") + padFourDigitsZero(yindx+1);
}

std::string McnpWriter::zsurf(int zindx)
{
    return std::string("3") + padFourDigitsZero(zindx+1);
}



