#include "mcnpwriter.h"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <QDebug>

#include "mesh.h"
#include "xsection.h"
#include "materialutils.h"
#include "sourceparams.h"

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

    surfString += "9     so 10000\n";

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

        for(unsigned int i = 0; i < m->atomDensity.size(); i++)
        {
            if(m->zoneId[i] > matVoxels.size())
            {
                qDebug() << "matVoxels wasn't big enough!";
            }
            matVoxels[m->zoneId[i]]++;
            coarseDensity[m->zoneId[i]] += m->atomDensity[i];
        }

        for(unsigned int i = 0; i < coarseDensity.size(); i++)
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
                unsigned int mindx = xi * m->yElemCt * m->zElemCt + yi * m->zElemCt + zi;

                //if(mindx == 208311)
                //    qDebug() << "stop";

                if(mindx >= m->zoneId.size())
                {
                    std::cerr << "ERROR: mindx=" << mindx << " > zoneId.size=" << m->zoneId.size() << std::endl;
                    m_failFlag = true;
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
                    matstr = std::to_string(m->zoneId[mindx]+1) + " " + std::to_string(atmden);
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
                        //" imp:p=" + std::to_string(importance) +
                        " imp:p=1" +
                        "  $ " + std::to_string(xi) + ", " + std::to_string(yi) + ", " + std::to_string(zi) + " \n";

                if(cellString.length() > 81)  // The newline char doesn't count in the 80 limit
                {
                    m_failFlag = true;
                }

                allCellString += cellString;
            }
        }
    }

    allCellString += std::string("99999998 0 ") +
            "(-" + xsurf(0) + ":" + xsurf(m->xNodeCt-1) +
            ":-" + ysurf(0) + ":" + ysurf(m->yNodeCt-1) +
            ":-" + zsurf(0) + ":" + zsurf(m->zNodeCt-1) +
            ") -9 imp:p=1  $ Void surrounding \n";

    allCellString += std::string("99999999 0  9 imp:p=0  $ Outside\n");

    return allCellString;
}

std::string McnpWriter::generateDataCards(SourceParams *p)
{
    std::string dataString = "c ==================== DATA DECK ====================\n";

    dataString += "nps 1E7\n";
    dataString += "mode p\n";

    float vx, vy, vz, vmag, dx, dy;
    float fx = 25.0,
          fy = 25.0,
          fz = 6.5;
    const float PI = 3.14159265359;
    float ttheta;
    float theta = p->sourceTheta;
    if(p->degrees)
        theta *= (PI/180);

    switch(p->sourceType)
    {
    case 0:  // Point source
        dataString += "sdef par=p pos=" + std::to_string(p->sourceX) + " " + std::to_string(p->sourceY) + " " + std::to_string(p->sourceZ) + " erg=d1\n";
        break;

    case 1: // Fan beam
    case 2: // Multifan
        break;

    case 3: // Cone beam
        vx = fx - p->sourceX;
        vy = fy - p->sourceY;
        vz = fz - p->sourceZ;
        vmag = sqrt(vx*vx + vy*vy + vz*vz);
        vx /= vmag;
        vy /= vmag;
        vz /= vmag;
        dataString += "sdef par=p \n     pos=" + std::to_string(p->sourceX) + " " + std::to_string(p->sourceY) + " " + std::to_string(p->sourceZ) +
                "\n     erg=d1\n     vec="+ std::to_string(vx) + " " + std::to_string(vy) + " " + std::to_string(vz) +
                "\n     dir=d2" +
                "\nsi2 -1 " + std::to_string(cos(theta)) + " 1" +
                "\nsp2 0 " + std::to_string(.5 + cos(theta)/2) + " " + std::to_string(.5 - cos(theta)/2) +
                "\nsb2 0 0 1\n";
        break;
    case 4: // Multicone
        vx = fx - p->sourceX;
        vy = fy - p->sourceY;
        vz = fz - p->sourceZ;
        vmag = sqrt(vx*vx + vy*vy + vz*vz);
        vx /= vmag;
        vy /= vmag;
        vz /= vmag;
        dataString += "sdef par=p \n     pos=" +
                std::to_string(p->sourceX) + " " + std::to_string(p->sourceY) + " " + std::to_string(p->sourceZ) +
                "\n     vec="+ std::to_string(vx) + " " + std::to_string(vy) + " " + std::to_string(vz) +
                "\n     erg=d1 dir=d2 TR=d4" +
                "\nsi2 -1 " + std::to_string(cos(theta)) + " 1" +
                "\nsp2 0 " + std::to_string(.5 + cos(theta)/2) + " " + std::to_string(.5 - cos(theta)/2) +
                "\nsb2 0 0 1\nsi4 L ";
        for(int i = 1; i <= p->sourceN; i++)
            dataString += std::to_string(i) + " ";
        dataString += "\nsp4 D ";
        for(int i = 1; i <= p->sourceN; i++)
            dataString += "1 ";
        dataString += "\nsb4 D ";
        for(int i = 1; i <= p->sourceN; i++)
            dataString += "1 ";
        dataString += "\nTR1 0 0 0\n";

        for(int i = 1; i < p->sourceN; i++)
        {
            ttheta = 2*PI * i/p->sourceN;
            dx = (p->sourceX-fx)*cos(ttheta) - (p->sourceY-fy)*sin(ttheta) + fx - p->sourceX*cos(ttheta) + p->sourceY*sin(ttheta);  //fx - fx*cos(ttheta) - fy*sin(ttheta);
            dy = (p->sourceX-fx)*sin(ttheta) + (p->sourceY-fy)*cos(ttheta) + fy - p->sourceX*sin(ttheta) - p->sourceY*cos(ttheta);
            dataString += "TR" + std::to_string(i+1) + " " + std::to_string(dx) + " " + std::to_string(dy) + " 0 " +
                    std::to_string(cos(ttheta)) + " " + std::to_string(cos(PI/2-ttheta)) + " 0   " +
                    std::to_string(cos(PI/2+ttheta)) + " " + std::to_string(cos(ttheta)) + " 0   0 0 1\n";
        }

        break;
    default:
        dataString += "sdef\n";
    }

    unsigned iEmax = 0;
    while(p->spectraIntensity[iEmax] <= 0 && iEmax < p->spectraIntensity.size())
        iEmax++;

    std::string si = "SI1 H   &\n     ";
    for(unsigned int i = p->spectraEnergyLimits.size()-1; i >= iEmax+1; i--)
          si += std::to_string(p->spectraEnergyLimits[i]/1e6) + " &\n     ";
    si += std::to_string(p->spectraEnergyLimits[iEmax]/1e6) + " \n";

    std::string sp = "SP1 D &\n     0     \n     ";
    for(unsigned int i = p->spectraIntensity.size()-1; i >= iEmax+1; i--)
        sp += std::to_string(p->spectraIntensity[i]) + " &\n     ";
    sp += std::to_string(p->spectraIntensity[iEmax]) + " \n";

    dataString += si + sp;  //limit80Char(si) + limit80Char(sp);

    return dataString;
}

std::string McnpWriter::generateMaterialString(Mesh *m)
{
    std::string allMatString = "c ---------- Materials ----------\n";

    if(m->material == MaterialUtils::HOUNSFIELD19)
    {
        allMatString += generatePhantom19MaterialString();
    }
    else if(m->material == MaterialUtils::WATER)
    {
        allMatString += generateWaterMaterialString();
    }
    else
    {
        qDebug() << "Mesh material id unknown: " << m->material;
    }

    return allMatString;
}

std::string McnpWriter::generatePhantom19MaterialString()
{
    std::string allMatString = "";

    for(unsigned int mid = 0; mid < MaterialUtils::hounsfieldRangePhantom19.size(); mid++)
    {
        std::vector<float> atomFractions = MaterialUtils::weightFracToAtomFrac(MaterialUtils::hounsfieldRangePhantom19Elements, MaterialUtils::hounsfieldRangePhantom19Weights[mid]);
        std::string matString = "m" + std::to_string(mid+1) + " \n";

        for(unsigned int zindx = 0; zindx < MaterialUtils::hounsfieldRangePhantom19Elements.size(); zindx++)
        {
            if(MaterialUtils::hounsfieldRangePhantom19Weights[mid][zindx] > 0)
                matString += "     " + std::to_string(MaterialUtils::hounsfieldRangePhantom19Elements[zindx]) + "000 " + std::to_string(atomFractions[zindx]) + "\n";
        }

        allMatString += matString;
    }

    return allMatString;
}

std::string McnpWriter::generateWaterMaterialString()
{
    std::string allMatString = "";

    for(unsigned int mid = 0; mid < MaterialUtils::water.size(); mid++)
    {
        std::vector<float> atomFractions = MaterialUtils::weightFracToAtomFrac(MaterialUtils::waterElements, MaterialUtils::waterWeights[mid]);
        std::string matString = "m" + std::to_string(mid+1) + " \n";

        for(unsigned int zindx = 0; zindx < MaterialUtils::waterElements.size(); zindx++)
        {
            if(MaterialUtils::waterWeights[mid][zindx] > 0)
                matString += "     " + std::to_string(MaterialUtils::waterElements[zindx]) + "000 " + std::to_string(atomFractions[zindx]) + "\n";
        }

        allMatString += matString;
    }

    return allMatString;
}

std::string McnpWriter::generateMeshTally(Mesh *m, SourceParams *p)
{
    unsigned int highestEnergy = 0;
    bool keepGoing = true;

    // Find the index of the highest non-zero group
    while(keepGoing)
    {
        if(p->spectraIntensity[highestEnergy] <= 0)
            highestEnergy++;
        else
            keepGoing = false;
    }

    std::string emeshunc = "EMESH ";
    std::string eintsunc = "EINTS ";
    std::string emeshcol = "EMESH ";
    std::string eintscol = "EINTS ";

    // Start at size() instead of the usual size()-1 because the limits vector is one element longer
    //   than the intensity vector.
    for(unsigned int ie = p->spectraIntensity.size(); ie >= highestEnergy; ie--)
    {
        emeshunc += std::to_string(p->spectraEnergyLimits[ie]/1e6) + " ";
        eintsunc += "1 ";
        emeshcol += std::to_string(p->spectraEnergyLimits[ie]/1e6) + " ";
        eintscol += "1 ";
    }

    std::string tallyString = std::string("FC4 Uncollided flux\n") +
            "FMESH4:p GEOM=REC ORIGIN=0 0 0 INC 0 \n     " +
            "IMESH " + std::to_string(m->xNodes[m->xNodeCt-1]) + " IINTS " + std::to_string(m->xElemCt) + "\n     " +
            "JMESH " + std::to_string(m->yNodes[m->yNodeCt-1]) + " JINTS " + std::to_string(m->yElemCt) + "\n     " +
            "KMESH " + std::to_string(m->zNodes[m->zNodeCt-1]) + " KINTS " + std::to_string(m->zElemCt) + "\n     " +
            emeshunc + "\n     " + eintsunc + "\n" +
            "FC14 Collided flux\n" +
            "FMESH14:p GEOM=REC ORIGIN=0 0 0 INC 1 INFINITE \n     " +
            "IMESH " + std::to_string(m->xNodes[m->xNodeCt-1]) + " IINTS " + std::to_string(m->xElemCt) + "\n     " +
            "JMESH " + std::to_string(m->yNodes[m->yNodeCt-1]) + " JINTS " + std::to_string(m->yElemCt) + "\n     " +
            "KMESH " + std::to_string(m->zNodes[m->zNodeCt-1]) + " KINTS " + std::to_string(m->zElemCt) + "\n     " +
            emeshcol + "\n     " + eintscol + "\n" +
            "UNC:p 0 " + std::to_string(m->voxelCount()+1) + "r\n";

    return tallyString;
}

/*
std::string McnpWriter::limit80Char(std::string s)
{
    std::string r;
    std::string next;

    //unsigned int wordstart = 0;
    unsigned int ptr = 0;
    unsigned int linelen = 0;

    while(ptr < s.length())
    {
        // while not whitespace, keep adding to the word
        // if this word would make the line too long:
        //     newline+space, reset and continue
        // else
        //     add the word to the line
        next = "";
        while(!std::isspace(s[ptr]) && ptr < s.length())
        {
            ptr++;
            next += s[ptr];
        }

        if(next.size() + linelen + 1 > 80)
        {
            r += '\n';
            linelen = next.size() + 1;
            r+= next + " ";
        }
        else
        {
            r += next + " ";
            linelen += next.size() + 1;
        }

        ptr++;
    }

    return r;
}
*/

void McnpWriter::writeMcnp(std::string filename, Mesh *m, SourceParams *p, bool fineDensity)
{
    std::ofstream fout;
    fout.open(filename.c_str());

    fout << "Automatically generated from CT Data\n";
    fout << "c MCNP input deck automatically generated by DOCTORS framework\n";
    fout << "c Authors: Edward Norris and Xin Liu, Missouri Univ. of Sci. and Tech.\n";
    fout << "c \n";
    fout << "c Mesh dimensions: " << m->xElemCt << " " << m->yElemCt << " " << m->zElemCt << "\n";
    fout << generateCellString(m, fineDensity);
    fout << "\n";
    fout << generateSurfaceString(m);
    fout << "\n";
    fout << generateDataCards(p);
    fout << generateMaterialString(m);
    fout << generateMeshTally(m, p);

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



