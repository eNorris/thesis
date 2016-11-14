#include "mcnpwriter.h"

#include <iostream>
#include <fstream>

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

    for(int i = 0; i < m->xNodeCt; i++)
    {
        surfString += ( xsurf(i) + " px " + std::to_string(m->xNodes[i]) + '\n');
    }

    for(int i = 0; i < m->yNodeCt; i++)
    {
        surfString += (ysurf(i) + " py " + std::to_string(m->yNodes[i]) + '\n');
    }

    for(int i = 0; i < m->zNodeCt; i++)
    {
        surfString += (zsurf(i) + " pz " + std::to_string(m->zNodes[i]) + '\n');
    }

    return surfString;
}

std::string McnpWriter::generateCellString(Mesh *m)
{
    std::string allCellString = "c ============================== CELL DECK ==============================\n";

    for(int xi = 0; xi < m->xElemCt; xi++)
    {
        for(int yi = 0; yi < m->yElemCt; yi++)
        {
            for(int zi = 0; zi < m->zElemCt; zi++)
            {
                int mindx = xi * m->yElemCt * m->zElemCt + yi * m->zElemCt + zi;
                //std::string cellString = "";

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

                // Increment zoneId by 1 because 0 is not legal for MCNP
                std::string cellString = padFiveDigitsSpace(mindx+1) + " " + std::to_string(m->zoneId[mindx]+1) + " " + std::to_string(m->atomDensity[mindx]) + " " +
                        xsurf(xi) + " -" + xsurf(xi+1) + " " +
                        ysurf(yi) + " -" + ysurf(yi+1) + " " +
                        zsurf(zi) + " -" + zsurf(zi+1) + " " +
                        " imp:p=1  $ " + std::to_string(xi) + ", " + std::to_string(yi) + ", " + std::to_string(zi) + " \n";

                if(cellString.length() > 81)  // The newline char doesn't count in the 80 limit
                {
                    //std::cerr << "ERROR: Cell <" << xi << ", " << yi << ", " << zi << "> = " << mindx << " exceeded 80 characters" << std::endl;
                    m_failFlag = true;
                }

                allCellString += cellString;
            }
        }
    }

    return allCellString;
}

std::string McnpWriter::generateDataCards()
{
    std::string dataString = "c ==================== DATA DECK ====================\n";

    dataString += "nps 1E7\n";
    dataString += "mode p\n";
    dataString += "sdef par=2 pos=0 0 0 erg=0.30\n";


    return dataString;
}

std::string McnpWriter::generatePhantom19MaterialString()
{
    std::string allMatString = "c ---------- Materials ----------\n";



    for(int mid = 0; mid < MaterialUtils::hounsfieldRangePhantom19.size(); mid++)
    {
        std::vector<float> atomFractions = MaterialUtils::weightFracToAtomFrac(MaterialUtils::hounsfieldRangePhantom19Elements, MaterialUtils::hounsfieldRangePhantom19Weights[mid]);
        std::string matString = std::to_string(mid+1) + " \n";

        for(int zindx = 0; zindx < MaterialUtils::hounsfieldRangePhantom19Elements.size(); zindx++)
        {
            if(MaterialUtils::hounsfieldRangePhantom19Weights[mid][zindx] > 0)
                matString += "     " + std::to_string(MaterialUtils::hounsfieldRangePhantom19Elements[zindx]) + "000 " + std::to_string(atomFractions[zindx]) + "\n";
        }

        allMatString += matString;
    }

    return allMatString;
}

void McnpWriter::writeMcnp(std::string filename, Mesh *m)
{
    std::ofstream fout;
    fout.open(filename.c_str());

    fout << "Automatically generated from CT Data\n";
    fout << "c MCNP input deck automatically generated by DOCTORS framework\n";
    fout << "c Authors: Edward Norris and Xin Liu, Missouri Univ. of Sci. and Tech.\n";
    fout << "c \n";
    fout << generateCellString(m);
    fout << "\n";
    fout << generateSurfaceString(m);
    fout << "\n";
    fout << generateDataCards();
    fout << generatePhantom19MaterialString();

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

std::string McnpWriter::padFiveDigitsSpace(int v)
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

    int maxbounds = 100000;

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



