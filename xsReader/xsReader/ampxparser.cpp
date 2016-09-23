#include "ampxparser.h"

#include <QDebug>
#include <ctime>

AmpxParser::AmpxParser(QObject *parent) : QObject(parent)
{

}

AmpxParser::~AmpxParser()
{
    for(unsigned int i = 0; i < directories.size(); i++)
        if(directories[i] != NULL)
            delete directories[i];

    for(unsigned int i = 0; i < data.size(); i++)
        if(data[i] != NULL)
            delete data[i];
}

bool AmpxParser::openFile(QString filename)
{
    binfile.open(filename.toStdString().c_str(), std::ios::binary | std::ios::in);

    if(!binfile.good())
    {
        emit error("File was not opened!");
        return false;
        //qDebug() << "Failed to open file!";
    }
    return true;
}

void AmpxParser::closeFile()
{
     binfile.close();
}

bool AmpxParser::parseHeader()
{
    if(!binfile.good())
    {
        emit error("AMPX Parse Header: File was not good!");
        return false;
    }

    header.parse(binfile);

    for(int i = 0; i < header.getNumberNuclides(); i++)
    {
        AmpxRecordParserType3 *dir = new AmpxRecordParserType3;
        dir->parse(binfile);
        directories.push_back(dir);
    }

    nBounds.parse(binfile, header.getGroupCountNeutron());
    gBounds.parse(binfile, header.getGroupCountGamma());

    return true;
}

bool AmpxParser::parseData()
{
    if(!binfile.good())
    {
        emit error("AMPX Parse Data: File was not good!");
        return false;
    }

    std::clock_t start;
    start = std::clock();

    for(int i = 0; i < header.getNumberNuclides(); i++)
    {
        // output the current nuclide
        qDebug() << "Reading nuclide " << (i+1) << "/" << header.getNumberNuclides();
        NuclideData *nextData = new NuclideData;
        nextData->parse(binfile, header.getGroupCountNeutron(), header.getGroupCountGamma());
        data.push_back(nextData);
    }

    qDebug() << "Time: " << (std::clock() - start)/(double)(CLOCKS_PER_SEC/1000) << " ms";

    qDebug() << "finished parsing";

    return true;
}

std::vector<int> AmpxParser::getZaids()
{
    std::vector<int> zaids;

    if(getNumberNuclides() == 0)
        return zaids;

    zaids.resize(getNumberNuclides());

    for(int i = 0; i < getNumberNuclides(); i++)
    {
        zaids[i] = directories[i]->getId();
    }

    return zaids;
}

int AmpxParser::getNumberNuclides() const
{
    return header.getNumberNuclides();
}

int AmpxParser::findIndexByZaid(int zaid)
{
    int indx = -1;
    for(unsigned int i = 0; i < directories.size(); i++)
    {
        if(directories[i]->getId() == zaid)
        {
            if(indx == -1)
            {
                indx = i;
            }
            else
            {
                emit error("Multiple indices!");
            }
        }
    }
    if(indx == -1)
        emit error("No index!");

    return indx;
}

NuclideData *AmpxParser::getData(unsigned int indx)
{
    if(indx < data.size())
        return data[indx];
    else
    {
        emit error("AmpxParser::getData(): Could not get nuclide number " + QString::number(indx));
        return NULL;
    }
}

void AmpxParser::debugNeutronEnergies()
{
    QString log("");
    for(unsigned int i = 0; i < getNeutronEnergy().size(); i++)
        log += QString::number(getNeutronEnergy()[i]) + "\t";
    qDebug() << ("Neutron Energy (" + QString::number(getNeutronEnergy().size()-1) + "): " + log);
}

void AmpxParser::debugGammaEnergies()
{
    QString log("");
    for(unsigned int i = 0; i < getGammaEnergy().size(); i++)
        log += QString::number(getGammaEnergy()[i]) + "\t";
    qDebug() << ("Gamma Energy (" + QString::number(getGammaEnergy().size()-1) + "): " + log);
}

void AmpxParser::debugAllEnergies()
{
    debugNeutronEnergies();
    debugGammaEnergies();
}
