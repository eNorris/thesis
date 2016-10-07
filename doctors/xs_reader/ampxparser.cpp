#include "ampxparser.h"

#include <QDebug>
#include <ctime>

AmpxParser::AmpxParser(QObject *parent) : QObject(parent)
{

}

AmpxParser::~AmpxParser()
{
    closeFile();

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
    }
    return true;
}

bool AmpxParser::parseFile(QString filename)
{
    //qDebug() << "Parsing begins...";
    if(openFile(filename))
    {
        if(parseHeader())
        {
            emit signalNotifyNumberNuclides(getNumberNuclides());
            if(parseData())
                return true;
        }
    }
    return false;
}

void AmpxParser::closeFile()
{
    if(binfile.is_open())
        binfile.close();
}

bool AmpxParser::parseHeader()
{
    if(!binfile.good())
    {
        emit error("AMPX Parse Header: File was not good!");
        return false;
    }

    if(!header.parse(binfile))
    {
        emit error("AMPX Parse Header: Could not read the main header");
        return false;
    }

    for(int i = 0; i < header.getNumberNuclides(); i++)
    {
        AmpxRecordParserType3 *dir = new AmpxRecordParserType3;
        if(!dir->parse(binfile))
        {
            emit error("AMPX Parse Header: Failed to read directory for nuclide #" + QString::number(i));
            return false;
        }
        directories.push_back(dir);
    }

    if(!nBounds.parse(binfile, header.getGroupCountNeutron()))
    {
        emit error("AMPX Parse Header: Failed to read neutron energy bounds");
        return false;
    }
    if(!gBounds.parse(binfile, header.getGroupCountGamma()))
    {
        emit error("AMPX Parse Header: Failed to read gamma energy bounds");
        return false;
    }

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

        //qDebug() << "Reading nuclide " << (i+1) << "/" << header.getNumberNuclides();
        NuclideData *nextData = new NuclideData;
        nextData->parse(binfile, header.getGroupCountNeutron(), header.getGroupCountGamma());
        data.push_back(nextData);
        emit signalXsUpdate(i);
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

int AmpxParser::getIndexByZaid(int zaid) const
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
                emit error("AmpxParser::getIndexByZaid(): Multiple indices!");
            }
        }
    }
    if(indx == -1)
        emit error("AmpxParser::getIndexByZaid(): No index!");

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
