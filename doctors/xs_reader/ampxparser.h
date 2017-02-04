#ifndef AMPXPARSER_H
#define AMPXPARSER_H

#include <QObject>

#include "ampxrecordparsers.h"
#include "nuclidedata.h"

#include <vector>

class AmpxParser : public QObject
{
    Q_OBJECT

protected:
    ifstream binfile;
    QString m_filename;

    AmpxRecordParserType1 header;
    std::vector<AmpxRecordParserType3*> directories;
    AmpxRecordParserType2 nBounds;
    AmpxRecordParserType2 gBounds;
    std::vector<NuclideData*> data;

public:
    explicit AmpxParser(QObject *parent = NULL);
    virtual ~AmpxParser();

    void closeFile();

    QString getFilename() const;

    std::vector<int> getZaids() const;
    int getNumberNuclides() const;

    int getNeutronEnergyGroups() const { return header.getGroupCountNeutron(); }
    const std::vector<float> &getNeutronEnergy() const { return nBounds.getEnergy(); }
    const std::vector<float> &getNeutronLethargy() const { return nBounds.getLethargy(); }

    int getGammaEnergyGroups() const { return header.getGroupCountGamma(); }
    const std::vector<float> &getGammaEnergy() const { return gBounds.getEnergy(); }
    const std::vector<float> &getGammaLethargy() const { return gBounds.getLethargy(); }

    const NuclideData *getNuclideEntry(const int indx) const { return data[indx]; }
    const AmpxRecordParserType3 *getDirectoryEntry(const int indx) const { return directories[indx]; }

    void debugNeutronEnergies();
    void debugGammaEnergies();
    void debugAllEnergies();

    int getIndexByZaid(int zaid) const;

    NuclideData *getData(unsigned int indx) const;

public slots:
    bool parseFile(QString filename);
    bool openFile(QString filename);
    bool parseHeader();
    bool parseData();

signals:
    void error(QString msg) const;
    void finishedParsing(AmpxParser*) const;
    void signalXsUpdate(int);
    void signalNotifyNumberNuclides(int);

};

#endif // AMPXPARSER_H
