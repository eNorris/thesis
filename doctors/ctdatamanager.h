#ifndef CTDATAREADER_H
#define CTDATAREADER_H

#include <QMessageBox>
#include <QObject>

#include "mesh.h"
#include "quadrature.h"

class CtMapDialog;

class CtDataManager : public QObject
{
    Q_OBJECT

protected:
    bool m_valid;
    Mesh *m_mesh;
    QMessageBox m_messageBox;
    CtMapDialog *m_mapDialog;

public:
    CtDataManager();
    ~CtDataManager();

    //Mesh *parse16(int xbins, int ybins, int zbins, std::string filename);
    Mesh *ctNumberToHumanPhantom(Mesh *mesh);
    Mesh *ctNumberToWater(Mesh *mesh);
    Mesh *ctNumberToQuickCheck(Mesh *mesh);

signals:
    //void signalBeginMeshParse(int,int,int,std::string);
    //void updateParseProgress(int v);
    void signalMeshUpdate(int);
    void finishedMeshParsing(Mesh*);
    //void signalMeshUpdate(int);

public slots:
    void parse16(int xbins, int ybins, int zbins, QString filename);

protected slots:
    void onWater();
    void onPhantom19();
};

#endif // CTDATAREADER_H
