#include "quadrature.h"

#include <QDebug>
#include <string>

// Static allocators
const Quadrature Quadrature::ms_sn2(2);

Quadrature::Quadrature() : m_angles(0)
{

}

Quadrature::Quadrature(const Config *config) : m_angles(0)
{
    load(config);
}

Quadrature::Quadrature(const int sn)
{
    loadSn(sn);
}

void Quadrature::load(const Config *config)
{
    //int sn = config->sn;
    if(config->quadType == "sn")
        loadSn(config->sn);
    else
        qDebug() << "Unknown quadrature type: " << QString::fromStdString(config->quadType);
}

void Quadrature::loadSn(const int sn)
{
    m_angles = sn * (sn + 2);

    mu.resize(m_angles);
    eta.resize(m_angles);
    zi.resize(m_angles);
    wt.resize(m_angles);

    if(sn == 2)
    {

        //mu.resize(m_angles);
        //eta.resize(m_angles);
        //zi.resize(m_angles);
        //wt.resize(m_angles);

        mu[0] = -.577350;
        mu[1] =  .577350;
        mu[2] = -.577350;
        mu[3] =  .577350;
        for(int i = 4; i < 8; i++)
            mu[i] = mu[i-4];

        for(int i = 0; i < 4; i++)
            eta[i] = -.577350;
        for(int i = 4; i < 8; i++)
            eta[i] = .577350;

        for(int i = 0; i < 2; i++)
            zi[i] = -sqrt(1 - mu[i]*mu[i] - eta[i]*eta[i]);
        for(int i = 2; i < 4; i++)
            zi[i] = -zi[i-2];
        for(int i = 4; i < 8; i++)
            zi[i] = zi[i-4];

        for(int i = 0; i < 8; i++)
            wt[i] = 1.0 / 8.0;
    }
    else if(sn == 4)
    {
        mu[0] = -0.868890;
        mu[1] =  -0.350021;
        mu[2] = -0.350021;
        mu[3] =  0.350021;
        mu[4] = .350021;
        mu[5] = 0.868890;
        for(int i = 6; i < 12; i++)
            mu[i] = mu[i-6];
        for(int i = 12; i < 24; i++)
            mu[i] = mu[i-12];

        eta[0] = -0.350021;
        eta[1] = -0.350021;
        eta[2] = -0.868890;
        eta[3] = -0.868890;
        eta[4] = -0.350021;
        eta[5] = -0.350021;
        for(int i = 6; i < 12; i++)
            eta[i] = eta[i-6];
        for(int i = 12; i < 24; i++)
            eta[i] = -eta[i-12];

        for(int i = 0; i < 6; i++)
            zi[i] = -sqrt(1 - mu[i]*mu[i] - eta[i]*eta[i]);
        for(int i = 6; i < 12; i++)
            zi[i] = -zi[i-6];
        for(int i = 12; i < 24; i++)
            zi[i] = zi[i-12];

        for(int i = 0; i < 8; i++)
            wt[i] = 1.0 / 8.0;
    }
    else if(sn == 6)
    {
        qDebug() << "ERROR: Not implemented quadrature!!";
    }
    else
    {
        qDebug() << "ERROR: Unknown quadrature!! Sn=" << sn;
    }
}

const Quadrature& Quadrature::getSn2()
{
    return ms_sn2;
}

int Quadrature::angleCount() const
{
    return m_angles;
}
