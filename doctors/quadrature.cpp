#include "quadrature.h"

#include <QDebug>

// Static allocators
const Quadrature Quadrature::ms_sn2(2);

Quadrature::Quadrature()
{

}

Quadrature::Quadrature(const Config &config)
{
    load(config);
}

Quadrature::Quadrature(const int sn)
{
    load(sn);
}

void Quadrature::load(const Config &config)
{
    int sn = config.sn;
    load(sn);

    /*
    if(config.sn == 2)
    {
        mu.resize(8);
        eta.resize(8);
        zi.resize(8);
        wt.resize(8);

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
    else if(config.sn == 4)
    {
        qDebug() << "ERROR: Not implemented quadrature!!";
    }
    else if(config.sn == 6)
    {
        qDebug() << "ERROR: Not implemented quadrature!!";
    }
    else
    {
        qDebug() << "ERROR: Unknown quadrature!!";
    }
    */
}

void Quadrature::load(const int sn)
{
    if(sn == 2)
    {
        mu.resize(8);
        eta.resize(8);
        zi.resize(8);
        wt.resize(8);

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
        qDebug() << "ERROR: Not implemented quadrature!!";
    }
    else if(sn == 6)
    {
        qDebug() << "ERROR: Not implemented quadrature!!";
    }
    else
    {
        qDebug() << "ERROR: Unknown quadrature!!";
    }
}

const Quadrature& Quadrature::getSn2()
{
    return ms_sn2;
}
