#include "quadrature.h"

#include <QDebug>
#include <string>
#include <qmath.h>

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
    else if(config->quadType == "special")
        loadSpecial(config->quadSpecial);
    else
        qDebug() << "Unknown quadrature type: " << QString::fromStdString(config->quadType);
}

void Quadrature::loadSpecial(const int special)
{
    if(special == 1)
    {
        int N = 100;
        m_angles = N;
        for(int i = 0; i < N; i++)
        {
            float theta = 2 * M_PI * static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            float u = 2  * static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 1;
            mu.push_back(sqrt(1-u*u)*cos(theta));
            zi.push_back(sqrt(1-u*u)*sin(theta));
            eta.push_back(u);
            wt.push_back(1.0/N);
        }
    }
    else if(special == 2)
    {
        int N = 100;
        m_angles = N;

        float dlong = M_PI * (3-sqrt(5));
        float dz = 2.0/N;
        float lng = 0;
        float z = 1 - dz/2;

        for(int i = 0; i < N; i++)
        {
            float r = sqrt(1 - z*z);
            mu.push_back(cos(lng)*r);
            zi.push_back(sin(lng)*r);
            eta.push_back(z);
            wt.push_back(1.0/N);
            z -= dz;
            lng += dlong;
        }
    }
    else
        qDebug() << "Unknown sqecial quadrature type: " << QString::number(special);
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

        for(int i = 0; i < 24; i++)
            wt[i] = 1.0 / 24.0;
    }
    else if(sn == 6)
    {
        mu[0] = -0.926181;
        mu[1] = -0.681508;
        mu[2] = -0.681508;
        mu[3] = -0.266636;
        mu[4] = -0.266636;
        mu[5] = -0.266636;
        mu[6] =  0.266636;
        mu[7] =  0.266636;
        mu[8] =  0.266636;
        mu[9] = 0.681508;
        mu[10] = 0.681508;
        mu[11] = 0.926181;
        for(int i = 12; i < 24; i++)
            mu[i] = mu[i-12];
        for(int i = 24; i < 48; i++)
            mu[i] = mu[i-24];

        eta[0] = -0.266636;
        eta[1] = -0.266636;
        eta[2] = -0.681508;
        eta[3] = -0.266636;
        eta[4] = -0.681508;
        eta[5] = -0.926181;
        eta[6] = -0.926181;
        eta[7] = -0.681508;
        eta[8] = -0.266636;
        eta[9] = -0.681508;
        eta[10] = -0.266636;
        eta[11] = -0.266636;
        for(int i = 12; i < 24; i++)
            eta[i] = eta[i-12];
        for(int i = 24; i < 48; i++)
            eta[i] = -eta[i-24];

        for(int i = 0; i < 12; i++)
            zi[i] = -sqrt(1 - mu[i]*mu[i] - eta[i]*eta[i]);
        for(int i = 12; i < 24; i++)
            zi[i] = -zi[i-12];
        for(int i = 24; i < 48; i++)
            zi[i] = zi[i-24];

        wt[0] = 0.176126/8;
        wt[1] = 0.157207/8;
        wt[2] = 0.157207/8;
        wt[3] = 0.176126/8;
        wt[4] = 0.157207/8;
        wt[5] = 0.176126/8;
        wt[6] = 0.176126/8;
        wt[7] = 0.157207/8;
        wt[8] = 0.176126/8;
        wt[9] = 0.157207/8;
        wt[10] = 0.157207/8;
        wt[11] = 0.176126/8;
        for(int i = 12; i < 24; i++)
            wt[i] = wt[i-12];
        for(int i = 24; i < 48; i++)
            wt[i] = wt[i-24];
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
