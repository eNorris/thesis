#include "quadrature.h"

#include <QDebug>
#include <string>
#include <qmath.h>

Quadrature::Quadrature() : m_angles(0)
{

}

Quadrature::Quadrature(const int sn)
{
    loadSn(sn);
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
    else if(special == 3)  // Unidirectional
    {
        m_angles = 1;
        mu.push_back(1.0);
        zi.push_back(0.0);
        eta.push_back(0.0);
        wt.push_back(1.0);
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
        mu[0] = -.577350f;
        mu[1] =  .577350f;
        mu[2] = -.577350f;
        mu[3] =  .577350f;
        for(int i = 4; i < 8; i++)
            mu[i] = mu[i-4];

        for(int i = 0; i < 4; i++)
            eta[i] = -.577350f;
        for(int i = 4; i < 8; i++)
            eta[i] = .577350f;

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
        mu[0] = -0.868890f;
        mu[1] =  -0.350021f;
        mu[2] = -0.350021f;
        mu[3] =  0.350021f;
        mu[4] = .350021f;
        mu[5] = 0.868890f;
        for(int i = 6; i < 12; i++)
            mu[i] = mu[i-6];
        for(int i = 12; i < 24; i++)
            mu[i] = mu[i-12];

        eta[0] = -0.350021f;
        eta[1] = -0.350021f;
        eta[2] = -0.868890f;
        eta[3] = -0.868890f;
        eta[4] = -0.350021f;
        eta[5] = -0.350021f;
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
            wt[i] = 1.0f / 24.0f;
    }
    else if(sn == 6)
    {
        mu[0] = -0.926181f;
        mu[1] = -0.681508f;
        mu[2] = -0.681508f;
        mu[3] = -0.266636f;
        mu[4] = -0.266636f;
        mu[5] = -0.266636f;
        mu[6] =  0.266636f;
        mu[7] =  0.266636f;
        mu[8] =  0.266636f;
        mu[9] = 0.681508f;
        mu[10] = 0.681508f;
        mu[11] = 0.926181f;
        for(int i = 12; i < 24; i++)
            mu[i] = mu[i-12];
        for(int i = 24; i < 48; i++)
            mu[i] = mu[i-24];

        eta[0] = -0.266636f;
        eta[1] = -0.266636f;
        eta[2] = -0.681508f;
        eta[3] = -0.266636f;
        eta[4] = -0.681508f;
        eta[5] = -0.926181f;
        eta[6] = -0.926181f;
        eta[7] = -0.681508f;
        eta[8] = -0.266636f;
        eta[9] = -0.681508f;
        eta[10] = -0.266636f;
        eta[11] = -0.266636f;
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

        wt[0] = 0.176126f/8.0f;
        wt[1] = 0.157207f/8.0f;
        wt[2] = 0.157207f/8.0f;
        wt[3] = 0.176126f/8.0f;
        wt[4] = 0.157207f/8.0f;
        wt[5] = 0.176126f/8.0f;
        wt[6] = 0.176126f/8.0f;
        wt[7] = 0.157207f/8.0f;
        wt[8] = 0.176126f/8.0f;
        wt[9] = 0.157207f/8.0f;
        wt[10] = 0.157207f/8.0f;
        wt[11] = 0.176126f/8.0f;
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

int Quadrature::angleCount() const
{
    return m_angles;
}
