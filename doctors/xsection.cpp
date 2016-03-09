#include "xsection.h"

#include <cmath>

#include <QDebug>

XSection::XSection()
{

}

XSection::XSection(const Config &config)
{
    load(config);
}

XSection::~XSection()
{

}

void XSection::load(const Config &config)
{
    if(config.njoy)
    {
        qDebug() << "Sorry, still under development";
    }
    else
    {
        qDebug() << "Reading XS locally";

        int energies = config.igm;

        // TODO - What is this?
        int t1 = round(config.mtm / (1 + config.isct));
        int t2 = 1 + config.isct;
        int t3 = 2 + config.ihm - config.ihs;

        msig.resize(energies * t1 * t2 * t3);

        for(int i = 0; i < energies; i++)
            for(int j = 0; j < t1; j++)
                for(int k = 0; k < t2; k++)
                    for(int m = 0; m < t3; m++)
                        msig[i*t1*t2*t3 + j*t2*t3 + k*t3 + m] = config.xsection[((j-1) * (1+config.isct) + k)*4 + config.iht + m - 1];
    }
}
