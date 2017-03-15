#include "ampxrecordparsers.h"

#include <QDebug>
#include <iostream>

void nextBinInt(ifstream &binfile, int &v)
{
    binfile.read((char*)&v, sizeof(int));
    endswap(&v);
    return;
}

int nextBinInt(ifstream &binfile)
{
    int x;
    binfile.read((char*)&x, sizeof(int));
    endswap(&x);
    return x;
}

void nextBinFloat(ifstream &binfile, float &v)
{
    binfile.read((char*)&v, sizeof(float));
    endswap(&v);
    return;
}

void nextBinFloats(ifstream &binfile, float &v, int count)
{
    binfile.read((char*)&v, sizeof(float)*count);
    for(int i = 0; i < count; i++)
        endswap(&v+sizeof(float)*i);
    return;
}

void nextBinString(ifstream &binfile, unsigned int words32bit, char array[])
{
    binfile.read(array, 4*words32bit);
    return;
}

// ---------------------------------------------------- //
// ----------------------  Base  ---------------------- //
// ---------------------------------------------------- //
void AmpxRecordParserBase::parse(ifstream &binfile)
{
    built = true;
    nextBinInt(binfile, recordSizePrev);
    nextBinInt(binfile, recordSize);
}

void AmpxRecordParserBase::parseHeader(ifstream &binfile)
{
    built = true;
    nextBinInt(binfile, recordSize);
}

// ---------------------------------------------------- //
// ---------------------- Type 1 ---------------------- //
// ---------------------------------------------------- //
AmpxRecordParserType1::AmpxRecordParserType1() :
    AmpxRecordParserBase(),
    idtape(-1),
    nnuc(-1),
    igm(-1),
    iftg(-1),
    msn(-1),
    ipm(-1),
    i1(-1),
    i2(-1),
    i3(-1),
    i4(-1),
    title()
{

}

AmpxRecordParserType1::~AmpxRecordParserType1()
{

}

bool AmpxRecordParserType1::parse(ifstream &binfile)
{
    AmpxRecordParserBase::parseHeader(binfile);
    nextBinInt(binfile, idtape);
    nextBinInt(binfile, nnuc);
    nextBinInt(binfile, igm);
    nextBinInt(binfile, iftg);
    nextBinInt(binfile, msn);
    nextBinInt(binfile, ipm);
    nextBinInt(binfile, i1);
    nextBinInt(binfile, i2);
    nextBinInt(binfile, i3);
    nextBinInt(binfile, i4);
    nextBinString(binfile, 100, title);

    return true;
}


// ---------------------------------------------------- //
// ---------------------- Type 2 ---------------------- //
// ---------------------------------------------------- //
bool AmpxRecordParserType2::parse(ifstream &binfile, int energyBins)
{
    //built = true;
    AmpxRecordParserBase::parse(binfile);

    energy.resize(energyBins+1);
    lethargy.resize(energyBins+1);

    for(int i = 0; i < energyBins+1; i++)
        nextBinFloat(binfile, energy[i]);

    for(int i = 0; i < energyBins+1; i++)
        nextBinFloat(binfile, lethargy[i]);

    return true;
}


// ---------------------------------------------------- //
// ---------------------- Type 3 ---------------------- //
// ---------------------------------------------------- //
AmpxRecordParserType3::AmpxRecordParserType3() :
    AmpxRecordParserBase(),
    text(),           // 1-18
    id(-1),           // 19
    nres(-1),         // 20
    nunr(-1),         // 21
    navg(-1),         // 22
    n2d(-1),          // 23
    w24(-1),          // 24
    gavg(-1),         // 25
    g2d(-1),          // 26
    ngct(-1),         // 27
    ord(-1),          // 28
    mass(-1),         // 29 (float)
    za(-1),           // 30 (float)
    w31(-1),          // 31
    w32(-1),          // 32
    w33(-1),          // 33
    pwr(-1),          // 34 (float)
    ec(-1),           // 35 (float)
    maxs(-1),         // 36
    nbond(-1),        // 37
    nsig(-1),         // 38
    nt(-1),           // 39
    nbondg(-1),       // 40
    w41(-1),          // 41
    w42(-1),          // 42
    sigp(-1),         // 43 (float)
    w44(-1),          // 44
    endffn(-1),       // 45
    endftn(-1),       // 46
    endfg(-1),        // 47
    endfgp(-1),       // 48
    sym(),            // 49 (char)
    nrec(-1)          // 50
{

}

AmpxRecordParserType3::~AmpxRecordParserType3()
{

}

bool AmpxRecordParserType3::parse(ifstream &binfile)
{
    AmpxRecordParserBase::parse(binfile);

    nextBinString(binfile, 18, text); // 1-18

    nextBinInt(binfile, id);      // 19
    nextBinInt(binfile, nres);    // 20
    nextBinInt(binfile, nunr);    // 21
    nextBinInt(binfile, navg);    // 22
    nextBinInt(binfile, n2d);     // 23
    nextBinInt(binfile, w24);     // 24
    nextBinInt(binfile, gavg);    // 25
    nextBinInt(binfile, g2d);     // 26
    nextBinInt(binfile, ngct);    // 27
    nextBinInt(binfile, ord);     // 28
    nextBinFloat(binfile, mass);  // 29
    nextBinFloat(binfile, za);    // 30
    nextBinInt(binfile, w31);     // 31
    nextBinInt(binfile, w32);     // 32
    nextBinInt(binfile, w33);     // 33
    nextBinFloat(binfile, pwr);   // 34
    nextBinFloat(binfile, ec);    // 35
    nextBinInt(binfile, maxs);    // 36
    nextBinInt(binfile, nbond);   // 37
    nextBinInt(binfile, nsig);    // 38
    nextBinInt(binfile, nt);      // 39
    nextBinInt(binfile, nbondg);  // 40
    nextBinInt(binfile, w41);     // 41
    nextBinInt(binfile, w42);     // 42
    nextBinFloat(binfile, sigp);  // 43
    nextBinInt(binfile, w44);     // 44
    nextBinInt(binfile, endffn);  // 45
    nextBinInt(binfile, endftn);  // 46
    nextBinInt(binfile, endfg);   // 47
    nextBinInt(binfile, endfgp);  // 48
    nextBinString(binfile, 1, sym);  // 49
    nextBinInt(binfile, nrec);    // 50

    return true;
}


// ---------------------------------------------------- //
// ---------------------- Type 4 ---------------------- //
// ---------------------------------------------------- //
bool AmpxRecordParserType4::parse(ifstream &binfile, int nres, int unres)
{
    throw "Record Type 4 is not implemented!";

    AmpxRecordParserBase::parse(binfile);

    return true;
}


// ---------------------------------------------------- //
// ---------------------- Type 5 ---------------------- //
// ---------------------------------------------------- //

AmpxRecordParserType5::AmpxRecordParserType5() :
    AmpxRecordParserBase(),
    sig0(),
    temp(),
    elo(-1),
    ehi(-1)
{

}

AmpxRecordParserType5::~AmpxRecordParserType5()
{

}

bool AmpxRecordParserType5::parse(ifstream &binfile, int nsig0, int nt)
{
    AmpxRecordParserBase::parse(binfile);

    sig0.resize(nsig0);
    temp.resize(nt);

    for(int i = 0; i < nsig0; i++)
        nextBinFloat(binfile, sig0[i]);

    for(int i = 0; i < nt; i++)
        nextBinFloat(binfile, temp[i]);

    nextBinFloat(binfile, elo);
    nextBinFloat(binfile, ehi);

    return true;
}


// ---------------------------------------------------- //
// ---------------------- Type 6 ---------------------- //
// ---------------------------------------------------- //
AmpxRecordParserType6::AmpxRecordParserType6() :
    AmpxRecordParserBase(),
    mt(),
    nf(),
    nl(),
    order(),
    ioff(),
    nz()
{

}

AmpxRecordParserType6::~AmpxRecordParserType6()
{

}
bool AmpxRecordParserType6::parse(ifstream &binfile, int nbond)
{
    AmpxRecordParserBase::parse(binfile);

    mt.resize(nbond);
    nf.resize(nbond);
    nl.resize(nbond);
    order.resize(nbond);
    ioff.resize(nbond);
    nz.resize(nbond);

    for(int i = 0; i < nbond; i++)
        nextBinInt(binfile, mt[i]);

    for(int i = 0; i < nbond; i++)
        nextBinInt(binfile, nf[i]);

    for(int i = 0; i < nbond; i++)
        nextBinInt(binfile, nl[i]);

    for(int i = 0; i < nbond; i++)
        nextBinInt(binfile, order[i]);

    for(int i = 0; i < nbond; i++)
        nextBinInt(binfile, ioff[i]);

    for(int i = 0; i < nbond; i++)
        nextBinInt(binfile, nz[i]);

    return true;
}


// ---------------------------------------------------- //
// ---------------------- Type 7 ---------------------- //
// ---------------------------------------------------- //
AmpxRecordParserType7::AmpxRecordParserType7() :
    AmpxRecordParserBase(),
    sigInf()
{

}

AmpxRecordParserType7::~AmpxRecordParserType7()
{

}

bool AmpxRecordParserType7::parse(ifstream &binfile, int firstGroup, int lastGroup)
{
    AmpxRecordParserBase::parse(binfile);

    sigInf.resize(lastGroup-firstGroup+1);

    for(int i = 0; i < lastGroup-firstGroup+1; i++)
        nextBinFloat(binfile, sigInf[i]);

    return true;
}


// ---------------------------------------------------- //
// ---------------------- Type 8 ---------------------- //
// ---------------------------------------------------- //
AmpxRecordParserType8::AmpxRecordParserType8() :
    AmpxRecordParserBase(),
    bond()
{

}

AmpxRecordParserType8::~AmpxRecordParserType8()
{

}

bool AmpxRecordParserType8::parse(ifstream &binfile, int firstGroup, int lastGroup, int num_t, int num_sig0)
{
    AmpxRecordParserBase::parse(binfile);

    nf = firstGroup;
    nl = lastGroup;
    nt = num_t;
    nsig0 = num_sig0;

    bond.resize((nl - nf + 1) * nt * nsig0);

    for(int i = 0; i < ((nl-nf+1)*nt*nsig0); i++)
        nextBinFloat(binfile, bond[i]);

    return true;
}

float AmpxRecordParserType8::getBondarenkoSig0(int eGroup, int tempIndx, int sig0Indx) const
{
    if(eGroup < nf || eGroup > nl)
        throw "";

    if(tempIndx >= nt)
        throw "";

    if(sig0Indx >= nsig0)
        throw "";

    return bond[(nf-nl+1)*nt*(eGroup-nf) + (nf-nl+1)*tempIndx + sig0Indx];
}


// ---------------------------------------------------- //
// ---------------------- Type 9 ---------------------- //
// ---------------------------------------------------- //
AmpxRecordParserType9::AmpxRecordParserType9() :
    AmpxRecordParserBase(),
    mt(),
    sigma(),
    groupCount(-1)
{

}

AmpxRecordParserType9::~AmpxRecordParserType9()
{

}

bool AmpxRecordParserType9::parse(ifstream &binfile, int mtCount, int groups)
{
    AmpxRecordParserBase::parse(binfile);

    mt.resize(mtCount);
    sigma.resize(mtCount * groups);
    groupCount = groups;

    for(int i = 0; i < mtCount; i++)
    {
        nextBinFloat(binfile, mt[i]);
        for(int j = 0; j < groups; j++)
            nextBinFloat(binfile, sigma[groups*i + j]);
    }

    return true;
}

std::vector<float> AmpxRecordParserType9::getSigma(const int mtIndex) const
{
    std::vector<float> sig;

    sig.resize(groupCount);

    for(int i = 0; i < groupCount; i++)
        sig[i] = sigma[mtIndex * mt.size() + i];

    return sig;
}

std::vector<float> AmpxRecordParserType9::getSigmaMt(int mtIndex) const
{
    std::vector<float> s;
    s.resize(groupCount, -1.0);

    for(int i = 0; i < groupCount; i++)
        s[i] = sigma[mtIndex*groupCount + i];

    return s;
}

int AmpxRecordParserType9::getMtIndex(const int mtId) const
{
    for(unsigned int i = 0; i < mt.size(); i++)
        if(mtId == mt[i])
            return i;
    return -1;
}

// ---------------------------------------------------- //
// ---------------------- Type 10 --------------------- //
// ---------------------------------------------------- //
AmpxRecordParserType10::AmpxRecordParserType10() :
    AmpxRecordParserBase(),
    mt(),
    l(),
    nl(),
    nt()
{

}

AmpxRecordParserType10::~AmpxRecordParserType10()
{

}

bool AmpxRecordParserType10::parse(ifstream &binfile, int nProcesses)
{
    AmpxRecordParserBase::parse(binfile);

    mt.resize(nProcesses);
    l.resize(nProcesses);
    nl.resize(nProcesses);
    nt.resize(nProcesses);

    for(int i = 0; i < nProcesses; i++)
        nextBinInt(binfile, mt[i]);

    for(int i = 0; i < nProcesses; i++)
        nextBinInt(binfile, l[i]);

    for(int i = 0; i < nProcesses; i++)
        nextBinInt(binfile, nl[i]);

    for(int i = 0; i < nProcesses; i++)
        nextBinInt(binfile, nt[i]);

    return true;
}

int AmpxRecordParserType10::getMtIndex(const int mtId) const
{
    for(unsigned int i = 0; i < mt.size(); i++)
        if(mtId == mt[i])
            return i;
    return -1;
}

int AmpxRecordParserType10::getNlIndex(const int nlId) const
{
    for(unsigned int i = 0; i < nl.size(); i++)
        if(nlId == nl[i])
            return i;
    return -1;
}


// ---------------------------------------------------- //
// ---------------------- Type 11 --------------------- //
// ---------------------------------------------------- //
AmpxRecordParserType11::AmpxRecordParserType11() :
    AmpxRecordParserBase(),
    temps()
{

}

AmpxRecordParserType11::~AmpxRecordParserType11()
{

}

bool AmpxRecordParserType11::parse(ifstream &binfile, int num_temps)
{
    AmpxRecordParserBase::parse(binfile);

    temps.resize(num_temps);

    for(int i = 0; i < num_temps; i++)
        nextBinFloat(binfile, temps[i]);

    return true;
}


// ---------------------------------------------------- //
// ---------------------- Type 12 --------------------- //
// ---------------------------------------------------- //
AmpxRecordParserType12::AmpxRecordParserType12() :
    AmpxRecordParserBase(),
    size(-1),
    maxSize(-1),
    groups(-1),
    magics(),
    xsData()
{

}

AmpxRecordParserType12::~AmpxRecordParserType12()
{
}

bool AmpxRecordParserType12::parse(ifstream &binfile, int maxlen, bool selfDefining, int nGroups)
{
    AmpxRecordParserBase::parse(binfile);

    if(selfDefining)
        nextBinInt(binfile, maxSize);
    else
        maxSize = maxlen;

    size = 0;
    groups = nGroups;
    xsData.resize(groups*groups, 0.0f);

    while(size < maxSize)
    {

        int nextMagic;
        nextBinInt(binfile, nextMagic);
        magics.push_back(nextMagic);
        size++;

        // Negative or zero magic specifies the end of data
        if(nextMagic <= 0)
        {
            int padBytes = recordSize - size*4;
            binfile.ignore(padBytes);
            return false;
        }

        int sinkGroup = nextMagic % 1000;                                        // III
        int lastScatGroup = (nextMagic % 1000000 - sinkGroup) / 1000;            // KKK
        int firstScatGroup = (nextMagic - lastScatGroup - sinkGroup) / 1000000;  // JJJ

        if(sinkGroup > groups || lastScatGroup > groups || firstScatGroup > groups)
        {
            throw "RecordType12: illegal group";
            return false;
        }

        for(int src = lastScatGroup; src >= firstScatGroup; src--)
        {
            nextBinFloat(binfile, xsData[(src-1)*groups + sinkGroup-1]);
            size++;
        }
    }

    return true;
}

float AmpxRecordParserType12::getXs(int srcEGrpIndx, int sinkEGrpIndx) const
{
    if(srcEGrpIndx >= groups || sinkEGrpIndx >= groups || srcEGrpIndx < 0 || sinkEGrpIndx < 0)
    {
        qDebug() << "Record 12: Illegal index value when accessing data";
        return false;
    }
    return xsData[(srcEGrpIndx)*groups + sinkEGrpIndx];
}


const std::vector<float> &AmpxRecordParserType12::getXsVector() const
{
    return xsData;
}





