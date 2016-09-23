#ifndef Interface_ABSTRACTCROSSSECTION1D_H
#define Interface_ABSTRACTCROSSSECTION1D_H

#include<iostream>
#include<vector>
#include<string>

namespace Standard {

/**
 * @class AbstractCrossSection1d
 * @brief Simple interface to a 1d cross section
 */
class AbstractCrossSection1d {
public:

    virtual float * getValues()=0;
    virtual void setValues(float * values, int size)=0;

    virtual int getSize()=0;
    virtual void setSize(int size)=0;

    virtual int getMt()=0;
    virtual void setMt(int mt)=0;

    virtual float getAt(int index)=0;
    virtual void setAt(int index, float value)=0;

private:

};
}//namespace Standard

#endif	/* Interface_ABSTRACTCROSSSECTION1D_H */
