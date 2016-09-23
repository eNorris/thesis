#ifndef AMPXLIB_RESOURCES_H
#define AMPXLIB_RESOURCES_H

/// enumerated types for parsing data
#define AMPXLIB_NEUTRON1D_DATA 0
#define AMPXLIB_GAMMA1D_DATA 1
#define AMPXLIB_NEUTRON2D_DATA 2
#define AMPXLIB_GAMMA2D_DATA 3
// Both GammaProduction and TOTAL_XS have the same self defining format
#define AMPXLIB_GAMMAPRODUCTION_DATA 4
#define AMPXLIB_TOTAL_XS_DATA 4


int headerSize();///{return sizeof(int);}
int footerSize();///{return sizeof(int);}
bool explodeMagicWord(const int magicWord, int &grp, int &start, int &end, int & length);
int getMagicWord(int grp, int start, int end, bool & error);
#endif
