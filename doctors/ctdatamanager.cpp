#include "ctdatamanager.h"

#include <fstream>
#include <cstdint>
#include <QDebug>
#include <fstream>

#include "materialutils.h"

/*
const std::vector<int> CtDataManager::hounsfieldRangePhantom19{
    -950, // 1 - air
    -100, // 2 - lung
    15,   // 3 - adipose/adrenal
    129,  // 4 - intestine/connective tissue
    200,  // 5 - bone
    300,
    400,
    500,
    600,
    700,  // 10
    800,
    900,
    1000,
    1100,
    1200,  // 15
    1300,
    1400,
    1500,
    3000,  // 19
    3000,  // 20
};

const std::vector<int> CtDataManager::hounsfieldRangePhantom19Elements{
    1,     6,     7,     8,     11,    12,    15,    16,    17,    18,    19,    20
};

const std::vector<std::vector<float> > CtDataManager::hounsfieldRangePhantom19Weights{
    std::vector<float> {0.000, 0.000, 0.757, 0.232, 0.000, 0.000, 0.000, 0.000, 0.000, 0.013, 0.000, 0.000}, // Air
    std::vector<float> {0.103, 0.105, 0.031, 0.749, 0.002, 0.000, 0.002, 0.003, 0.003, 0.000, 0.002, 0.000}, // Lung
    std::vector<float> {0.112, 0.508, 0.012, 0.364, 0.001, 0.000, 0.000, 0.001, 0.001, 0.000, 0.000, 0.000}, // Adipose/adrenal
    std::vector<float> {0.100, 0.163, 0.043, 0.684, 0.004, 0.000, 0.000, 0.004, 0.003, 0.000, 0.000, 0.000}, // Small intestine
    std::vector<float> {0.097, 0.447, 0.025, 0.359, 0.000, 0.000, 0.023, 0.002, 0.001, 0.000, 0.001, 0.045}, // Bone
    std::vector<float> {0.091, 0.414, 0.027, 0.368, 0.000, 0.001, 0.032, 0.002, 0.001, 0.000, 0.001, 0.063}, // Bone
    std::vector<float> {0.085, 0.378, 0.029, 0.379, 0.000, 0.001, 0.041, 0.002, 0.001, 0.000, 0.001, 0.082}, // Bone
    std::vector<float> {0.080, 0.345, 0.031, 0.388, 0.000, 0.001, 0.050, 0.002, 0.001, 0.000, 0.001, 0.010}, // Bone
    std::vector<float> {0.075, 0.316, 0.032, 0.397, 0.000, 0.001, 0.058, 0.002, 0.001, 0.000, 0.000, 0.116}, // Bone
    std::vector<float> {0.071, 0.289, 0.034, 0.404, 0.000, 0.001, 0.066, 0.002, 0.001, 0.000, 0.000, 0.131}, // Bone
    std::vector<float> {0.067, 0.264, 0.035, 0.412, 0.000, 0.002, 0.072, 0.003, 0.000, 0.000, 0.000, 0.144}, // Bone
    std::vector<float> {0.063, 0.242, 0.037, 0.418, 0.000, 0.002, 0.078, 0.003, 0.000, 0.000, 0.000, 0.157}, // Bone
    std::vector<float> {0.060, 0.221, 0.038, 0.424, 0.000, 0.002, 0.084, 0.003, 0.000, 0.000, 0.000, 0.168}, // Bone
    std::vector<float> {0.056, 0.201, 0.039, 0.430, 0.000, 0.002, 0.089, 0.003, 0.000, 0.000, 0.000, 0.179}, // Bone
    std::vector<float> {0.053, 0.183, 0.040, 0.435, 0.000, 0.002, 0.094, 0.003, 0.000, 0.000, 0.000, 0.189}, // Bone
    std::vector<float> {0.051, 0.166, 0.041, 0.440, 0.000, 0.002, 0.099, 0.003, 0.000, 0.000, 0.000, 0.198}, // Bone
    std::vector<float> {0.048, 0.150, 0.042, 0.444, 0.000, 0.002, 0.103, 0.003, 0.000, 0.000, 0.000, 0.207}, // Bone
    std::vector<float> {0.046, 0.136, 0.042, 0.449, 0.000, 0.002, 0.107, 0.003, 0.000, 0.000, 0.000, 0.215}, // Bone
    std::vector<float> {0.043, 0.122, 0.043, 0.453, 0.000, 0.002, 0.111, 0.003, 0.000, 0.000, 0.000, 0.222}, // Bone
};
*/

CtDataManager::CtDataManager() : m_valid(false)
{
    //setup();
}

CtDataManager::~CtDataManager()
{

}

//Mesh CtDataReader::parse_256_256_64_16(std::string filename)
Mesh *CtDataManager::parse16(int xbins, int ybins, int zbins, std::string filename)
{
    Mesh *m = new Mesh;
    long tbins = xbins * ybins * zbins;

    m->uniform(xbins, ybins, zbins, 50.0, 50.0, 12.5);
    m->initCtVariables();

    std::vector<u_int16_t> zoneIds;
    zoneIds.resize(tbins);
    //zoneIds.resize(50);

    std::ifstream fin(filename.c_str(), std::ios::binary);

    if(fin.good())
    {
        int numx = fin.tellg()/2;
        qDebug() << "numx = " << numx;

        for(int i = 0; i < tbins; i++)
        {
            fin.read((char*)&zoneIds[i], 2);
        }
    }
    else
    {
        return NULL;
    }

    int gindx = 0;
    int YX = 256 * 256;
    int X = 256;
    for(int i = 0; i < xbins; i++)
        for(int j = 0; j < ybins; j++)
            for(int k = 0; k < zbins; k++)
            {
                m->ct[gindx] = zoneIds[k*YX + j*X + i];
                gindx++;
            }

    //return ctNumberToQuickCheck(m);
    return ctNumberToHumanPhantom(m);
}

Mesh *CtDataManager::ctNumberToQuickCheck(Mesh *mesh)
{
    float maxi = -1;
    float mini = 1E10;
    for(unsigned int i = 0; i < mesh->voxelCount(); i++)
    {
        int ctv = mesh->ct[i];

        // Determine the density
        if(ctv <= 55)
            mesh->density[i] = 0.001 * (1.02 * ctv - 7.65);
        else
            mesh->density[i] = 0.001 * (0.58 * ctv + 467.79);

        if(mesh->density[i] > maxi)
            maxi = mesh->density[i];
        if(mesh->density[i] < mini)
            mini = mesh->density[i];

        // Determine the material
        mesh->zoneId[i] = mesh->density[i]*1.68;
        if(mesh->zoneId[i] > 63)
            mesh->zoneId[i] = 63;
    }

    qDebug() << "Max: " << maxi << "   Min: " << mini;

    return mesh;
}

Mesh *CtDataManager::ctNumberToHumanPhantom(Mesh *mesh)
{
    std::vector<float> atomPerG;

    std::vector<std::vector<float> > hounsfieldRangePhantom19Fractions;

    // Convert the weight fractions to atom fractions
    for(unsigned int i = 0; i < MaterialUtils::hounsfieldRangePhantom19Elements.size(); i++)
        hounsfieldRangePhantom19Fractions.push_back(MaterialUtils::weightFracToAtomFrac(MaterialUtils::hounsfieldRangePhantom19Elements, MaterialUtils::hounsfieldRangePhantom19Weights[i]));

    // Convert atom fraction to atom density
    for(unsigned int i = 0; i < MaterialUtils::hounsfieldRangePhantom19Elements.size(); i++)
        atomPerG.push_back(MaterialUtils::atomsPerGram(MaterialUtils::hounsfieldRangePhantom19Elements, hounsfieldRangePhantom19Fractions[i]));

    for(unsigned int i = 0; i < mesh->voxelCount(); i++)
    {
        int ctv = mesh->ct[i];

        // Determine the density (atom density is after the zoneId calculation)
        if(ctv <= 55)
            mesh->density[i] = 0.001 * (1.02 * ctv - 7.65);
        else
            mesh->density[i] = 0.001 * (0.58 * ctv + 467.79);

        // Determine the material
        mesh->zoneId[i] = MaterialUtils::hounsfieldRangePhantom19.size() - 1;  // Last bin is "illegal"
        for(unsigned int j = 0; j < MaterialUtils::hounsfieldRangePhantom19.size(); j++)
        {
            //int ctvm = ctv - 1024;
            //int hsf = hounsfieldRangePhantom19[j];
            //bool truth = ctvm < hsf;
            if(ctv - 1024 < MaterialUtils::hounsfieldRangePhantom19[j])
            {
                mesh->zoneId[i] = j;
                break;
            }
        }

        mesh->atomDensity[i] = mesh->density[i] * atomPerG[mesh->zoneId[i]] * 1E-24;  // g/cc * @/g * cm^2/b = @/cm-b
    }

    return mesh;
}

//void CtDataManager::setup()
//{
    /*
    hounsfieldRangePhantom19.push_back(-950); // 1 - air
    hounsfieldRangePhantom19.push_back(-100); // 2 - lung
    hounsfieldRangePhantom19.push_back(15);   // 3 - adipose/adrenal
    hounsfieldRangePhantom19.push_back(129);  // 4 - intestine/connective tissue
    hounsfieldRangePhantom19.push_back(200);  // 5 - bone
    hounsfieldRangePhantom19.push_back(300);
    hounsfieldRangePhantom19.push_back(400);
    hounsfieldRangePhantom19.push_back(500);
    hounsfieldRangePhantom19.push_back(600);
    hounsfieldRangePhantom19.push_back(700);  // 10
    hounsfieldRangePhantom19.push_back(800);
    hounsfieldRangePhantom19.push_back(900);
    hounsfieldRangePhantom19.push_back(1000);
    hounsfieldRangePhantom19.push_back(1100);
    hounsfieldRangePhantom19.push_back(1200);  // 15
    hounsfieldRangePhantom19.push_back(1300);
    hounsfieldRangePhantom19.push_back(1400);
    hounsfieldRangePhantom19.push_back(1500);
    hounsfieldRangePhantom19.push_back(3000);  // 19
    hounsfieldRangePhantom19.push_back(3000);  // 20 - Illegal
    */
//}


