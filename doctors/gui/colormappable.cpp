#include "colormappable.h"

ColorMappable::ColorMappable() : brushes(), errBrush(Qt::green), m_colorId(NONE)
{

}

void ColorMappable::loadParulaBrush()
{

    if(m_colorId == PARULA)
        return;
    m_colorId = PARULA;

    brushes.clear();

    brushes.push_back(QBrush(QColor::fromRgbF(0.2081,    0.1663,    0.5292)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.2116,    0.1898,    0.5777)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.2123,    0.2138,    0.6270)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.2081,    0.2386,    0.6771)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.1959,    0.2645,    0.7279)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.1707,    0.2919,    0.7792)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.1253,    0.3242,    0.8303)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0591,    0.3598,    0.8683)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0117,    0.3875,    0.8820)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0060,    0.4086,    0.8828)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0165,    0.4266,    0.8786)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0329,    0.4430,    0.8720)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0498,    0.4586,    0.8641)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0629,    0.4737,    0.8554)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0723,    0.4887,    0.8467)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0779,    0.5040,    0.8384)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0793,    0.5200,    0.8312)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0749,    0.5375,    0.8263)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0641,    0.5570,    0.8240)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0488,    0.5772,    0.8228)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0343,    0.5966,    0.8199)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0265,    0.6137,    0.8135)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0239,    0.6287,    0.8038)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0231,    0.6418,    0.7913)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0228,    0.6535,    0.7768)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0267,    0.6642,    0.7607)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0384,    0.6743,    0.7436)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0590,    0.6838,    0.7254)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.0843,    0.6928,    0.7062)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.1133,    0.7015,    0.6859)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.1453,    0.7098,    0.6646)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.1801,    0.7177,    0.6424)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.2178,    0.7250,    0.6193)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.2586,    0.7317,    0.5954)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.3022,    0.7376,    0.5712)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.3482,    0.7424,    0.5473)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.3953,    0.7459,    0.5244)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.4420,    0.7481,    0.5033)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.4871,    0.7491,    0.4840)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.5300,    0.7491,    0.4661)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.5709,    0.7485,    0.4494)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.6099,    0.7473,    0.4337)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.6473,    0.7456,    0.4188)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.6834,    0.7435,    0.4044)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.7184,    0.7411,    0.3905)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.7525,    0.7384,    0.3768)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.7858,    0.7356,    0.3633)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.8185,    0.7327,    0.3498)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.8507,    0.7299,    0.3360)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.8824,    0.7274,    0.3217)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.9139,    0.7258,    0.3063)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.9450,    0.7261,    0.2886)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.9739,    0.7314,    0.2666)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.9938,    0.7455,    0.2403)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.9990,    0.7653,    0.2164)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.9955,    0.7861,    0.1967)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.9880,    0.8066,    0.1794)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.9789,    0.8271,    0.1633)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.9697,    0.8481,    0.1475)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.9626,    0.8705,    0.1309)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.9589,    0.8949,    0.1132)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.9598,    0.9218,    0.0948)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.9661,    0.9514,    0.0755)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.9763,    0.9831,    0.0538)));
}

void ColorMappable::loadUniqueBrush()
{

    if(m_colorId == UNIQUE)
        return;
    m_colorId = UNIQUE;

    brushes.clear();

    brushes.push_back(QBrush(QColor::fromRgbF(0.0000,    0.4470,    0.7410)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.8500,    0.3250,    0.0980)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.9290,    0.6940,    0.1250)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.4940,    0.1840,    0.5560)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.4660,    0.6740,    0.1880)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.3010,    0.7450,    0.9330)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.6350,    0.0780,    0.1840)));
}

void ColorMappable::loadPhantom19Brush()
{

    if(m_colorId == PHANTOM19)
        return;
    m_colorId = PHANTOM19;

    brushes.clear();

    brushes.push_back(QBrush(QColor::fromRgbF(0.0000,    1.0000,    1.0000)));  // 1 - air
    brushes.push_back(QBrush(QColor::fromRgbF(0.0000,    0.0000,    1.0000)));  // 2 - lung
    brushes.push_back(QBrush(QColor::fromRgbF(1.0000,    0.0000,    1.0000)));  // 3 - adipose/adrenal
    brushes.push_back(QBrush(QColor::fromRgbF(1.0000,    0.5000,    0.5000)));  // 4 - intestine/connective tissue
    brushes.push_back(QBrush(QColor::fromRgbF(0.267004,  0.004874,  0.329415)));  // 5 - bone
    brushes.push_back(QBrush(QColor::fromRgbF(0.282656,  0.100196,  0.42216)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.277134,  0.185228,  0.489898)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.253935,  0.265254,  0.529983)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.221989,  0.339161,  0.548752)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.190631,  0.407061,  0.556089)));  // 10
    brushes.push_back(QBrush(QColor::fromRgbF(0.163625,  0.471133,  0.558148)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.139147,  0.533812,  0.555298)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.120565,  0.596422,  0.543611)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.134692,  0.658636,  0.517649)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.20803,   0.718701,  0.472873)));  // 15
    brushes.push_back(QBrush(QColor::fromRgbF(0.327796,  0.77398,   0.40664)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.477504,  0.821444,  0.318195)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.647257,  0.8584,    0.209861)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.82494,   0.88472,   0.106217)));  // 19
    brushes.push_back(QBrush(QColor::fromRgbF(1.0000,    0.0000,    1.0000)));  // 20 - Illegal
}

void ColorMappable::loadViridis256Brush()
{

    if(m_colorId == VIRIDIS256)
        return;
    m_colorId = VIRIDIS256;

    brushes.clear();

    brushes.push_back(QBrush(QColor::fromRgbF(0.267004,    0.004874,    0.329415)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.26851,    0.009605,    0.335427)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.269944,    0.014625,    0.341379)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.271305,    0.019942,    0.347269)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.272594,    0.025563,    0.353093)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.273809,    0.031497,    0.358853)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.274952,    0.037752,    0.364543)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.276022,    0.044167,    0.370164)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.277018,    0.050344,    0.375715)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.277941,    0.056324,    0.381191)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.278791,    0.062145,    0.386592)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.279566,    0.067836,    0.391917)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.280267,    0.073417,    0.397163)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.280894,    0.078907,    0.402329)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.281446,    0.08432,    0.407414)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.281924,    0.089666,    0.412415)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.282327,    0.094955,    0.417331)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.282656,    0.100196,    0.42216)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.28291,    0.105393,    0.426902)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.283091,    0.110553,    0.431554)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.283197,    0.11568,    0.436115)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.283229,    0.120777,    0.440584)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.283187,    0.125848,    0.44496)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.283072,    0.130895,    0.449241)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.282884,    0.13592,    0.453427)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.282623,    0.140926,    0.457517)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.28229,    0.145912,    0.46151)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.281887,    0.150881,    0.465405)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.281412,    0.155834,    0.469201)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.280868,    0.160771,    0.472899)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.280255,    0.165693,    0.476498)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.279574,    0.170599,    0.479997)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.278826,    0.17549,    0.483397)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.278012,    0.180367,    0.486697)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.277134,    0.185228,    0.489898)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.276194,    0.190074,    0.493001)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.275191,    0.194905,    0.496005)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.274128,    0.199721,    0.498911)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.273006,    0.20452,    0.501721)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.271828,    0.209303,    0.504434)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.270595,    0.214069,    0.507052)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.269308,    0.218818,    0.509577)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.267968,    0.223549,    0.512008)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.26658,    0.228262,    0.514349)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.265145,    0.232956,    0.516599)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.263663,    0.237631,    0.518762)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.262138,    0.242286,    0.520837)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.260571,    0.246922,    0.522828)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.258965,    0.251537,    0.524736)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.257322,    0.25613,    0.526563)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.255645,    0.260703,    0.528312)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.253935,    0.265254,    0.529983)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.252194,    0.269783,    0.531579)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.250425,    0.27429,    0.533103)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.248629,    0.278775,    0.534556)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.246811,    0.283237,    0.535941)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.244972,    0.287675,    0.53726)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.243113,    0.292092,    0.538516)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.241237,    0.296485,    0.539709)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.239346,    0.300855,    0.540844)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.237441,    0.305202,    0.541921)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.235526,    0.309527,    0.542944)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.233603,    0.313828,    0.543914)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.231674,    0.318106,    0.544834)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.229739,    0.322361,    0.545706)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.227802,    0.326594,    0.546532)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.225863,    0.330805,    0.547314)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.223925,    0.334994,    0.548053)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.221989,    0.339161,    0.548752)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.220057,    0.343307,    0.549413)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.21813,    0.347432,    0.550038)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.21621,    0.351535,    0.550627)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.214298,    0.355619,    0.551184)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.212395,    0.359683,    0.55171)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.210503,    0.363727,    0.552206)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.208623,    0.367752,    0.552675)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.206756,    0.371758,    0.553117)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.204903,    0.375746,    0.553533)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.203063,    0.379716,    0.553925)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.201239,    0.38367,    0.554294)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.19943,    0.387607,    0.554642)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.197636,    0.391528,    0.554969)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.19586,    0.395433,    0.555276)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.1941,    0.399323,    0.555565)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.192357,    0.403199,    0.555836)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.190631,    0.407061,    0.556089)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.188923,    0.41091,    0.556326)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.187231,    0.414746,    0.556547)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.185556,    0.41857,    0.556753)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.183898,    0.422383,    0.556944)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.182256,    0.426184,    0.55712)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.180629,    0.429975,    0.557282)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.179019,    0.433756,    0.55743)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.177423,    0.437527,    0.557565)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.175841,    0.44129,    0.557685)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.174274,    0.445044,    0.557792)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.172719,    0.448791,    0.557885)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.171176,    0.45253,    0.557965)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.169646,    0.456262,    0.55803)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.168126,    0.459988,    0.558082)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.166617,    0.463708,    0.558119)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.165117,    0.467423,    0.558141)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.163625,    0.471133,    0.558148)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.162142,    0.474838,    0.55814)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.160665,    0.47854,    0.558115)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.159194,    0.482237,    0.558073)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.157729,    0.485932,    0.558013)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.15627,    0.489624,    0.557936)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.154815,    0.493313,    0.55784)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.153364,    0.497,    0.557724)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.151918,    0.500685,    0.557587)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.150476,    0.504369,    0.55743)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.149039,    0.508051,    0.55725)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.147607,    0.511733,    0.557049)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.14618,    0.515413,    0.556823)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.144759,    0.519093,    0.556572)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.143343,    0.522773,    0.556295)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.141935,    0.526453,    0.555991)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.140536,    0.530132,    0.555659)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.139147,    0.533812,    0.555298)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.13777,    0.537492,    0.554906)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.136408,    0.541173,    0.554483)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.135066,    0.544853,    0.554029)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.133743,    0.548535,    0.553541)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.132444,    0.552216,    0.553018)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.131172,    0.555899,    0.552459)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.129933,    0.559582,    0.551864)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.128729,    0.563265,    0.551229)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.127568,    0.566949,    0.550556)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.126453,    0.570633,    0.549841)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.125394,    0.574318,    0.549086)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.124395,    0.578002,    0.548287)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.123463,    0.581687,    0.547445)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.122606,    0.585371,    0.546557)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.121831,    0.589055,    0.545623)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.121148,    0.592739,    0.544641)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.120565,    0.596422,    0.543611)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.120092,    0.600104,    0.54253)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.119738,    0.603785,    0.5414)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.119512,    0.607464,    0.540218)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.119423,    0.611141,    0.538982)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.119483,    0.614817,    0.537692)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.119699,    0.61849,    0.536347)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.120081,    0.622161,    0.534946)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.120638,    0.625828,    0.533488)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.12138,    0.629492,    0.531973)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.122312,    0.633153,    0.530398)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.123444,    0.636809,    0.528763)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.12478,    0.640461,    0.527068)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.126326,    0.644107,    0.525311)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.128087,    0.647749,    0.523491)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.130067,    0.651384,    0.521608)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.132268,    0.655014,    0.519661)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.134692,    0.658636,    0.517649)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.137339,    0.662252,    0.515571)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.14021,    0.665859,    0.513427)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.143303,    0.669459,    0.511215)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.146616,    0.67305,    0.508936)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.150148,    0.676631,    0.506589)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.153894,    0.680203,    0.504172)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.157851,    0.683765,    0.501686)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.162016,    0.687316,    0.499129)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.166383,    0.690856,    0.496502)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.170948,    0.694384,    0.493803)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.175707,    0.6979,    0.491033)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.180653,    0.701402,    0.488189)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.185783,    0.704891,    0.485273)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.19109,    0.708366,    0.482284)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.196571,    0.711827,    0.479221)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.202219,    0.715272,    0.476084)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.20803,    0.718701,    0.472873)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.214,    0.722114,    0.469588)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.220124,    0.725509,    0.466226)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.226397,    0.728888,    0.462789)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.232815,    0.732247,    0.459277)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.239374,    0.735588,    0.455688)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.24607,    0.73891,    0.452024)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.252899,    0.742211,    0.448284)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.259857,    0.745492,    0.444467)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.266941,    0.748751,    0.440573)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.274149,    0.751988,    0.436601)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.281477,    0.755203,    0.432552)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.288921,    0.758394,    0.428426)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.296479,    0.761561,    0.424223)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.304148,    0.764704,    0.419943)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.311925,    0.767822,    0.415586)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.319809,    0.770914,    0.411152)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.327796,    0.77398,    0.40664)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.335885,    0.777018,    0.402049)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.344074,    0.780029,    0.397381)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.35236,    0.783011,    0.392636)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.360741,    0.785964,    0.387814)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.369214,    0.788888,    0.382914)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.377779,    0.791781,    0.377939)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.386433,    0.794644,    0.372886)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.395174,    0.797475,    0.367757)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.404001,    0.800275,    0.362552)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.412913,    0.803041,    0.357269)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.421908,    0.805774,    0.35191)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.430983,    0.808473,    0.346476)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.440137,    0.811138,    0.340967)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.449368,    0.813768,    0.335384)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.458674,    0.816363,    0.329727)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.468053,    0.818921,    0.323998)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.477504,    0.821444,    0.318195)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.487026,    0.823929,    0.312321)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.496615,    0.826376,    0.306377)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.506271,    0.828786,    0.300362)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.515992,    0.831158,    0.294279)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.525776,    0.833491,    0.288127)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.535621,    0.835785,    0.281908)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.545524,    0.838039,    0.275626)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.555484,    0.840254,    0.269281)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.565498,    0.84243,    0.262877)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.575563,    0.844566,    0.256415)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.585678,    0.846661,    0.249897)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.595839,    0.848717,    0.243329)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.606045,    0.850733,    0.236712)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.616293,    0.852709,    0.230052)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.626579,    0.854645,    0.223353)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.636902,    0.856542,    0.21662)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.647257,    0.8584,    0.209861)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.657642,    0.860219,    0.203082)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.668054,    0.861999,    0.196293)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.678489,    0.863742,    0.189503)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.688944,    0.865448,    0.182725)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.699415,    0.867117,    0.175971)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.709898,    0.868751,    0.169257)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.720391,    0.87035,    0.162603)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.730889,    0.871916,    0.156029)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.741388,    0.873449,    0.149561)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.751884,    0.874951,    0.143228)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.762373,    0.876424,    0.137064)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.772852,    0.877868,    0.131109)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.783315,    0.879285,    0.125405)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.79376,    0.880678,    0.120005)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.804182,    0.882046,    0.114965)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.814576,    0.883393,    0.110347)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.82494,    0.88472,    0.106217)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.83527,    0.886029,    0.102646)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.845561,    0.887322,    0.099702)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.85581,    0.888601,    0.097452)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.866013,    0.889868,    0.095953)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.876168,    0.891125,    0.09525)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.886271,    0.892374,    0.095374)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.89632,    0.893616,    0.096335)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.906311,    0.894855,    0.098125)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.916242,    0.896091,    0.100717)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.926106,    0.89733,    0.104071)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.935904,    0.89857,    0.108131)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.945636,    0.899815,    0.112838)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.9553,    0.901065,    0.118128)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.964894,    0.902323,    0.123941)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.974417,    0.90359,    0.130215)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.983868,    0.904867,    0.136897)));
    brushes.push_back(QBrush(QColor::fromRgbF(0.993248,    0.906157,    0.143936)));
}




















