#include "materialutils.h"

#include <iostream>
#include <QDebug>

const float MaterialUtils::AVOGADRO = 6.0221409E23;

const std::vector<std::string> MaterialUtils::elementNames {
    "Hydrogen",
    "Helium",
    "Lithium",
    "Beryllium",
    "Boron",
    "Carbon",
    "Nitrogen",
    "Oxygen",
    "Fluorine",
    "Neon",
    "Sodium",
    "Magnesium",
    "Aluminum",
    "Silicon",
    "Phosphorus",
    "Sulfur",
    "Chlorine",
    "Argon",
    "Potassium",
    "Calcium",
    "Scandium",
    "Titanium",
    "Vanadium",
    "Chromium",
    "Manganese",
    "Iron",
    "Cobalt",
    "Nickel",
    "Copper",
    "Zinc",
    "Gallium",
    "Germanium",
    "Arsenic",
    "Selenium",
    "Bromine",
    "Krypton",
    "Rubidium",
    "Strontium",
    "Yttrium",
    "Zirconium",
    "Niobium",
    "Molybdenum",
    "Technetium",
    "Ruthenium",
    "Rhodium",
    "Palladium",
    "Silver",
    "Cadmium",
    "Indium",
    "Tin",
    "Antimony",
    "Tellurium",
    "Iodine",
    "Xenon",
    "Cesium",
    "Barium",
    "Lanthanum",
    "Cerium",
    "Praseodymium",
    "Neodymium",
    "Promethium",
    "Samarium",
    "Europium",
    "Gadolinium",
    "Terbium",
    "Dysprosium",
    "Holmium",
    "Erbium",
    "Thulium",
    "Ytterbium",
    "Lutetium",
    "Hafnium",
    "Tantalum",
    "Tungsten",
    "Rhenium",
    "Osmium",
    "Iridium",
    "Platinum",
    "Gold",
    "Mercury",
    "Thallium",
    "Lead",
    "Bismuth",
    "Polonium",
    "Astatine",
    "Radon",
    "Francium",
    "Radium",
    "Actinium",
    "Thorium",
    "Protactinium",
    "Uranium",
    "Neptunium",
    "Plutonium",
    "Americium",
    "Curium",
    "Berkelium",
    "Californium",
    "Einsteinium",
    "Fermium",
    "Mendelevium",
    "Nobelium",
    "Lawrencium",
    "Rutherfordium",
    "Dubnium",
    "Seaborgium",
    "Bohrium",
    "Hassium",
    "Meitnerium",
    "Darmstadtium",
    "Roentgenium",
    "Copernicium",
    "Ununtrium",
    "Flerovium",
    "Ununpentium",
    "Livermorium",
    "Ununseptium",
    "Ununoctium"
};

const std::vector<float> MaterialUtils::atomicMass {
    1.008,          // H
    4.0026022,      // He
    6.94,           // Li
    9.01218315,     // Be
    10.81,          // B
    12.011,         // C
    14.007,         // N
    15.999,         // O
    18.9984031636,  // F
    20.17976,       // Ne
    22.989769282,   // Na
    24.305,         // Mg
    26.98153857,    // Al
    28.085,         // Si
    30.9737619985,  // P
    32.06,          // S
    35.45,          // Cl
    39.9481,        // Ar
    39.09831,       // K
    40.0784,        // Ca
    44.9559085,     // Sc
    47.8671,        // Ti
    50.94151,       // V
    51.99616,       // Cr
    54.9380443,     // Mn
    55.8452,        // Fe
    58.9331944,     // Co
    58.69344,       // Ni
    63.5463,        // Cu
    65.382,         // Zn
    69.7231,        // Ga
    72.6308,        // Ge
    74.9215956,     // As
    78.9718,        // Se
    79.904,         // Br
    83.7982,        // Kr
    85.46783,       // Rb
    87.621,         // Sr
    88.905842,      // Y
    91.2242,        // Zr
    92.906372,      // Nb
    95.951,         // Mo
    97,             // Tc
    101.072,        // Ru
    102.905502,     // Rh
    106.421,        // Pd
    107.86822,      // Ag
    112.4144,       // Cd
    114.8181,       // In
    118.7107,       // Sn
    121.7601,       // Sb
    127.603,        // Te
    126.904473,     // I
    131.2936,       // Xe
    132.905451966,  // Cs
    137.3277,       // Ba
    138.905477,     // La
    140.1161,       // Ce
    140.907662,     //Pr
    144.2423,       // Nd
    145,            // Pm
    150.362,        // Sm
    151.9641,       // Eu
    157.253,        // Gd
    158.925352,     // Tb
    162.5001,       // Dy
    164.930332,     // Ho
    167.2593,       // Er
    168.934222,     // Tm
    173.04510,      // Yb
    174.96681,      // Lu
    178.492,        // Hf
    180.947882,     // Ta
    183.841,        // W
    186.2071,       // Re
    190.233,        // Os
    192.2173,       // Ir
    195.0849,       // Pt
    196.9665695,    // Au
    200.5923,       // Hg
    204.38,         // Tl
    207.21,         // Pb
    208.980401,     // Bi
    209,            // Po
    210,            // At
    222,            // Rn
    223,            // Fr
    226,            // Ra
    227,            // Ac
    232.03774,      // Th
    231.035882,     // Pa
    238.028913,     // U
    237,            // Np
    244,            // Pu
    243,            // Am
    247,            // Cm
    247,            // Bk
    251,            // Cf
    252,            // Es
    257,            // Fm
    258,            // Md
    259,            // No
    262,            // Lr
    267,            // Rf
    270,            // Db
    269,            // Sg
    270,            // Bh
    270,            // Hs
    278,            // Mt
    281,            // Ds
    281,            // Rg
    285,            // Cn
    286,            // Nh
    289,            // Fl
    289,            // Mc
    293,            // Lv
    293,            // Ts
    294             // Og
};

const std::vector<std::vector<int> > MaterialUtils::naturalIsotopes {
    std::vector<int>{1, 2},           // H
    std::vector<int>{3, 4},           // He
    std::vector<int>{6, 7},           // Li
    std::vector<int>{9},              // Be
    std::vector<int>{10, 11},         // B
    std::vector<int>{12, 13},         // C
    std::vector<int>{14, 15},         // N
    std::vector<int>{16, 17, 18},     // O
    std::vector<int>{19},             // F
    std::vector<int>{20, 21, 22},     // Ne
    std::vector<int>{23},             // Na
    std::vector<int>{24, 25, 26},     // Mg
    std::vector<int>{27},             // Al
    std::vector<int>{28, 29, 30},     // Si
    std::vector<int>{31},             // P
    std::vector<int>{32, 33, 34, 36},     // S
    std::vector<int>{35, 37},             // Cl
    std::vector<int>{36, 38, 40},         // Ar
    std::vector<int>{39, 40, 41},         // K
    std::vector<int>{40, 42, 43, 44, 46, 48},     // Ca
    std::vector<int>{45},                     // Sc
    std::vector<int>{46, 47, 48, 49, 50},     // Ti
    std::vector<int>{50, 51},                 // V
    std::vector<int>{50, 52, 53, 54},         // Cr
    std::vector<int>{55},                     // Mn
    std::vector<int>{54, 56, 57, 58},         // Fe
    std::vector<int>{59},                     // Co
    std::vector<int>{58, 60, 61, 62, 64},     // Ni
    std::vector<int>{63, 65},                 // Cu
    std::vector<int>{64, 66, 67, 68, 70},     // Zn
    std::vector<int>{69, 71},                 // Ga
    std::vector<int>{70, 72, 73, 74, 76},     // Ge
    std::vector<int>{75},                     // As
    std::vector<int>{74, 76, 77, 78, 80, 82},     // Se
    std::vector<int>{79, 81},                     // Br
    std::vector<int>{78, 80, 82, 83, 84, 86},     // Kr
    std::vector<int>{85, 87},                 // Rb
    std::vector<int>{84, 86, 87, 88},         // Sr
    std::vector<int>{89},                     // Y
    std::vector<int>{90, 91, 92, 94, 96},     // Zr
    std::vector<int>{93},                     // Nb
    std::vector<int>{92, 94, 95, 96, 97, 98, 100},     // Mo
    std::vector<int>{},                                // Tc
    std::vector<int>{96, 98, 99, 100, 101, 102, 104},     // Ru
    std::vector<int>{103},                                // Rh
    std::vector<int>{102, 104, 105, 106, 108, 110},       // Pd
    std::vector<int>{107, 109},                                   // Ag
    std::vector<int>{106, 108, 110, 111, 112, 113, 114, 116},     // Cd
    std::vector<int>{113, 115},                                   // In
    std::vector<int>{112, 114, 115, 116, 117, 118, 119, 120, 122, 124},     // Sn
    std::vector<int>{121, 123},                                   // Sb
    std::vector<int>{120, 122, 123, 124, 125, 126, 128, 130},     // Te
    std::vector<int>{127},                                        // I
    std::vector<int>{124, 126, 128, 129, 130, 131, 132, 134, 136},     // Xe
    std::vector<int>{133},                                       // Cs
    std::vector<int>{130, 132, 134, 135, 136, 137, 138},     // Ba
    std::vector<int>{138, 139},                      // La
    std::vector<int>{136, 138, 140, 142},            // Ce
    std::vector<int>{141},                           //Pr
    std::vector<int>{142, 143, 144, 145, 146, 148, 150},     // Nd
    std::vector<int>{},                                      // Pm
    std::vector<int>{144, 147, 148, 149, 150, 152, 154},     // Sm
    std::vector<int>{151, 153},                           // Eu
    std::vector<int>{152, 154, 155, 156, 157, 158, 160},     // Gd
    std::vector<int>{159},                                  // Tb
    std::vector<int>{156, 158, 160, 161, 162, 163, 164},     // Dy
    std::vector<int>{165},                                    // Ho
    std::vector<int>{162, 164, 166, 167, 168, 170},     // Er
    std::vector<int>{169},                                     // Tm
    std::vector<int>{168, 170, 171, 172, 173, 174, 176},     // Yb
    std::vector<int>{175, 176},                                    // Lu
    std::vector<int>{174, 176, 177, 178, 179, 180},     // Hf
    std::vector<int>{180, 181},                        // Ta
    std::vector<int>{180, 182, 183, 184, 186},     // W
    std::vector<int>{185, 187},                    // Re
    std::vector<int>{184, 186, 187, 188, 189, 190, 192},     // Os
    std::vector<int>{191, 193},                           // Ir
    std::vector<int>{190, 192, 194, 195, 196, 198},     // Pt
    std::vector<int>{197},                                // Au
    std::vector<int>{196, 198, 199, 200, 201, 202, 204},     // Hg
    std::vector<int>{203, 205},                        // Tl
    std::vector<int>{204, 206, 207, 208},     // Pb
    std::vector<int>{209},            // Bi
    std::vector<int>{},     // Po
    std::vector<int>{},     // At
    std::vector<int>{},     // Rn
    std::vector<int>{},     // Fr
    std::vector<int>{},     // Ra
    std::vector<int>{},     // Ac
    std::vector<int>{232},     // Th
    std::vector<int>{},     // Pa
    std::vector<int>{234, 235, 238},     // U
    std::vector<int>{},     // Np
    std::vector<int>{},     // Pu
    std::vector<int>{},     // Am
    std::vector<int>{},     // Cm
    std::vector<int>{},     // Bk
    std::vector<int>{},     // Cf
    std::vector<int>{},     // Es
    std::vector<int>{},     // Fm
    std::vector<int>{},     // Md
    std::vector<int>{},     // No
    std::vector<int>{},     // Lr
    std::vector<int>{},     // Rf
    std::vector<int>{},     // Db
    std::vector<int>{},     // Sg
    std::vector<int>{},     // Bh
    std::vector<int>{},     // Hs
    std::vector<int>{},     // Mt
    std::vector<int>{},     // Ds
    std::vector<int>{},     // Rg
    std::vector<int>{},     // Cn
    std::vector<int>{},     // Nh
    std::vector<int>{},     // Fl
    std::vector<int>{},     // Mc
    std::vector<int>{},     // Lv
    std::vector<int>{},     // Ts
    std::vector<int>{},     // Og
};

const std::vector<std::vector<float> > MaterialUtils::naturalAbundances {
    std::vector<float>{0.999885, 0.000115},           // H
    std::vector<float>{0.00000137, 0.99999863},       // He
    std::vector<float>{0.0759, 0.9241},               // Li
    std::vector<float>{1.0},                          // Be
    std::vector<float>{0.199, 0.801},                 // B
    std::vector<float>{0.9893, 0.0107},               // C
    std::vector<float>{0.99632, 0.00368},             // N
    std::vector<float>{0.99757, 0.00038, 0.00205},    // O
    std::vector<float>{1.0},                         // F
    std::vector<float>{0.9048, 0.0027, 0.0925},     // Ne
    std::vector<float>{1.0},                         // Na
    std::vector<float>{0.7899, 0.1000, 0.1101},     // Mg
    std::vector<float>{1.0},                            // Al
    std::vector<float>{0.922296, 0.046832, 0.030872},     // Si
    std::vector<float>{1.0},                           // P
    std::vector<float>{0.9493, 0.0076, 0.0429, 0.0002},     // S
    std::vector<float>{0.7578, 0.2422},                     // Cl
    std::vector<float>{0.003365, 0.000632, 0.996003},         // Ar
    std::vector<float>{0.932581, 0.000117, 0.067302},         // K
    std::vector<float>{0.96941, 0.00647, 0.00135, 0.02086, 0.00004, 0.00187},     // Ca
    std::vector<float>{1.0},                                        // Sc
    std::vector<float>{0.0825, 0.0744, 0.7372, 0.0541, 0.0518},     // Ti
    std::vector<float>{0.00250, 0.99750},                                // V
    std::vector<float>{0.04345, 0.83789, 0.09501, 0.02365},         // Cr
    std::vector<float>{1.0},                                       // Mn
    std::vector<float>{0.05845, 0.91754, 0.02119, 0.00282},         // Fe
    std::vector<float>{1.0},                                          // Co
    std::vector<float>{0.680769, 0.262231, 0.011399, 0.036345, 0.009256},     // Ni
    std::vector<float>{0.6917, 0.3083},                             // Cu
    std::vector<float>{0.4863, 0.2790, 0.0410, 0.1875, 0.0062},     // Zn
    std::vector<float>{0.60108, 0.39892},                           // Ga
    std::vector<float>{0.2084, 0.2754, 0.0773, 0.3628, 0.0761},     // Ge
    std::vector<float>{1.0},                                         // As
    std::vector<float>{0.0089, 0.0937, 0.0763, 0.2377, 0.4961, 0.0873},     // Se
    std::vector<float>{0.5069, 0.4931},                                     // Br
    std::vector<float>{0.0035, 0.0228, 0.1158, 0.1149, 0.5700, 0.1730},     // Kr
    std::vector<float>{0.7217, 0.2783},                        // Rb
    std::vector<float>{0.0056, 0.0986, 0.0700, 0.8258},         // Sr
    std::vector<float>{1.0},                                       // Y
    std::vector<float>{0.5145, 0.1122, 0.1715, 0.1738, 0.0280},     // Zr
    std::vector<float>{1.0},                                        // Nb
    std::vector<float>{0.1484, 0.0925, 0.1592, 0.1668, 0.0955, .2413, 0.0963},     // Mo
    std::vector<float>{},                                                          // Tc
    std::vector<float>{0.0554, 0.0187, 0.1276, 0.1260, 0.1706, 0.3155, 0.1862},     // Ru
    std::vector<float>{1.0},                                                        // Rh
    std::vector<float>{0.0102, 0.1114, 0.2233, 0.2733, 0.2646, 0.1172},             // Pd
    std::vector<float>{0.51839, 0.48161},                                           // Ag
    std::vector<float>{0.0125, 0.0089, 0.1249, 0.1280, 0.2413, 0.1222, 0.2873, 0.0749},     // Cd
    std::vector<float>{0.0429, 0.9571},                                                        // In
    std::vector<float>{0.0097, 0.0066, 0.0034, 0.1454, 0.0768, 0.2422, 0.0859, 0.3258, 0.0463, 0.0579},     // Sn
    std::vector<float>{0.5721, 0.4279},                                                     // Sb
    std::vector<float>{0.0009, 0.0255, 0.0089, 0.0474, 0.0707, 0.1884, 0.3174, 0.3408},     // Te
    std::vector<float>{1.0},                                                                 // I
    std::vector<float>{0.0009, 0.0009, 0.0192, 0.2644, 0.0408, 0.2118, 0.2689, 0.1044, 0.0887},     // Xe
    std::vector<float>{1.0},                                                                      // Cs
    std::vector<float>{0.00106, 0.00101, 0.02417, 0.06592, 0.07854, 0.11232, 0.71698},     // Ba
    std::vector<float>{0.00090, 0.99910},                              // La
    std::vector<float>{0.00185, 0.00251, 0.88450, 0.11114},            // Ce
    std::vector<float>{1.0},                                               //Pr
    std::vector<float>{0.272, 0.122, 0.238, 0.083, 0.172, 0.057, 0.056},     // Nd
    std::vector<float>{},                                                       // Pm
    std::vector<float>{0.0307, 0.1499, 0.1124, 0.1382, 0.0738, 0.2675, 0.2275},     // Sm
    std::vector<float>{0.4781, 0.5219},                                               // Eu
    std::vector<float>{0.0020, 0.0218, 0.1480, 0.2047, 0.1565, 0.2484, 0.2186},     // Gd
    std::vector<float>{1.0},                                                        // Tb
    std::vector<float>{0.0006, 0.0010, 0.0234, 0.1891, 0.2551, 0.2490, 0.2818},     // Dy
    std::vector<float>{1.0},                                                       // Ho
    std::vector<float>{0.0014, 0.0161, 0.3361, 0.2293, 0.2678, 0.1493},              // Er
    std::vector<float>{1.0},                                                         // Tm
    std::vector<float>{0.0013, 0.0304, 0.1428, 0.2183, 0.1613, 0.3183, 0.1276},     // Yb
    std::vector<float>{0.9741, 0.0259},                                             // Lu
    std::vector<float>{0.0016, 0.0526, 0.1860, 0.2728, 0.1362, 0.3508},              // Hf
    std::vector<float>{0.00012, 0.99988},                                            // Ta
    std::vector<float>{0.0012, 0.2650, 0.1431, 0.3064, 0.2843},                        // W
    std::vector<float>{0.3740, 0.6260},                                                 // Re
    std::vector<float>{0.0002, 0.0159, 0.0196, 0.1324, 0.1615, 0.2626, 0.4078},     // Os
    std::vector<float>{0.373, 0.627},                                                  // Ir
    std::vector<float>{0.00014, 0.00782, 0.32967, 0.33832, 0.25242, 0.07163},     // Pt
    std::vector<float>{1.0},                                                         // Au
    std::vector<float>{0.0015, 0.0997, 0.1687, 0.2310, 0.1318, 0.2986, 0.0687},     // Hg
    std::vector<float>{0.29524, 0.70476},                                             // Tl
    std::vector<float>{0.014, 0.241, 0.221, 0.524},                                // Pb
    std::vector<float>{1.0},                                                         // Bi
    std::vector<float>{},        // Po
    std::vector<float>{},        // At
    std::vector<float>{},        // Rn
    std::vector<float>{},        // Fr
    std::vector<float>{},        // Ra
    std::vector<float>{},        // Ac
    std::vector<float>{1.0},     // Th
    std::vector<float>{},        // Pa
    std::vector<float>{0.000055, 0.007200, 0.992745},     // U
    std::vector<float>{},     // Np
    std::vector<float>{},     // Pu
    std::vector<float>{},     // Am
    std::vector<float>{},     // Cm
    std::vector<float>{},     // Bk
    std::vector<float>{},     // Cf
    std::vector<float>{},     // Es
    std::vector<float>{},     // Fm
    std::vector<float>{},     // Md
    std::vector<float>{},     // No
    std::vector<float>{},     // Lr
    std::vector<float>{},     // Rf
    std::vector<float>{},     // Db
    std::vector<float>{},     // Sg
    std::vector<float>{},     // Bh
    std::vector<float>{},     // Hs
    std::vector<float>{},     // Mt
    std::vector<float>{},     // Ds
    std::vector<float>{},     // Rg
    std::vector<float>{},     // Cn
    std::vector<float>{},     // Nh
    std::vector<float>{},     // Fl
    std::vector<float>{},     // Mc
    std::vector<float>{},     // Lv
    std::vector<float>{},     // Ts
    std::vector<float>{},     // Og
};

const std::vector<int> MaterialUtils::hounsfieldRangePhantom19{
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

const std::vector<int> MaterialUtils::hounsfieldRangePhantom19Elements{
    1,     6,     7,     8,     11,    12,    15,    16,    17,    18,    19,    20
};

const std::vector<std::vector<float> > MaterialUtils::hounsfieldRangePhantom19Weights{
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

MaterialUtils::MaterialUtils()
{

}

MaterialUtils::~MaterialUtils()
{

}

std::vector<float> MaterialUtils::weightFracToAtomFrac(std::vector<int> elements, std::vector<float> weights)
{
    std::vector<float> atomFrac;

    if(elements.size() != weights.size())
    {
        qDebug() << "MaterialUtils::weightFracToAtomFrac(): 554: vector size mismatch!";
        return atomFrac;
    }

    float totalWeight = 0.0f;
    for(unsigned int i = 0; i < elements.size(); i++)
    {
        atomFrac.push_back(weights[i]/MaterialUtils::atomicMass[elements[i]]);
        totalWeight += atomFrac[i];
    }

    for(unsigned int i = 0; i < atomFrac.size(); i++)
        atomFrac[i] /= totalWeight;

    return atomFrac;
}

float MaterialUtils::atomsPerGram(std::vector<int> elements, std::vector<float> atomFractions)
{
    float s = 0.0f;
    for(int i = 0; i < atomFractions.size(); i++)
        s += atomFractions[i];
    if(abs(s - 1.0) > 1E-6)
    {
        qDebug() << "MaterialUtils::atomsPerGram(): 578: atom fractions didn't add to 1.0, they added to " << s;
    }

    float gpm = 0.0f;

    for(int i = 0; i < atomFractions.size(); i++)
    {
        gpm += MaterialUtils::atomicMass[elements[i]] * atomFractions[i];  // grams per mol
    }

    float gpa = gpm / MaterialUtils::AVOGADRO;  // grams per atom

    return 1.0f/gpa;  // atoms per gram
}

bool MaterialUtils::validate()
{
    for(unsigned int i = 1; i < MaterialUtils::atomicMass.size(); i++)
        if(MaterialUtils::atomicMass[i] < MaterialUtils::atomicMass[i-1])
        {
            std::cout << "Warning: atomic mass check failed at element " << i << std::endl;
            //return false;
        }

    if(MaterialUtils::elementNames.size() != MaterialUtils::naturalAbundances.size())
    {
        std::cout << "element name vector and abundance vectors are of unequal size!" << std::endl;
        return false;
    }

    if(MaterialUtils::atomicMass.size() != MaterialUtils::naturalAbundances.size())
    {
        std::cout << "atomic mass vector and abundance vectors are of unequal size!" << std::endl;
        return false;
    }

    if(MaterialUtils::naturalAbundances.size() != MaterialUtils::naturalIsotopes.size())
    {
        std::cout << "abundance and isotope vectors are of unequal size!" << std::endl;
        return false;
    }

    // Verify a matching number of isotopes for each element
    for(unsigned int i = 0; i < MaterialUtils::atomicMass.size(); i++)
        if(MaterialUtils::naturalIsotopes[i].size() != MaterialUtils::naturalAbundances[i].size())
        {
            std::cout << "For element " << i << " the number of isotopes does not match the number of abundances" << std::endl;
            return false;
        }

    // Verify that the abundances sum to one
    for(unsigned int i = 0; i < MaterialUtils::atomicMass.size(); i++)
    {
        if(MaterialUtils::naturalIsotopes[i].size() != 0)
        {
            float s = 0.0;
            for(unsigned int j = 0; j < MaterialUtils::naturalIsotopes[i].size(); j++)
                s += MaterialUtils::naturalAbundances[i][j];
            float t = s - 1.0;
            float tol = 1E-6;
            if(t > tol || t < -tol)
            {
                std::cout << "Non unity sum at isotope " << i << std::endl;
                std::cout << "Difference: " << t << std::endl;
                return false;
            }
        }
    }

    return true;

}
