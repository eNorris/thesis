#include "materialutils.h"

#include <iostream>
#include <QDebug>

const float MaterialUtils::AVOGADRO = 6.0221409E23f;

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
    1.008f,          // H
    4.0026022f,      // He
    6.94f,           // Li
    9.01218315f,     // Be
    10.81f,          // B
    12.011f,         // C
    14.007f,         // N
    15.999f,         // O
    18.9984031636f,  // F
    20.17976f,       // Ne
    22.989769282f,   // Na
    24.305f,         // Mg
    26.98153857f,    // Al
    28.085f,         // Si
    30.9737619985f,  // P
    32.06f,          // S
    35.45f,          // Cl
    39.9481f,        // Ar
    39.09831f,       // K
    40.0784f,        // Ca
    44.9559085f,     // Sc
    47.8671f,        // Ti
    50.94151f,       // V
    51.99616f,       // Cr
    54.9380443f,     // Mn
    55.8452f,        // Fe
    58.9331944f,     // Co
    58.69344f,       // Ni
    63.5463f,        // Cu
    65.382f,         // Zn
    69.7231f,        // Ga
    72.6308f,        // Ge
    74.9215956f,     // As
    78.9718f,        // Se
    79.904f,         // Br
    83.7982f,        // Kr
    85.46783f,       // Rb
    87.621f,         // Sr
    88.905842f,      // Y
    91.2242f,        // Zr
    92.906372f,      // Nb
    95.951f,         // Mo
    97.0f,             // Tc
    101.072f,        // Ru
    102.905502f,     // Rh
    106.421f,        // Pd
    107.86822f,      // Ag
    112.4144f,       // Cd
    114.8181f,       // In
    118.7107f,       // Sn
    121.7601f,       // Sb
    127.603f,        // Te
    126.904473f,     // I
    131.2936f,       // Xe
    132.905451966f,  // Cs
    137.3277f,       // Ba
    138.905477f,     // La
    140.1161f,       // Ce
    140.907662f,     //Pr
    144.2423f,       // Nd
    145.0f,            // Pm
    150.362f,        // Sm
    151.9641f,       // Eu
    157.253f,        // Gd
    158.925352f,     // Tb
    162.5001f,       // Dy
    164.930332f,     // Ho
    167.2593f,       // Er
    168.934222f,     // Tm
    173.04510f,      // Yb
    174.96681f,      // Lu
    178.492f,        // Hf
    180.947882f,     // Ta
    183.841f,        // W
    186.2071f,       // Re
    190.233f,        // Os
    192.2173f,       // Ir
    195.0849f,       // Pt
    196.9665695f,    // Au
    200.5923f,       // Hg
    204.38f,         // Tl
    207.21f,         // Pb
    208.980401f,     // Bi
    209.0f,            // Po
    210.0f,            // At
    222.0f,            // Rn
    223.0f,            // Fr
    226.0f,            // Ra
    227.0f,            // Ac
    232.03774f,      // Th
    231.035882f,     // Pa
    238.028913f,     // U
    237.0f,            // Np
    244.0f,            // Pu
    243.0f,            // Am
    247.0f,            // Cm
    247.0f,            // Bk
    251.0f,            // Cf
    252.0f,            // Es
    257.0f,            // Fm
    258.0f,            // Md
    259.0f,            // No
    262.0f,            // Lr
    267.0f,            // Rf
    270.0f,            // Db
    269.0f,            // Sg
    270.0f,            // Bh
    270.0f,            // Hs
    278.0f,            // Mt
    281.0f,            // Ds
    281.0f,            // Rg
    285.0f,            // Cn
    286.0f,            // Nh
    289.0f,            // Fl
    289.0f,            // Mc
    293.0f,            // Lv
    293.0f,            // Ts
    294.0f             // Og
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
    std::vector<float>{0.999885f, 0.000115f},           // H
    std::vector<float>{0.00000137f, 0.99999863f},       // He
    std::vector<float>{0.0759f, 0.9241f},               // Li
    std::vector<float>{1.0f},                          // Be
    std::vector<float>{0.199f, 0.801f},                 // B
    std::vector<float>{0.9893f, 0.0107f},               // C
    std::vector<float>{0.99632f, 0.00368f},             // N
    std::vector<float>{0.99757f, 0.00038f, 0.00205f},    // O
    std::vector<float>{1.0f},                         // F
    std::vector<float>{0.9048f, 0.0027f, 0.0925f},     // Ne
    std::vector<float>{1.0f},                         // Na
    std::vector<float>{0.7899f, 0.1000f, 0.1101f},     // Mg
    std::vector<float>{1.0f},                            // Al
    std::vector<float>{0.922296f, 0.046832f, 0.030872f},     // Si
    std::vector<float>{1.0f},                           // P
    std::vector<float>{0.9493f, 0.0076f, 0.0429f, 0.0002f},     // S
    std::vector<float>{0.7578f, 0.2422f},                     // Cl
    std::vector<float>{0.003365f, 0.000632f, 0.996003f},         // Ar
    std::vector<float>{0.932581f, 0.000117f, 0.067302f},         // K
    std::vector<float>{0.96941f, 0.00647f, 0.00135f, 0.02086f, 0.00004f, 0.00187f},     // Ca
    std::vector<float>{1.0f},                                        // Sc
    std::vector<float>{0.0825f, 0.0744f, 0.7372f, 0.0541f, 0.0518f},     // Ti
    std::vector<float>{0.00250f, 0.99750f},                                // V
    std::vector<float>{0.04345f, 0.83789f, 0.09501f, 0.02365f},         // Cr
    std::vector<float>{1.0f},                                       // Mn
    std::vector<float>{0.05845f, 0.91754f, 0.02119f, 0.00282f},         // Fe
    std::vector<float>{1.0f},                                          // Co
    std::vector<float>{0.680769f, 0.262231f, 0.011399f, 0.036345f, 0.009256f},     // Ni
    std::vector<float>{0.6917f, 0.3083f},                             // Cu
    std::vector<float>{0.4863f, 0.2790f, 0.0410f, 0.1875f, 0.0062f},     // Zn
    std::vector<float>{0.60108f, 0.39892f},                           // Ga
    std::vector<float>{0.2084f, 0.2754f, 0.0773f, 0.3628f, 0.0761f},     // Ge
    std::vector<float>{1.0f},                                         // As
    std::vector<float>{0.0089f, 0.0937f, 0.0763f, 0.2377f, 0.4961f, 0.0873f},     // Se
    std::vector<float>{0.5069f, 0.4931f},                                     // Br
    std::vector<float>{0.0035f, 0.0228f, 0.1158f, 0.1149f, 0.5700f, 0.1730f},     // Kr
    std::vector<float>{0.7217f, 0.2783f},                        // Rb
    std::vector<float>{0.0056f, 0.0986f, 0.0700f, 0.8258f},         // Sr
    std::vector<float>{1.0f},                                       // Y
    std::vector<float>{0.5145f, 0.1122f, 0.1715f, 0.1738f, 0.0280f},     // Zr
    std::vector<float>{1.0f},                                        // Nb
    std::vector<float>{0.1484f, 0.0925f, 0.1592f, 0.1668f, 0.0955f, .2413f, 0.0963f},     // Mo
    std::vector<float>{},                                                          // Tc
    std::vector<float>{0.0554f, 0.0187f, 0.1276f, 0.1260f, 0.1706f, 0.3155f, 0.1862f},     // Ru
    std::vector<float>{1.0f},                                                        // Rh
    std::vector<float>{0.0102f, 0.1114f, 0.2233f, 0.2733f, 0.2646f, 0.1172f},             // Pd
    std::vector<float>{0.51839f, 0.48161f},                                           // Ag
    std::vector<float>{0.0125f, 0.0089f, 0.1249f, 0.1280f, 0.2413f, 0.1222f, 0.2873f, 0.0749f},     // Cd
    std::vector<float>{0.0429f, 0.9571f},                                                        // In
    std::vector<float>{0.0097f, 0.0066f, 0.0034f, 0.1454f, 0.0768f, 0.2422f, 0.0859f, 0.3258f, 0.0463f, 0.0579f},     // Sn
    std::vector<float>{0.5721f, 0.4279f},                                                     // Sb
    std::vector<float>{0.0009f, 0.0255f, 0.0089f, 0.0474f, 0.0707f, 0.1884f, 0.3174f, 0.3408f},     // Te
    std::vector<float>{1.0f},                                                                 // I
    std::vector<float>{0.0009f, 0.0009f, 0.0192f, 0.2644f, 0.0408f, 0.2118f, 0.2689f, 0.1044f, 0.0887f},     // Xe
    std::vector<float>{1.0f},                                                                      // Cs
    std::vector<float>{0.00106f, 0.00101f, 0.02417f, 0.06592f, 0.07854f, 0.11232f, 0.71698f},     // Ba
    std::vector<float>{0.00090f, 0.99910f},                              // La
    std::vector<float>{0.00185f, 0.00251f, 0.88450f, 0.11114f},            // Ce
    std::vector<float>{1.0f},                                               //Pr
    std::vector<float>{0.272f, 0.122f, 0.238f, 0.083f, 0.172f, 0.057f, 0.056f},     // Nd
    std::vector<float>{},                                                       // Pm
    std::vector<float>{0.0307f, 0.1499f, 0.1124f, 0.1382f, 0.0738f, 0.2675f, 0.2275f},     // Sm
    std::vector<float>{0.4781f, 0.5219f},                                               // Eu
    std::vector<float>{0.0020f, 0.0218f, 0.1480f, 0.2047f, 0.1565f, 0.2484f, 0.2186f},     // Gd
    std::vector<float>{1.0f},                                                        // Tb
    std::vector<float>{0.0006f, 0.0010f, 0.0234f, 0.1891f, 0.2551f, 0.2490f, 0.2818f},     // Dy
    std::vector<float>{1.0f},                                                       // Ho
    std::vector<float>{0.0014f, 0.0161f, 0.3361f, 0.2293f, 0.2678f, 0.1493f},              // Er
    std::vector<float>{1.0f},                                                         // Tm
    std::vector<float>{0.0013f, 0.0304f, 0.1428f, 0.2183f, 0.1613f, 0.3183f, 0.1276f},     // Yb
    std::vector<float>{0.9741f, 0.0259f},                                             // Lu
    std::vector<float>{0.0016f, 0.0526f, 0.1860f, 0.2728f, 0.1362f, 0.3508f},              // Hf
    std::vector<float>{0.00012f, 0.99988f},                                            // Ta
    std::vector<float>{0.0012f, 0.2650f, 0.1431f, 0.3064f, 0.2843f},                        // W
    std::vector<float>{0.3740f, 0.6260f},                                                 // Re
    std::vector<float>{0.0002f, 0.0159f, 0.0196f, 0.1324f, 0.1615f, 0.2626f, 0.4078f},     // Os
    std::vector<float>{0.373f, 0.627f},                                                  // Ir
    std::vector<float>{0.00014f, 0.00782f, 0.32967f, 0.33832f, 0.25242f, 0.07163f},     // Pt
    std::vector<float>{1.0f},                                                         // Au
    std::vector<float>{0.0015f, 0.0997f, 0.1687f, 0.2310f, 0.1318f, 0.2986f, 0.0687f},     // Hg
    std::vector<float>{0.29524f, 0.70476f},                                             // Tl
    std::vector<float>{0.014f, 0.241f, 0.221f, 0.524f},                                // Pb
    std::vector<float>{1.0f},                                                         // Bi
    std::vector<float>{},        // Po
    std::vector<float>{},        // At
    std::vector<float>{},        // Rn
    std::vector<float>{},        // Fr
    std::vector<float>{},        // Ra
    std::vector<float>{},        // Ac
    std::vector<float>{1.0f},     // Th
    std::vector<float>{},        // Pa
    std::vector<float>{0.000055f, 0.007200f, 0.992745f},     // U
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
    65000,  // 20 (Al gantry)
    1000000,  // 21
};

const std::vector<int> MaterialUtils::hounsfieldRangePhantom19Elements{
    1,     // Hydrogen
    6,     // Carbon
    7,     // Nitrogen
    8,     // Oxygen
    11,    // Sodium
    12,    // Magnesium
    15,    // Phosphorus
    16,    // Sulfur
    17,    // Chlorine
    18,    // Argon
    19,    // Potassium
    20,    // Calcium
    13     // Aluminum (for gantry)
};

const std::vector<std::vector<float> > MaterialUtils::hounsfieldRangePhantom19Weights{
    std::vector<float> {0.000f, 0.000f, 0.757f, 0.232f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.013f, 0.000f, 0.000f, 0.000f}, // Air 1
    std::vector<float> {0.103f, 0.105f, 0.031f, 0.749f, 0.002f, 0.000f, 0.002f, 0.003f, 0.003f, 0.000f, 0.002f, 0.000f, 0.000f}, // Lung 2
    std::vector<float> {0.112f, 0.508f, 0.012f, 0.364f, 0.001f, 0.000f, 0.000f, 0.001f, 0.001f, 0.000f, 0.000f, 0.000f, 0.000f}, // Adipose/adrenal 3
    std::vector<float> {0.100f, 0.163f, 0.043f, 0.684f, 0.004f, 0.000f, 0.000f, 0.004f, 0.003f, 0.000f, 0.000f, 0.000f, 0.000f}, // Small intestine 4
    std::vector<float> {0.097f, 0.447f, 0.025f, 0.359f, 0.000f, 0.000f, 0.023f, 0.002f, 0.001f, 0.000f, 0.001f, 0.045f, 0.000f}, // Bone 5
    std::vector<float> {0.091f, 0.414f, 0.027f, 0.368f, 0.000f, 0.001f, 0.032f, 0.002f, 0.001f, 0.000f, 0.001f, 0.063f, 0.000f}, // Bone 6
    std::vector<float> {0.085f, 0.378f, 0.029f, 0.379f, 0.000f, 0.001f, 0.041f, 0.002f, 0.001f, 0.000f, 0.001f, 0.082f, 0.000f}, // Bone 7
    std::vector<float> {0.080f, 0.345f, 0.031f, 0.388f, 0.000f, 0.001f, 0.050f, 0.002f, 0.001f, 0.000f, 0.001f, 0.010f, 0.000f}, // Bone 8
    std::vector<float> {0.075f, 0.316f, 0.032f, 0.397f, 0.000f, 0.001f, 0.058f, 0.002f, 0.001f, 0.000f, 0.000f, 0.116f, 0.000f}, // Bone 9
    std::vector<float> {0.071f, 0.289f, 0.034f, 0.404f, 0.000f, 0.001f, 0.066f, 0.002f, 0.001f, 0.000f, 0.000f, 0.131f, 0.000f}, // Bone 10
    std::vector<float> {0.067f, 0.264f, 0.035f, 0.412f, 0.000f, 0.002f, 0.072f, 0.003f, 0.000f, 0.000f, 0.000f, 0.144f, 0.000f}, // Bone 11
    std::vector<float> {0.063f, 0.242f, 0.037f, 0.418f, 0.000f, 0.002f, 0.078f, 0.003f, 0.000f, 0.000f, 0.000f, 0.157f, 0.000f}, // Bone 12
    std::vector<float> {0.060f, 0.221f, 0.038f, 0.424f, 0.000f, 0.002f, 0.084f, 0.003f, 0.000f, 0.000f, 0.000f, 0.168f, 0.000f}, // Bone 13
    std::vector<float> {0.056f, 0.201f, 0.039f, 0.430f, 0.000f, 0.002f, 0.089f, 0.003f, 0.000f, 0.000f, 0.000f, 0.179f, 0.000f}, // Bone 14
    std::vector<float> {0.053f, 0.183f, 0.040f, 0.435f, 0.000f, 0.002f, 0.094f, 0.003f, 0.000f, 0.000f, 0.000f, 0.189f, 0.000f}, // Bone 15
    std::vector<float> {0.051f, 0.166f, 0.041f, 0.440f, 0.000f, 0.002f, 0.099f, 0.003f, 0.000f, 0.000f, 0.000f, 0.198f, 0.000f}, // Bone 16
    std::vector<float> {0.048f, 0.150f, 0.042f, 0.444f, 0.000f, 0.002f, 0.103f, 0.003f, 0.000f, 0.000f, 0.000f, 0.207f, 0.000f}, // Bone 17
    std::vector<float> {0.046f, 0.136f, 0.042f, 0.449f, 0.000f, 0.002f, 0.107f, 0.003f, 0.000f, 0.000f, 0.000f, 0.215f, 0.000f}, // Bone 18
    std::vector<float> {0.043f, 0.122f, 0.043f, 0.453f, 0.000f, 0.002f, 0.111f, 0.003f, 0.000f, 0.000f, 0.000f, 0.222f, 0.000f}, // Bone 19
    std::vector<float> {0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 1.000f}, // Aluminum gantry
    std::vector<float> {1.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f, 0.000f}  // Empty 20
};

const std::vector<int> MaterialUtils::water{
    -66,     // 1 - air
    60,      // 2 - water
    150,     // 3 - Container
    1000000, // 4 - VOID
};

const std::vector<int> MaterialUtils::waterElements{
    1,     // Hydrogen
    7,     // Nitrogen
    8,     // Oxygen
    18,    // Argon
};

const std::vector<std::vector<float> > MaterialUtils::waterWeights{
    std::vector<float> {0.000f, 0.755f, 0.232f, 0.013f}, // Air
    std::vector<float> {0.112f, 0.000f, 0.888f, 0.000f}, // Water
    std::vector<float> {0.112f, 0.000f, 0.888f, 0.000f}, // Container
    std::vector<float> {1.000f, 0.000f, 0.000f, 0.000f}  // Empty 20
};

MaterialUtils::MaterialUtils()
{

}

MaterialUtils::~MaterialUtils()
{

}

std::vector<float> MaterialUtils::weightFracToAtomFrac(std::vector<int> elementZ, std::vector<float> weights)
{
    std::vector<float> atomFrac;

    if(elementZ.size() != weights.size())
    {
        qDebug() << "MaterialUtils::weightFracToAtomFrac(): 554: vector size mismatch!";
        return atomFrac;
    }

    float totalWeight = 0.0f;
    for(unsigned int i = 0; i < elementZ.size(); i++)
    {
        atomFrac.push_back(weights[i]/MaterialUtils::atomicMass[elementZ[i]-1]);  // Subtact 1 for z to index
        totalWeight += atomFrac[i];
    }

    for(unsigned int i = 0; i < atomFrac.size(); i++)
        atomFrac[i] /= totalWeight;

    return atomFrac;
}

float MaterialUtils::atomsPerGram(std::vector<int> elementZ, std::vector<float> atomFractions)
{
    if(elementZ.size() != atomFractions.size())
    {
        qDebug() << "vector sizes don't match!";
    }

    float s = 0.0f;
    for(unsigned int i = 0; i < atomFractions.size(); i++)
        s += atomFractions[i];
    if(fabs(s - 1.0) > 1E-6)
    {
        qDebug() << "atom fractions didn't add to 1.0, they added to " << s;
    }

    float gpm = 0.0f;

    for(unsigned int i = 0; i < elementZ.size(); i++)
    {
        //float tstgpm = MaterialUtils::atomicMass[elementZ[i]-1] * atomFractions[i];
        gpm += MaterialUtils::atomicMass[elementZ[i]-1] * atomFractions[i];  // grams per mol
    }

    return MaterialUtils::AVOGADRO / gpm;
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















