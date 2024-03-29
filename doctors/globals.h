#ifndef GLOBALS_H
#define GLOBALS_H

// Define an unsigned 16 bit int in Win and Linux
#ifdef __linux__
#define U16_T u_int16_t
#elif _WIN32
#define U16_T uint16_t
#else
#undef U16_T
#endif

#define SOL_T double
#define RAY_T double

#endif // GLOBALS_H
