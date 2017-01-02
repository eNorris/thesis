#ifndef GLOBALS_H
#define GLOBALS_H

#ifdef __linux__
#define U16_T u_int16_t
#elif _WIN32
#define U16_T ushort
#else
#undef U16_T
#endif

#endif // GLOBALS_H
