/*!
 ********************************************************
 * @author: Jordan P. Lefebvre
 * @brief: Endianess utilities for reversing bytes
 **********************************************************/
#include <stdio.h>
#include <cstdlib>
#include <stdint.h>

namespace EndianUtils {

void reverse_bytes( char * original, void * returnVal, int length )
{
    char *reversed = (char*) returnVal;
    switch(length) {
    case 2:
        reversed[0]=original[1];
        reversed[1]=original[0];
        return;
    case 4:
        reversed[0]=original[3];
        reversed[1]=original[2];
        reversed[2]=original[1];
        reversed[3]=original[0];
        return;
    case 8:
        reversed[0]=original[7];
        reversed[1]=original[6];
        reversed[2]=original[5];
        reversed[3]=original[4];
        reversed[4]=original[3];
        reversed[5]=original[2];
        reversed[6]=original[1];
        reversed[7]=original[0];
        return;
    }

}//reverse_bytes

/*!
 * Reverses the bytes of an integer
 * */
int reverse_int_bytes(int value)
{
    int returnVal;
    reverse_bytes((char*)&value, (void*)&returnVal, sizeof(int));
    return returnVal;
}//reverse_int_bytes

/*!
 * Reverses the bytes of all integers in an array
 * */
void reverse_array_int_bytes(int * array, int size)
{
    for(int i = 0; i < size; i++ ) {
        array[i] = reverse_int_bytes(array[i]);
    }
}//reverse_array_int_bytes

/*!
* Reverses the bytes of a long
 * */
long reverse_long_bytes(long value)
{
    long returnVal;
    reverse_bytes((char*)&value, (void*)&returnVal, sizeof(long));
    return returnVal;
}//reverse_long_bytes

/*!
* Reverses the bytes of a float
 * */
float reverse_float_bytes(float value)
{
    float returnVal;
    reverse_bytes((char*)&value, (void*)&returnVal, sizeof(float));
    return returnVal;
}//reverse_float_bytes

/*!
 * Reverses the bytes of all float in an array
 * */
void reverse_array_float_bytes(float * array, int size)
{
    for(int i = 0; i < size; i++ ) {
        array[i] = reverse_float_bytes(array[i]);
    }
}//reverse_array_float_bytes

/*!
* Reverses the bytes of a double
 * */
double reverse_double_bytes(double value)
{
    double returnVal;
    reverse_bytes((char*)&value, (void*)&returnVal, sizeof(double));
    return returnVal;
}//reverse_double_bytes

/*!
* Reverses the bytes of all double in an array
 * */
void reverse_array_double_bytes(double * array, int size)
{
    for(int i = 0; i < size; i++ ) {
        array[i] = reverse_double_bytes(array[i]);
    }
}//reverse_array_double_bytes

/*!
* Check if the system is big endian (if not it's little)
 * */
bool system_is_big_endian()
{
#ifdef _WIN32
    typedef unsigned __int32 uint32_t;
#endif
    union {
        uint32_t i;
        char c[4];
    } bint = {0x01020304};

    return bint.c[0] == 1;
}

/*!
* A convenience function that takes whether a file is native endian
* and adds knowledge of the system endianness to deduce file 
* endianness.
 * */
bool file_is_big_endian(bool file_is_native_endian){
    if( system_is_big_endian() ){
        return file_is_native_endian;
    } else {
        return !file_is_native_endian;
    }
}

}//end namespace
