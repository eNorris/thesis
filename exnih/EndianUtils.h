/*********************************************************
 * Author: Jordan P. Lefebvre
 * Description: Endianess utilities for reversing bytes
 **********************************************************/

namespace EndianUtils{

	void reverse_bytes( char * original, void * returnVal, int length );
	int reverse_int_bytes(int value);
	void reverse_array_int_bytes(int * array, int size);
	void reverse_array_float_bytes(float * array, int size);
	long reverse_long_bytes(long value);
	float reverse_float_bytes(float value);
	double reverse_double_bytes(double value);
	void reverse_array_double_bytes(double * array, int size);
    bool system_is_big_endian();
    bool file_is_big_endian(bool file_is_native_endian);
}//end namespace
