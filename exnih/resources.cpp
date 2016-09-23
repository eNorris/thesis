


int headerSize(){ return sizeof(int);}
int footerSize(){ return sizeof(int);}

/// Explode a magic word into its appropriate parts
/// @param const int magicWord - the magic word
/// @param int &grp - the sink grp number to be populated
/// @param int & start - the start of the data to be populated
/// @param int & end - the end of the data to be populated
/// @param int & length - the length of the data to be polulated
/// @return bool - true, if the grp value non-negative
///                false, otherwise 
bool explodeMagicWord(const int magicWord, int &grp, int &start, int &end, int &length)
{
    const int million = 1000000, thousand = 1000;
    start = magicWord / million;
    end = (magicWord-(start*million))/thousand;
    grp = (magicWord-(start*million))-(end*thousand);
    start -=1; /// decrement start due to fortran starting at index 1
    length = end - start;
    if( grp < 0 ) return false;
    return true;
}// end of explodeMagic

/// Construct a magic word from its appropriate parts
/// @param int grp - the number of groups
/// @param int start - the starting group
/// @param int end - the ending group
/// @param bool & error - error flag to indicate error occured
///                      and magicword failed to be constructed.
/// NOTE: This constructs a fortran indexed magic word
/// @return int magicWord - the constructed magic word
int getMagicWord(int grp,int start,int end, bool & error)
{
    if( grp > 999 || start > 999 || end > 999 ){
        error = true;
        return 0;
    }
    const int million = 1000000, thousand = 1000;
    start += 1; /// fortran indexing start at 1 not zero
    start *= million; /// start is first component and is in the millions place
    end *= thousand;
    
    ///i.e., start = 1, end = 2, grp = 3
    /// magicword = 001002003  = 1,002,003
    /// magic word is ssseeeggg
    return start+end+grp;    
}// end of getMagicWord
