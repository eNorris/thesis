

#ifndef LIBRARYSOURCEDEFS_H
#define	LIBRARYSOURCEDEFS_H

#include <string>

namespace ScaleData {
class LibrarySourceDefs {
public:
    LibrarySourceDefs(){}
    LibrarySourceDefs(const LibrarySourceDefs& ){}
    virtual ~LibrarySourceDefs(){}
    
    static void getLibrarySourceAsString(int source, std::string & name) {
            switch (source) {
                case 0:
                case 1:
                case 4:
                    name.resize(0);
                    name.append("endf");                  
                    break;             
                case 2:
                case 3:
                    name.resize(0);
                    name.append("jeff");                    
                    break;
                case 5:
                    name.resize(0);
                    name.append("cendl");                      
                    break;
                case 6:
                    name.resize(0);
                    name.append("jendl");                       
                    break;
                case 21:
                    name.resize(0);
                    name.append("sg-33");                       
                    break;
                case 31:                  
                case 32:
                    name.resize(0);
                    name.append("indl");                      
                    break;
                case 33:
                case 37:
                    name.resize(0);
                    name.append("fendl");                      
                    break;
                case 34:
                    name.resize(0);
                    name.append("irdf");                     
                    break;
                case 35:
                case 41:
                    name.resize(0);
                    name.append("brond");                      
                    break;
                case 36:
                    name.resize(0);
                    name.append("ingdb");                       
                    break;
                default:
                    name.resize(0);
                    name.append("unkown");                         
                    break;
            }
        }
    private:

};
}
#endif	/* LIBRARYSOURCEDEFS_H */

