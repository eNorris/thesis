#ifndef RESOURCE_H
#define RESOURCE_H

//#include "TableOfContents.h"
#include <QString>
//#include "Standard/Interface/Serializable.h"
///SCALE requires several types of data, depending on the running modules, to
///run. This data is represented by the Resource class. This includes nuclear
///data files, the standard composition library, fluxes and others. The Resource
///base class contains all of the operations that are common across these
///different types of data and deals mostly with management, not technical,
///activities.
class Resource //: public Standard::Serializable
{
    protected:

        ///The version number of the resource
        int version;

        ///The name of the resource
        QString resourceName;

        /// The table of contents of the resource
        //TableOfContents * tableOfContents;
	
	/// The resource's filename
	QString filename;

    public:

        ///This operation opens the resource so that it may be used to retrieve
        ///or store information according to its subtype.
        virtual int open() = 0; 

        ///This operation closes the resource and it may no longer be used.
        virtual int close() = 0; 
	
	/// Determine if this resource is currently open
	/// @return bool - true, if the resource is open, false otherwise
	virtual bool isOpen() = 0;

        ///This operation returns the title of the resource.
        virtual QString getTitle() const = 0;

        ///This operation returns the version of the resource.
        int getVersion() const {return version;}

        ///This operation returns the table of contents for the resource.
        //TableOfContents * getTableOfContents(){return tableOfContents;}
	
	/// Set this resource's name
	/// @param QString resourceName - The name of this resource        
        void setResourceName(QString resourceName){this->resourceName=resourceName;}
	
	/// Obtain the name of this resource
	/// @return QString - The name of the resource
        QString getResourceName() const {return resourceName;}
        
	/// Set the filename, absolulte or relative to this resource on disk
	/// @param QString filename - The filename of this resource
	void setFileName(QString filename){this->filename = filename;}

	/// Obtain the filename of this resource
	/// @return QString - the filename of this resource
	QString getFileName() const {return filename;}
// serialization interface
    virtual Resource * getCopy() const {throw "Unimplemented!";}
    /**
     * @brief Serialize the object into a contiguous block of data
     * @param Standard::AbstractStream * stream - the stream into which the contiguous data will be stored
     * @return int - 0 upon success, error otherwise
     */
    //virtual int serialize(Standard::AbstractStream * ) const {throw "Unimplemented!";}

    /**
     * @brief deserialize the object from the given Standard::AbstractStream 
     * @param Standard::AbstractStream * stream - the stream from which the object will be inflated
     * @return int - 0 upon success, error otherwise
     */
    //virtual int deserialize(Standard::AbstractStream * ) {throw "Unimplemented!";}
    /**
     * @brief Obtain the size in bytes of this object when serialized 
     * @return unsigned long - the size in bytes of the object when serialized
     */
    virtual unsigned long getSerializedSize() const {throw "Unimplemented!";}
    /**
     * @brief Obtain the universal version identifier
     * the uid should be unique for all Standard::Serializable objects such that 
     * an object factory can retrieve prototypes of the desired Standard::Serializable
     * object and inflate the serial object into a manageable object
     * @return long - the uid of the object
     */
    virtual long getUID() const {throw "Unimplemented!";}
    /**
     * @brief Determine if this object is a child of the given parent UID
     * This is intended to assist in object deserialization, when an object
     * may have been subclassed, the appropriate subclass prototype must be 
     * obtained from the object factory in order to correctly deserialize and
     * validate object inheritance.
     * i.e.
     * Object X contains a list of object Y. Object Y has child class object Z.
     * When deserializing object X, we expect Y, or a child of Y to be 
     * deserialized, any other objects would be an error.
     * @return bool - true, if this object is a child of a class with the given uid, false otherwise
     */
    //virtual bool childOf(long parentUID) const {return Standard::Serializable::getUID()==parentUID || Standard::Serializable::childOf(parentUID);}
public:
    /**
     * @brief the universal version identifier for this class
     */
    static const long uid;
};  // end class Resource

#endif
