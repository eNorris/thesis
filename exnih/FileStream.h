/*
 * File:   FileStream.h
 * Author: raq
 *
 * Created on November 12, 2011, 2:00 PM
 */

#ifndef FILESTREAM_H
#define	FILESTREAM_H

#include <iostream>
#include <fstream>

//#include "Standard/Interface/AbstractStream.h"

namespace Standard {
   class SerialFactory;
   class Serializable;
}

class FileStream {
public:
    FileStream();
    FileStream(std::fstream * stream);
    virtual ~FileStream();
private:
    //Standard::SerialFactory * factory;
    std::fstream * stream;
    long file_size;
    bool ownsStream;
public:

    /**
     * @brief Open the FileStream with the given file path
     * @param const car * file - the file path from which to open this stream
     * @param std::ios_base::openmode mode - indicates how to open the fstream
     * @return bool - true, if the file was successfully opened, false otherwise
     */
    virtual bool open(const char * file, std::ios_base::openmode mode = (std::ios_base::in | std::ios_base::out | std::ios_base::binary | std::ios_base::trunc) );

    /**
     * @brief Close the FileStream
     */
    virtual void close();

    /**
     * @brief Obtain the size of this stream
     */
    virtual long getSize();

    /**
     * @brief Write the given size byte block into this SerialStream
     * Upon completion of this call, the write head will be position size bytes
     * forward, immediately following the data block just written
     * @param char * data - the memory block containing the data to be written
     * @param long size - the size of the memory block in bytes
     */
    virtual void write(const char * data, long size);
    /**
     * @brief Read into the given byte block size bytes
     * Upon complete of this call, the read head will have been moved size bytes
     * forward.
     * @param char * data - the memory block to which the data will be written
     * @param long size - the size of the memory block in bytes
     */
    virtual void read(char* data, long size);

    /**
     * @brief Obtain the universal version identifier located under the read head
     * This will not increment the read head, but provides convenience for
     * objects to conduct a look-ahead into the buffer to determine
     * what type of object was serialized, and if they are that type of object
     * they know how to inflate the given object under the read head
     * @return long - the UID of the next object in the stream, -1 if end of stream
     */
    virtual long getNextUID();

    /**
     * @brief ignores count bytes
     * This will move the read head count bytes ahead in the stream
     */
    virtual void ignore(long count);

    /**
     * @brief Obtain the position of the read head
     * @param long - the read head position
     */
    virtual long getReadHead();
    /**
     * @brief Set the position of the read head
     * @param long pos - the position from which the next read will occur
     */
    virtual void setReadHead(long pos);

    /**
     * @brief Obtain the position of the write head
     * @param long - the write head position
     */
    virtual long getWriteHead();
    /**
     * @brief Set the position of the write head
     * @param long pos - the position at which the next write will occur
     */
    virtual void setWriteHead(long pos);

    /**
     * @brief Set the object factory from which to obtain prototypes for deserialization
     * @param SerialFactory * factory - the factory containing prototypes
     */
    //virtual void setFactory(Standard::SerialFactory * factory);

    /**
     * @brief determine if next object in stream is of the given type
     * @param long uid - the universal version id of desired type
     * @return bool - true, if the next object in the stream is the given type
     *                or a child of the given type, false otherwise
     */
    virtual bool isNext(long uid);
    /**
     * @brief Deserialize the next object in the stream
     * @return Standard::Serializable * - the next object in the stream, NULL if eof encountered
     */
    //virtual Standard::Serializable * deserializeNext();

};

#endif	/* FILESTREAM_H */

