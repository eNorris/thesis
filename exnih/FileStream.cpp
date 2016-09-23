/*
 * File:   FileStream.cpp
 * Author: raq
 *
 * Created on November 12, 2011, 2:00 PM
 */

//#include "Standard/Interface/Serializable.h"
//#include "Standard/Interface/SerialFactory.h"
#include "FileStream.h"
#include <QDebug>

//#include <Standard/Interface/standard_interface_config.h>

FileStream::FileStream() {
    stream = NULL;
    //factory = NULL;
    ownsStream = true;
}
FileStream::FileStream(std::fstream * stream) {
    this->stream = stream;
    ownsStream = false;
    //factory = NULL;
}


FileStream::~FileStream() {
    if( ownsStream ){
        if( stream != NULL ){
            if( stream->is_open() ){
                stream->close();
            }
            delete stream;
        }
    }
}

/**
 * @brief Open the FileStream with the given file path
 * @param const char * file - the file path from which to open this stream
 * @param std::ios_base::openmode mode - indicates how to open the fstream
 * @return bool - true, if the file was successfully opened, false otherwise
 */
bool FileStream::open(const char * file, std::ios_base::openmode mode) {
    qDebug() << "opening "<<file;
    if( file == NULL ){
        return false;
    }
    if( stream != NULL ){
        delete stream;
        stream = NULL;
    }
    stream = new std::fstream(file, mode);

    if( !stream->is_open() || stream->fail() ){
        qDebug() << "Is open "<<stream->is_open()<<", has failed "<<stream->fail();
        return false;
    }
    ownsStream = true;
    // get file size
    stream->seekg(0, std::ios::end);
    file_size = stream->tellg();
    stream->seekg(0, std::ios::beg);

    return true;
}

/**
 * @brief Close the FileStream
 */
void FileStream::close() {
    if( stream != NULL){
        if( stream->is_open() ){
            stream->close();
        }
    }
}

/**
 * @brief Obtain the size of this stream
 */
long FileStream::getSize() {
    long writeHead = stream->tellp();
    if( writeHead > file_size ) file_size = writeHead;
    return file_size;
}

/**
 * @brief Write the given size byte block into this SerialStream
 * Upon completion of this call, the write head will be position size bytes
 * forward, immediately following the data block just written
 * @param char * data - the memory block containing the data to be written
 * @param long size - the size of the memory block in bytes
 */
void FileStream::write(const char * data, long size) {
    //Require(stream != NULL);
    stream->write(data, size);
}

/**
 * @brief Read into the given byte block size bytes
 * Upon complete of this call, the read head will have been moved size bytes
 * forward.
 * @param char * data - the memory block to which the data will be written
 * @param long size - the size of the memory block in bytes
 */
void FileStream::read(char* data, long size) {
    //Require(stream != NULL);
    stream->read(data, size);
}

/**
 * @brief Obtain the universal version identifier located under the read head
 * This will not increment the read head, but provides convenience for
 * objects to conduct a look-ahead into the buffer to determine
 * what type of object was serialized, and if they are that type of object
 * they know how to inflate the given object under the read head
 * @return long - the UID of the next object in the stream, -1 if end of stream
 */
long FileStream::getNextUID() {
    //Require(stream != NULL );
    long read_uid=0;
    stream->read((char*)&read_uid, sizeof(long));
    // from the current position, rewind back to before the uid
    stream->seekg(-sizeof(read_uid), std::ios_base::cur);
    return read_uid;
}

/**
 * @brief ignores count bytes
 * This will move the read head count bytes ahead in the stream
 */
void FileStream::ignore(long count) {
    stream->ignore(count);
}

/**
 * @brief Obtain the position of the read head
 * @param long - the read head position
 */
long FileStream::getReadHead() {
    return stream->tellg();
}

/**
 * @brief Set the position of the read head
 * @param long pos - the position from which the next read will occur
 */
void FileStream::setReadHead(long pos) {
    stream->seekg(pos);
}

/**
 * @brief Obtain the position of the write head
 * @param long - the write head position
 */
long FileStream::getWriteHead() {
    return stream->tellp();
}

/**
 * @brief Set the position of the write head
 * @param long pos - the position at which the next write will occur
 */
void FileStream::setWriteHead(long pos) {
    stream->seekp(pos);
}

/**
 * @brief Set the object factory from which to obtain prototypes for deserialization
 * @param SerialFactory * factory - the factory containing prototypes
 */
//void FileStream::setFactory(Standard::SerialFactory * factory) {
//    this->factory = factory;
//}

/**
 * @brief determine if next object in stream is of the given type
 * @param long uid - the universal version id of desired type
 * @return bool - true, if the next object in the stream is the given type
 *                or a child of the given type, false otherwise
 */
bool FileStream::isNext(long uid) {
    long read_uid = getNextUID();
    if( read_uid == uid ) return true;
    //if( factory == NULL ){
    //    qDebug() << "***Failed: expecting "<<uid<<", found "<<read_uid<<", but no factory set!";
    //    return false;
    //}

    // determine if the next object is
    // a child object of the given type (uid)
    //Standard::Serializable * read_obj = factory->getSerializable(read_uid);
    //if( read_obj == NULL ){
    //    qDebug() << "***Failed: expecting "<<uid<<", found "<<read_uid<<", but no object mapped!";
    //    return false;
    //}
    //if( read_obj->childOf(uid) ) return true;
    qDebug() << "***Failed: expecting "<<uid<<", found "<<read_uid<<", but not a child!";
    return false;
}


