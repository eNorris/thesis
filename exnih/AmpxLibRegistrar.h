/* 
 * File:   AmpxLibRegistrar.h
 * Author: raq
 *
 * Created on November 4, 2011, 10:36 AM
 */

#ifndef AMPXLIBREGISTRAR_H
#define	AMPXLIBREGISTRAR_H
namespace Standard {
class SerialFactory;
}

/**
 * @brief AmpxLibrary registrar for all serializable AmpxLib objects
 * @param SerialFactory * factory - the factory to which AmpxLib objects are registered
 */
void AmpxLibRegistrar( Standard::SerialFactory * factory);

#endif	/* AMPXLIBREGISTRAR_H */

