#ifndef AMPXLIB_NUCLIDE_FILTER_H
#define AMPXLIB_NUCLIDE_FILTER_H

#include "Filter.h"
#include <algorithm>
class AmpxLibrary;

/**
 * NuclideFilter - Intended to allow for filtering nuclide components
 */
class NuclideFilter : public ScaleData::Filter
{
public:
    /**
     * TypeDef the types of data structures that describe our filter
     */
    typedef std::vector<int> MtList;

    /**
     * TypeDef for convenience the policy
     */
    typedef ScaleData::Filter::Policy Policy;
private:
    /**
     * Convenience method to set a filter description with policy 
     * and including the data key and policy key. The data key (aKey) and 
     * policy key (pKey) cannot be the same
     */
    template<class T> 
    void set(T& a, const Policy& policy, const std::string& aKey, const std::string& pKey)
    {
        std::sort(a.begin(), a.end()); // sort it for future binary searches
        Filter::set(aKey, a); // set the filter description
        Filter::set(pKey, policy); // set the filter policy
        
    }
    template<class C,class I> 
    bool accepts(const I& a, const std::string& aKey, const std::string& pKey) const
    {
        Policy policy = Filter::getGlobalPolicy();
        // Check for key specific/local policy
        if( Filter::contains<Policy>(pKey) ){
            policy = Filter::get<Policy>(pKey);
        }
        // if specific ids are not specified, produce result based on provided policy
        if( !Filter::contains<C>(aKey) ){
            return policy == Keep;
        }
        // acquire ids
        const C & ids = Filter::get<C>(aKey);
        // conduct a binary search for object id
        bool listed = std::binary_search(ids.begin(), ids.end(), a);

        return (listed && policy == Keep ) || (!listed && policy==Ignore);
    }

public:
    /**
     * Construct a nuclide filter with default global policy of ignoring 
     * everything not specified
     * */
    NuclideFilter(Policy globalPolicy=Ignore)
    : Filter(globalPolicy)
    {
    }
    NuclideFilter(const NuclideFilter& orig)
    : Filter(orig)
    {
    }
    /**
     * Determine if this filter contains an AmpxLibrary used to backup
     * the nuclide during filtering
     * @return bool - true, if and only if and AmpxLibrary is present, false otherwise
     */
    bool containsAmpxLib()const
    {
        return Filter::contains<const AmpxLibrary*>(AmpxHelperLibKey);  
    } 
    /**
     * Acquire the AmpxLibrary used to backup nuclides being filtered by this object
     * @return AmpxLibrary * - The AmpxLibrary that was set to backup nuclides being filtered
     *                         If no ampx library was set, NULL is returned
     */
    AmpxLibrary * getAmpxLib() const
    {
        if( !containsAmpxLib() ) return NULL;
        return const_cast<AmpxLibrary*>(Filter::get<const AmpxLibrary*>(AmpxHelperLibKey));
    }
    /**
     * Set the AmpxLibrary to back up filtered nuclides
     * @param lib - the AmpxLibrary 
     */
    void setAmpxLib(const AmpxLibrary * lib)
    {
        Filter::set(AmpxHelperLibKey,lib);
    }
    /**
     * Set a discrete description of bondarenko data to be filtered
     * @param ids - the list of ids of interest
     * @param policy = Ignore - the Filter policy to apply, default to Ignore
     */
    void setBondarenkoMts(MtList ids,Policy policy = Ignore)
    {
        set(ids, policy, BondarenkoMtListKey, BondarenkoMtListPolicyKey); // set the filter description
    }
    /**
     * Does this filter accept the given id
     */
    bool acceptsBondarenko(int id) const
    {
        return accepts<MtList,int>(id,BondarenkoMtListKey, BondarenkoMtListPolicyKey); 
    } // end of acceptsBondarenko
    
    /**
     * Set a discrete description of neutron 1ds to be filtered
     * @param ids - the list of ids of interest
     * @param policy = Ignore - the Filter policy to apply, default to Ignore
     */
    void setNeutron1dMts(MtList ids,Policy policy = Ignore)
    {
        set(ids, policy, Neutron1dMtListKey, Neutron1dMtListPolicyKey); // set the filter description
    }
    /**
     * Set a discrete description of neutron 2ds to be filtered
     * @param ids - the list of ids of interest
     * @param policy = Ignore - the Filter policy to apply, default to Ignore
     */
    void setNeutron2dMts(MtList ids,Policy policy = Ignore)
    {
        set(ids, policy, Neutron2dMtListKey, Neutron2dMtListPolicyKey); // set the filter description
    }
    /**
     * Does this filter accept the given id
     */
    bool acceptsNeutron1d(int id) const
    {
        return accepts<MtList,int>(id,Neutron1dMtListKey, Neutron1dMtListPolicyKey); 
    } // end of acceptsNeutron1d
    /**
     * Does this filter accept the given id
     */
    bool acceptsNeutron2d(int id) const
    {
        return accepts<MtList,int>(id,Neutron2dMtListKey, Neutron2dMtListPolicyKey); 
    } // end of acceptsNeutron2d
    
    /**
     * Set a discrete description of gamma 1ds to be filtered
     * @param ids - the list of ids of interest
     * @param policy = Ignore - the Filter policy to apply, default to Ignore
     */
    void setGamma1dMts(MtList ids,Policy policy = Ignore)
    {
        set(ids, policy, Gamma1dMtListKey, Gamma1dMtListPolicyKey); // set the filter description
    }
    /**
     * Set a discrete description of gamma 2ds to be filtered
     * @param ids - the list of ids of interest
     * @param policy = Ignore - the Filter policy to apply, default to Ignore
     */
    void setGamma2dMts(MtList ids,Policy policy = Ignore)
    {
        set(ids, policy, Gamma2dMtListKey, Gamma2dMtListPolicyKey); // set the filter description
    }
    /**
     * Does this filter accept the given id
     */
    bool acceptsGamma1d(int id) const
    {
        return accepts<MtList,int>(id,Gamma1dMtListKey, Gamma1dMtListPolicyKey); 
    } // end of acceptsGamma1d
    /**
     * Does this filter accept the given id
     */
    bool acceptsGamma2d(int id) const
    {
        return accepts<MtList,int>(id,Gamma2dMtListKey, Gamma2dMtListPolicyKey); 
    } // end of acceptsGamma2d
    
    /**
     * Set a discrete description of gamma Prods to be filtered
     * @param ids - the list of ids of interest
     * @param policy = Ignore - the Filter policy to apply, default to Ignore
     */
    void setGammaProdMts(MtList ids,Policy policy = Ignore)
    {
        set(ids, policy, GammaProdMtListKey, GammaProdMtListPolicyKey); // set the filter description
    }

    /**
     * Does this filter accept the given id
     */
    bool acceptsGammaProd(int id) const
    {
        return accepts<MtList,int>(id,GammaProdMtListKey, GammaProdMtListPolicyKey); 
    } // end of acceptsGammaProd
private:
    //
    // Description and Policy keys
    // 
    static const std::string BondarenkoMtListKey;
    static const std::string BondarenkoMtListPolicyKey;
    static const std::string Neutron1dMtListKey;
    static const std::string Neutron1dMtListPolicyKey;
    static const std::string Neutron2dMtListKey;
    static const std::string Neutron2dMtListPolicyKey;
    static const std::string Gamma1dMtListKey;
    static const std::string Gamma1dMtListPolicyKey;
    static const std::string Gamma2dMtListKey;
    static const std::string Gamma2dMtListPolicyKey;
    static const std::string GammaProdMtListKey;
    static const std::string GammaProdMtListPolicyKey;
    static const std::string AmpxHelperLibKey;

}; // end of NuclideFilter
#endif // end of AMPXLIB_NUCLIDE_FILTER_H include gaurd
