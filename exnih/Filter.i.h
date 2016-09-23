namespace ScaleData {

    template<class T>
    bool Filter::contains(const std::string& key) const {
        return data.contains<T>(key);
    } // end of contains

    template<class T>
    T Filter::get(const std::string& key) const {
        if (!data.contains<T>(key)) throw std::runtime_error("Filter does not contain the appropriately typed description of '" + key + "'");
        return data.get<T>(key);
    } // end of get

    template<class T>
    void Filter::set(const std::string& key, T filterDescription) {
        data.put(key, filterDescription);
    }
}//namespace ScaleData
