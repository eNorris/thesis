#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <vector>
#include <iostream>

class Histogram
{
public:
    Histogram(float min, float max, float bins);
    ~Histogram();

    void insert(float x);

    friend std::ostream& operator<<(std::ostream& out, const Histogram& hist);

protected:
    int m_underflows;
    int m_overflows;
    std::vector<int> m_counts;
    std::vector<float> m_upperBounds;

    float m_min;
    float m_max;
    float m_bins;
};

#endif // HISTOGRAM_H
