#include "histogram.h"

Histogram::Histogram(float min, float max, float bins) : m_min(min), m_max(max), m_bins(bins)
{
    if(bins >= 0)
    {
        m_upperBounds.resize(m_bins);
        m_counts.resize(m_bins, 0);
    }

    const float dx = (m_max-m_min)/m_bins;
    float v = m_min + dx;
    int indx = 0;

    while(v < m_max)
    {
        m_upperBounds[indx] = v;
        v += dx;
        indx++;
    }
    m_upperBounds[m_upperBounds.size()-1] = m_max;

}

Histogram::~Histogram()
{

}

void Histogram::insert(float x)
{
    if(x < m_min)
        m_underflows++;
    else if(x > m_max)
        m_overflows++;
    else
    {
        for(unsigned int indx = 0; indx < m_upperBounds.size(); indx++)
        {
            if(m_upperBounds[indx] > x)
            {
                m_counts[indx]++;
                break;
            }
        }
    }
    return;
}

std::ostream& operator<<(std::ostream& out, const Histogram& hist)
{
    out << hist.m_min << '\t' << hist.m_underflows << '\n';
    for(int i = 0; i < hist.m_bins; i++)
    {
        out << hist.m_upperBounds[i] << '\t' << hist.m_counts[i] << '\n';
    }
    out << "0\t" << hist.m_overflows;

    return out;
}
