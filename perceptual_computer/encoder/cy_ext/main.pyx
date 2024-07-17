from numpy import zeros,linspace,float64
cimport numpy as cnp

def cy_fuzzy_statistic(cnp.ndarray[cnp.float64_t,ndim=1] left,cnp.ndarray[cnp.float64_t,ndim=1] right,
                       int N_bins=100, float begin = 0.,float end = 10.,):
    cdef cnp.ndarray[cnp.float64_t,ndim=1] x = linspace(begin, end, N_bins+1)
    cdef cnp.ndarray[cnp.float64_t,ndim=1] membership = zeros(N_bins,dtype=float64)
    cdef int i,j,N=left.shape[0]
    for i in range(N_bins):
        for j in range(N):
            if left[j] <= x[i] and x[i+1] <= right[j]:
                membership[i] = membership[i] + 1.
    return (x[:-1] + x[1:])/2, membership/membership.max()