//
// Created by Gary27 on 2025/3/24.
//

#ifndef HIP_RUNTIME_DROPCAST_H
#define HIP_RUNTIME_DROPCAST_H

#include "TypeTraits.h"


template<int N, typename T, class = Require<HasEnoughComponents<T, N>>>
__host__ __device__ auto DropCast(T v)
{
    using RT = MakeType<BaseType<T>, N>;
    if constexpr (std::is_same_v<T, RT>)
    {
        return v;
    }
    else
    {
        RT out{};

        GetElement<0>(out) = GetElement<0>(v);
        if constexpr (NumElements<RT> >= 2)
        GetElement<1>(out) = GetElement<1>(v);
        if constexpr (NumElements<RT> >= 3)
        GetElement<2>(out) = GetElement<2>(v);
        if constexpr (NumElements<RT> == 4)
        GetElement<3>(out) = GetElement<3>(v);

        return out;
    }
}


#endif //HIP_RUNTIME_DROPCAST_H
