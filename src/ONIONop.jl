module ONIONop

using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using StaticArrays
using Memoize: @memoize
using LRUCache: LRU

import ChainRulesCore as CRC
import KernelAbstractions as KA

const Maybe{T} = Union{T, Nothing}

include("groupreduce.jl")
include("softmax.jl")
include("attention/attention.jl")
include("norm/norm.jl")

include("rope/llama_rope.jl")

@memoize LRU{Tuple{Any,Integer},UInt64}(maxsize=32) shared_memory(kab, device_id::Integer) =
    _shared_memory(kab, device_id)

_shared_memory(kab, device_id::Integer) = error("Not implemented.")

end
