module ONIONopAMDGPUExt

using AMDGPU
using ONIONop

function ONIONop._shared_memory(::ROCBackend, device_id::Integer)
    dev = AMDGPU.devices()[device_id]
    return UInt64(AMDGPU.HIP.properties(dev).sharedMemPerBlock)
end

end
