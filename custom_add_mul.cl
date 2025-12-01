// Custom OpenCL Kernel for (in0 + in1) * in2

__kernel void custom_add_mul(
    const __global float* in0,
    const __global float* in1,
    const __global float* in2,
    __global float* out)
{
    const uint idx = get_global_id(0);
    
    // Full operation
    out[idx] = (in0[idx] + in1[idx]) * in2[idx];
}
