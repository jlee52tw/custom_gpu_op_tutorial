// Custom OpenCL Kernel for (in0 + in1) * in2

__kernel void custom_add_mul(
    const __global float* in0,
    const __global float* in1,
    const __global float* in2,
    __global float* out)
{
    const uint idx = get_global_id(0);
    
    // Simple element-wise operation
    // Ensure that the global work size matches the total number of elements
    out[idx] = (in0[idx] + in1[idx]) * in2[idx];
}
