__kernel void square_kernal(__global int* values, __global int* final ) {
    int global_id = get_global_id(0);
    final[global_id] = values[global_id]*values[global_id];
}
