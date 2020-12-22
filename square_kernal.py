import numpy
import pyopencl as cl
import time

TASKS = 6400

if __name__ == '__main__':

    print('load program from cl source file')
    f = open('square_kernal.cl', 'r', encoding='utf-8')
    kernels = ''.join(f.readlines())
    f.close()

    print('prepare data ... ')
    start_time = time.time()
    matrix = numpy.arange(1, TASKS, dtype=numpy.int32  )
    matrix_final = numpy.zeros(  TASKS, dtype=numpy.int32  )
    time_hostdata_loaded = time.time()

    print('create context')
    ctx = cl.create_some_context()
    print('create command queue')
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    time_ctx_queue_creation = time.time()

    # prepare device memory for OpenCL
    print('prepare device memory for input / output')
    dev_matrix = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=matrix)
    final_matrix = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY ,matrix_final.nbytes)
    time_devicedata_loaded = time.time()

    print('compile kernel code')
    prg = cl.Program(ctx, kernels).build()
    time_kernel_compilation = time.time()

    print('execute kernel programs')
    evt = prg.square_kernal(queue,(TASKS ,), (1, ), dev_matrix,final_matrix)
    #evt = prg.hello_world(queue, (TASKS, ), (1, ), dev_matrix)
    print('wait for kernel executions')
    evt.wait()
    elapsed = 1e-9 * (evt.profile.end - evt.profile.start)
    cl.enqueue_copy(queue, matrix_final, final_matrix).wait()
    print(matrix_final)
    print('done')

    print('Prepare host data took       : {}'.format(time_hostdata_loaded - start_time))
    print('Create CTX/QUEUE took        : {}'.format(time_ctx_queue_creation - time_hostdata_loaded))
    print('Upload data to device took   : {}'.format(time_devicedata_loaded - time_ctx_queue_creation))
    print('Compile kernel took          : {}'.format(time_kernel_compilation - time_devicedata_loaded))
    print('OpenCL elapsed time          : {}'.format(elapsed))
