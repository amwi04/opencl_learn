### OPENCL CONTEXT
- context is an abstract environment within which coordination and memory management for kernel execution is valid and well defined. A context coordinates the mechanisms for host-device interaction, manages the memory objects available to the devices, and keeps track of the programs and kernels that are created for each device. The API function to create a context is clCreateContext().

```
 clCreateContext (
   const cl_context_properties *properties,
   cl_uint num_devices,
   const cl_device:id *devices,
   void (CL_CALLBACK *pfn_notify)(
      const char *errinfo,
      const void *private_info,
      size_t cb,
      void *user_data),
   void *user_data,
   cl_int *errcode_ret)
```
 - A command-queue is the communication mechanism that the host uses to request action by a device. Once the host has decided which devices to work with and a context has been created, one command-queue needs to be created per device.
 
 ```
  cl_command_queue
   clCreateCommandQueueWithProperties(
   cl_context context,
   cl_device:id device,
   cl_command_queue_properties properties,
   cl_int* errcode_ret)
```

- Events
    - Queued: The command has been placed into a command-queue.

    - Submitted: The command has been removed from the command-queue and has been submitted for execution on the device.

    - Ready: The command is ready for execution on the device.

    - Running: Execution of the command has started on the device.

    - Ended: Execution of the command has finished on the device.

    - Complete: The command and all of its child commands have finished.

```
clWaitForEvents (
   cl_uint num_events,
   const cl_event *event_list)
```
- Enqueuing a command to a device to begin kernel execution is done with a call to clEnqueueNDRangeKernel(). A command-queue must be specified so the target device is known. The kernel object identifies the code to be executed. Four fields are then related to work-item creation. The work_dim parameter specifies the number of dimensions (one, two, or three) in which work-items will be created. 

```
clEnqueueNDRangeKernel(
   cl_command_queue command_queue,
   cl_kernel kernel,
   cl_uint work_dim,
   const size_t *global_work_offset,
   const size_t *global_work_size,
   const size_t *local_work_size,
   cl_uint num_events_in_wait_list,
   const cl_event *event_wait_list,
   cl_event *event)

```
### MEMORY OBJECTS

1. Buffers
- Buffers are equivalent to arrays in C created using malloc(), where data elements are stored contiguously in memory. Conceptually, it may help to visualize an OpenCL buffer object as a pointer that is valid on a device. The API function clCreateBuffer() allocates space for the buffer and returns a memory object.

```
cl_mem
clCreateBuffer(
   cl_context context,
   cl_mem_flags flags,
   size_t size,
   void *host_ptr,
   cl_int *errcode_ret)
```

2. Images
- Images are OpenCL memory objects that abstract the storage of physical data to allow device-specific optimizations. Unlike buffers, images cannot be directly referenced as if they were arrays. Further, adjacent data elements are not guaranteed to be stored contiguously in memory. The purpose of using images is to allow the hardware to take advantage of spatial locality and to utilize the hardware acceleration available on many devices.

```
cl_mem
clCreateImage(
   cl_context context,
   cl_mem_flags flags,
   const cl_image_format *image_format,
   const cl_image_desc *image_desc,
   void *host_ptr,
   cl_int *errcode_ret)
```


3. Pipes
- A pipe memory object is an ordered sequence of data items (referred to as packets) that are stored on the basis of a first in, first out (FIFO) method. A pipe has a write endpoint into which data items are inserted, and a read endpoint from which data items are removed. When creating a pipe using the OpenCL API call clCreatePipe(), one must supply the packet size along with the number of entries in the pipe (i.e. the maximum number of packets that can fit into the pipe at once). The function clGetPipeInfo() can return information about the size of the pipe and the maximum number of packets that can reside in the pipe. The properties argument is reserved for future use, and should be NULL in OpenCL 2.0.

```
cl_mem
clCreatePipe (
   cl_context context,
   cl_mem_flags flags,
   cl_uint pipe_packet_size,
   cl_uint pipe_max_packets,
   const cl_pipe_properties *properties,
   cl_int *errcode_ret)
```







