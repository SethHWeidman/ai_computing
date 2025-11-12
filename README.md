# ai_computing

## `vector_add`

Script containing a CUDA kernel that:

* Initializes two vectors of length 1M, one of which has values of 1.0, the other of which has values of 2.0.
* Adds them on the GPU, using 256 threads per block.
* Confirms that the resulting values are all 3.0.

### Instructions to run

To build, run:

```
nvcc vector_add.cu -o vector_add
```

Then run `./vector_add` to see: 

```
Max error = 0.000000
```

printed.