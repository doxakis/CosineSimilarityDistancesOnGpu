# Cosine similarity distances on GPU
Compute cosine similarity distances for all combinations of the dataset on the gpu with CUDA

This was coded in c# with the library Alea GPU. A similar logic could be reuse in any language (Python, c++, etc.)

I plan to use it as a preprocessing step before running HDBSCAN (text clustering in a unsupervised way).
Calculating distances could make the algorithm faster and can be a way to scale out. (No need to use PCA to reduce the complexity)
This is more like a proof of concept.

Please note that the first time the kernel function run, a JIT compilation occur. It takes about 1 sec.
I would recommend to run it when starting your application if possible to minimize the impact on perceived performance.

# Future works
- Batch processing (if the array is too large, it does not work. We got : System.Exception: '[CUDAError] CUDA_ERROR_OUT_OF_MEMORY')
- Find optimal parameter (determine if it's better to use CPU only)

# Copyright and license
Code released under the MIT license.
