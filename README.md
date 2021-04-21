# Parallel TTr1-Tensor: Randomized Compression-based Scheme for Tensor Train Rank-1 Decomposition  
code for the implementation of Parallel TTr1-Tensor  

## Abstract    
Tensor train rank-1 (TTr1) decomposition, as a variant of tensor train decomposition, is widely used in quantum machine learning to decompose high-order tensors into low-order tree-like tensor networks. However, due to the storage limitation, tensors can not be fully loaded in the main memory simultaneously. A key step in the optimization of TTr1 is the utilization of Singular Value Decomposition (SVD) for the tensor matricization, and memory limitation becomes the bottleneck of existing TTr1 decomposition. In this work, we propose Parallel TTr1-Tensor, a randomized compression-based scheme for tensor train rank-1 decomposition by trading computation for storage. In addition, we maximize the parallelism of tensor operations to accelerate the proposed scheme on GPUs. In experiments, we test the relative error of reconstruction and running time of our scheme on various scales. Our method achieves a maximum of 151.2× speedup on GPUs versus CPUs with less than 20% relative errors. To our knowledge, this is the first work to utilize the compression-based technique to handle the TTr1 decomposition bottleneck due to the limited memory.  

## Overview of Parallel TTr1-Tensor  
![image](https://github.com/reroze/TTr1/blob/main/overview.png)  
An example for our parallel TTr1-Tensor scheme. A tensor X ∈ R^{100×100×100} is first compressed into 3 replicas, then each replica is factorized to a tree-like structure. The recovery stage is executed to recover the factorization factors.  

## Optimization  
Preliminary strategies including handling large scale tenosrs, improving the utilizaiton of GPUs and reducing the communication cost.  

## Performance Evaluation  

![image](https://github.com/reroze/TTr1/blob/main/ex.png)    
(a) shows the influence on the number of replicas P , and the compression ratio on the relative construction error with High-order Sketch Compression. (b) and (c) respectively show the comparison on the relative construction error and time consumption between the CPU and GPU implementation with Gaussian Compression.  

![image](https://github.com/reroze/TTr1/blob/main/ex2.png)  
(a) and (b) respectively show the comparison on the relative construction error and time consumption between the CPU and GPU implementation for large scale tensors with Gaussian Compression.  

## Running Envionment  
Python 3.7  with Numpy and Pycuda  

## Citation  
If you find Parallel TTr1-Tensor useful in your research, please consider citing:  
@article{Zhang2020parallelttr1,  
    Author = {Zeliang Zhang, Junzhe Zhang, Guoping Lin, Zeyuan Yin, Kun He},  
    Title = {Parallel TTr1-Tensor: Randomized Compression-based Scheme for Tensor Train Rank-1 Decomposition},  
    Journal = {QTML workshop on NIPS2020},   
    Year = {2020}   
}     
Our paper can be accessed online from https://tensorworkshop.github.io/NeurIPS2020/accepted_papers.html  
