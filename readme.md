# DVABatch

This repository contains the source code for a research paper that was submitted for publication at the [International Conference for High Performance Computing, Networking, Storage, and Analysis](https://sc21.supercomputing.org/) (SC21).

## What is Abacus

**Abacus** is a runtime system that runs multiple DNN queries simultaneously with stable and predictable latency. **Abacus** enables deterministic operator overlap to enforce the latency predictability. **Abacus** is comprised of an overlap-aware latency predictor, a headroom-based query controller, and segmental model executors. The latency predictor is able to precisely predict the latencies of queries when the operator overlap is determined. The query controller determines the appropriate operator overlap to guarantee the QoS of all the DNN services on a GPU. The model executors run the operators as needed to support the deterministic operator overlap. Our evaluation using seven popular DNNs on an Nvidia A100 GPU shows that **Abacus** significantly reduces the QoS violation and improves the throughput compared with state-of-the-art solutions.

## Environment Preparation

- Hardware&software requirements

  1. Hardware Requirements

     1. CPU: Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz
     2. Memroy: 252G
     3. NVIDIA Ampere 100

  2. Software Requirements

     1. Ubuntu 20.04.1 (Kernel 5.8.0)
     2. GPU Driver: 460.39
     3. CUDA 11.2
     4. CUDNN 8.1
     5. Anaconda3-2020.7
     6. Pytorch 1.8.0
