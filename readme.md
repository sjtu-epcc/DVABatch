# DVABatch

This repository contains the source code for a research paper that was submitted for publication at the [2022 USENIX Annual Technical Conference](https://www.usenix.org/conference/atc22) (ATC22).

## What is DVABatch

The DNN inferences are often batched for better utilizing the hardware in existing DNN serving systems. However, DNN serving exhibits diversity in many aspects, such as input, operator, and load. The unawareness of these diversities results in inefficient processing. Our investigation shows that the inefficiency roots in the feature of existing batching mechanism: one entry and one exit. Therefore, we propose **DVABatch**, a runtime batching system that enables the multi-entry multi- exit batching scheme for existing DNN serving system.

## Environment Preparation

- Hardware&software requirements

  1. Hardware Requirements

     1. CPU: Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz
     2. Memroy: 252G
     3. NVIDIA TitanRTX

  2. Software Requirements

     1. Ubuntu 18.04.6 (Kernel 4.15.0)
     2. GPU Driver: 460.39
     3. CUDA 11.3
     4. CUDNN 8.2
     5. TensorRT 8.0.3.4
     6. RapidJSON
     7. cmake 3.17

- Some software installation tips
  
  1. Environment variables should be added for TensorRT and RapidJSON, including `PATH, LIBRARY_PATH,  LD_LIBRARY_PATH, CMAKE_PREFIX_PATH, CPLUS_INCLUDE_PATH`.

## Getting Start

- Asuming you have the above requirements and you are in the `$HOME` directory, you can clone the repository and start the installation.

```bash
mkdir DVAbatch
git clone git@github.com:sjtu-epcc/DVABatch.git DVAbatch/lego
```

- We use following instructions to compile the DVABatch runtime system.

```shell
cd $HOME/DVABatch/lego
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=~/DVABatch/install ..
make -j
make install
```

- This repo only consists the source files for DVABatch runtime system. DVABatch relies on [Triton Inference Server](https://github.com/triton-inference-server/server.git) to provide DNN services. Therefore, we need to install Triton Inference Server. We also provide a customized Triton Inference Server [here](https://github.com/Raphael-Hao/lego_server) for DVABatch in this repo for simple test. We now build the customized Triton Inference Server.

```bash
git clone https://github.com/Raphael-Hao/lego_server.git $HOME/DVABatch/server
cmake -DCMAKE_INSTALL_PREFIX=~/DVABatch/install ../build
make server
```

- User needs to prepare the sliced models and then we can simply run DVABatch with following commnads.

```bash
cd $HOME/DVABatch/install
./bin/benchmark ../server/config/resnet_152_04.json
```
