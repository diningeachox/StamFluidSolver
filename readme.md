# C++ Implementation of Jos Stam's [Stable Fluid Solver](http://graphics.cs.cmu.edu/nsp/course/15-464/Fall09/papers/StamFluidforGames.pdf)

Jos Stam's seminal paper in fluid simulation provided a stable (meaning that the flows don't blow up) way to simulate the Navier-Stokes equations of a fluid.
This implementation was written in C++ and rendered using OpenGL.

## Usage
- Right-mouse: drop coloured dye into the fluid
- Left-mouse: move the fluid around
- Q: switch to **red** dye
- W: switch to **green** dye
- E: switch to **blue** dye

## Packages/Hardware required
- OpenGL (glfw and glad)
- cmake
- CUDA compute toolkit (>=11.3)
- Microsoft Visual Studio 2019
- An NVIDIA GPU

## Build Instructions
After installing the packages in the previous section, make sure they are  all included in the PATH environment variable.
Then check the compute capability of your NVIDIA GPU here: (https://developer.nvidia.com/cuda-gpus). This will be needed
for buliding the CUDA components.
First make a build subdirectory in the root and go into it
```
mkdir build
cd build
```

then run
```
cmake -G "Visual Studio 16" -DCMAKE_CUDA_FLAGS="-arch=$(compute_capability)" ..
msbuild StamFluidSolver.sln /t:Rebuild /p:Configuration=$(build_config);Platform=$(arch)
```
where ```$(compute_capability)``` equals "sm_" plus the compute capability of your GPU without decimals. So
if your GPU has compute capabiility 6.1, then your ```-arch``` flag equals ```sm_61```.
```$(compute_capability)``` is either "Debug" or "Release".
```$(arch)``` is x64 (64-bit) or x86 (32-bit).

## Video

https://user-images.githubusercontent.com/24876548/179864629-a01feb21-14e5-4fca-8a7d-6036a3d4bba9.mp4
