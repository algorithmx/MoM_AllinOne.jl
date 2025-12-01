## 激活环境
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

## 安装包
# Add local dependencies
Pkg.add(path = joinpath(@__DIR__, "..", "..", "IterativeSolvers.jl"))
Pkg.add(path = joinpath(@__DIR__, "..", "..", "MoM_Basics.jl"))
Pkg.add(path = joinpath(@__DIR__, "..", "..", "MoM_Kernels.jl"))
Pkg.add(path = joinpath(@__DIR__, "..", "..", "MoM_Visualizing.jl"))

## 初始化
Pkg.instantiate()
