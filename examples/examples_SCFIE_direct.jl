## 导入程序包
push!(LOAD_PATH, joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "../../MoM_Basics.jl"))
push!(LOAD_PATH, joinpath(@__DIR__, "../../MoM_Kernels.jl"))
push!(LOAD_PATH, joinpath(@__DIR__, "../../MoM_Visualizing.jl"))
push!(LOAD_PATH, joinpath(@__DIR__, "../../IterativeSolvers.jl"))
using MoM_AllinOne
# using MKL, MKLSparse
using DataFrames, CSV, LaTeXStrings
using CairoMakie, MoM_Visualizing

## 参数设置
# 设置精度，是否运行时出图等
setPrecision!(Float32)
SimulationParams.SHOWIMAGE = true

# 网格文件
filename = joinpath(@__DIR__, "..", "meshfiles/sphere_600MHz.nas")
meshUnit = :m
## 设置输入频率（Hz）从而修改内部参数
frequency = 6e8

# 积分方程类型
ieT  = :CFIE

# 更新基函数类型参数(不推荐更改)
sbfT = :RWG
vbfT = :nothing

# 求解器类型
solverT = :direct

# 设置 gmres 求解器精度，重启步长(步长越大收敛越快但越耗内存)
rtol    = 1e-3
restart = 50

# 源
source  =   PlaneWave(π/2, 0, 0f0, 1f0)

## 远场观测角度
θs_obs  =   LinRange{Precision.FT}(  -π, π,  721)
ϕs_obs  =   LinRange{Precision.FT}(  0, π/2,  2 )

##开始计算
# 计算脚本
include(joinpath(@__DIR__, "../src/direct_solver.jl"))

## 比较绘图
# 导入feko数据
feko_RCS_file = joinpath(@__DIR__, "../deps/compare_feko/sphere_600MHzRCS.csv")
# Skip the first header line which has irregular spacing and causes warnings
data_feko = DataFrame(CSV.File(feko_RCS_file, delim=' ', ignorerepeated=true, header=false, skipto=2))

# Filter data based on PHI column (Column 2) to handle potential row count mismatch
# Column 7 is RCS
RCS_phi0  = data_feko[isapprox.(data_feko[!, 2], 0, atol=1e-1), 7]
RCS_phi90 = data_feko[isapprox.(data_feko[!, 2], 90, atol=1e-1), 7]

# Ensure lengths match θs_obs (721)
if length(RCS_phi0) != length(θs_obs)
    @warn "Mismatch in FEKO phi=0 data length: $(length(RCS_phi0)) vs $(length(θs_obs)). Truncating or padding."
    resize!(RCS_phi0, length(θs_obs))
end
if length(RCS_phi90) != length(θs_obs)
    @warn "Mismatch in FEKO phi=90 data length: $(length(RCS_phi90)) vs $(length(θs_obs)). Truncating or padding (duplicating last)."
    # If missing one, duplicate last
    if length(RCS_phi90) < length(θs_obs)
        append!(RCS_phi90, fill(RCS_phi90[end], length(θs_obs) - length(RCS_phi90)))
    else
        resize!(RCS_phi90, length(θs_obs))
    end
end

RCS_feko = hcat(RCS_phi0, RCS_phi90)

# 绘图保存
fig = farfield2D(θs_obs, 10log10.(RCS_feko), 10log10.(RCS),
                [L"\text{Feko}\;\quad (\phi = \enspace0^{\circ})", L"\text{Feko}\;\quad (\phi = 90^{\circ})"], 
                [L"\text{JuMoM} (\phi = \enspace0^{\circ})", L"\text{JuMoM} (\phi = 90^{\circ})"],
                xlabel = L"\theta (^{\circ})", ylabel = L"\text{RCS(dBsm)}", x_unit = :rad, legendposition = :rt)
savedir = joinpath(@__DIR__, "..", "figures")
!ispath(savedir) && mkpath(savedir)
save(joinpath(savedir, "SCFIE_RCS_sphere_600MHz_direct.pdf"), fig)

# Output RCS comparison to CSV file
comparison_df = DataFrame(
    Theta_deg = θs_obs .* (180/π),
    RCS_Feko_Phi0_dB = 10log10.(RCS_feko[:, 1]),
    RCS_JuMoM_Phi0_dB = 10log10.(RCS[:, 1]),
    RCS_Feko_Phi90_dB = 10log10.(RCS_feko[:, 2]),
    RCS_JuMoM_Phi90_dB = 10log10.(RCS[:, 2]),
    Diff_Phi0_dB = 10log10.(RCS_feko[:, 1]) .- 10log10.(RCS[:, 1]),
    Diff_Phi90_dB = 10log10.(RCS_feko[:, 2]) .- 10log10.(RCS[:, 2])
)
CSV.write(joinpath(savedir, "RCS_Comparison_Sphere_600MHz.csv"), comparison_df)
println("RCS comparison data saved to: ", joinpath(savedir, "RCS_Comparison_Sphere_600MHz.csv"))
