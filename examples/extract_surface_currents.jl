## 导入程序包
push!(LOAD_PATH, joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "../../MoM_Basics.jl"))
push!(LOAD_PATH, joinpath(@__DIR__, "../../MoM_Kernels.jl"))
push!(LOAD_PATH, joinpath(@__DIR__, "../../MoM_Visualizing.jl"))
push!(LOAD_PATH, joinpath(@__DIR__, "../../IterativeSolvers.jl"))

using MoM_AllinOne
using MoM_Basics
using StaticArrays

# 1. Setup Environment
setPrecision!(Float32)

# 2. Configure Parameters
frequency = 1e8
meshfilename = joinpath(@__DIR__, "..", "meshfiles/jet_100MHz.nas")
meshUnit = :m
ieT  = :EFIE
sbfT = :RWG
vbfT = :nothing
solverT = :direct
rtol    = 1e-5
restart = 50

println("Initializing parameters...")
inputParameters(;frequency = frequency, ieT = ieT)
updateVSBFTParams!(;sbfT = sbfT, vbfT = vbfT)

# 3. Load Mesh
println("Loading mesh from $meshfilename ...")
meshData, εᵣs = getMeshData(meshfilename; meshUnit=meshUnit)

println("Generating geometry info...")
ngeo, nbf, geosInfo, bfsInfo = getBFsFromMeshData(meshData; sbfT = sbfT, vbfT = vbfT)

# 4. Setup Problem
setGeosPermittivity!(geosInfo, 2(1-0.0002im))

# Define Source and observation angles
source  = PlaneWave(π/2, 0, 0f0, 1f0)
θs_obs  = LinRange{Precision.FT}(-π, π, 721)
ϕs_obs  = LinRange{Precision.FT}(0, π/2, 2)

# 5. Solve for Surface Currents
println("Computing impedance matrix and excitation vector...")
Zmat = getImpedanceMatrix(geosInfo, nbf)
V    = getExcitationVector(geosInfo, size(Zmat, 1), source)

println("Solving linear system...")
ICoeff, ch = solve(Zmat, V; solverT = solverT, rtol = rtol, restart = restart)

# 6. Extract and Save Data
# We extract both incident fields and result currents, merge them, and save to a single .npz file.

println("Calculating field data...")
data_curr = calSurfaceCurrents(geosInfo, bfsInfo, ICoeff)
data_inc  = calIncidentFields(geosInfo, source)

println("Merging data...")
mergeFieldData!(data_curr, data_inc)

output_file = "combined_fields.npz"
saveFieldData(output_file, data_curr)
println("Combined data saved to $output_file")

println("Extraction complete.")
