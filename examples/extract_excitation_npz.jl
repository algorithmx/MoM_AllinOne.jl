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
# Use a default mesh file found in the codebase
meshfilename = joinpath(@__DIR__, "..", "meshfiles/jet_100MHz.nas")
meshUnit = :m

println("Initializing parameters...")
inputParameters(;frequency = frequency)

# 3. Load Mesh
println("Loading mesh from $meshfilename ...")
meshData, εᵣs = getMeshData(meshfilename; meshUnit=meshUnit)

println("Generating geometry info...")
# We use sbfT=:RWG as default for surface mesh
ngeo, nbf, geosInfo, bfsInfo = getBFsFromMeshData(meshData; sbfT = :RWG)

# Handle return type variability of getBFsFromMeshData
triangles = []
if eltype(geosInfo) <: MoM_Basics.TriangleInfo
    triangles = geosInfo
elseif eltype(geosInfo) <: Vector
    # If it returns a vector of vectors (e.g. for mixed meshes or generic return)
    # Flatten it
    for part in geosInfo
        if eltype(part) <: MoM_Basics.TriangleInfo
            append!(triangles, part)
        end
    end
else
    println("Unknown geometry info type: $(typeof(geosInfo))")
end

println("Found $(length(triangles)) triangles.")

# 4. Define Source
# PlaneWave(θ, ϕ, α, V)
source = PlaneWave(π/2, 0, 0f0, 1f0)

# 5. Extract Fields
println("Extracting [r, E(r), H(r)] and saving to NPZ...")

output_file = "excitation_fields.npz"
saveExcitationFields(output_file, triangles, source)

println("Extraction complete. Data saved to $output_file")

# Verify the file exists
if isfile(output_file)
    println("Verified: $output_file exists.")
else
    println("Error: $output_file was not created.")
end
