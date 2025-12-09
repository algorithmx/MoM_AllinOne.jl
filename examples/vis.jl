# ==============================================================================
# Example: SEFIE Direct Solver for Jet Model
# 
# This script performs a Moment Method simulation (EFIE) on a jet mesh.
# It calculates the Radar Cross Section (RCS) and Surface Currents.
# ==============================================================================

## 导入程序包
push!(LOAD_PATH, joinpath(@__DIR__, ".."))
push!(LOAD_PATH, joinpath(@__DIR__, "../../MoM_Basics.jl"))
push!(LOAD_PATH, joinpath(@__DIR__, "../../MoM_Kernels.jl"))
push!(LOAD_PATH, joinpath(@__DIR__, "../../MoM_Visualizing.jl"))
push!(LOAD_PATH, joinpath(@__DIR__, "../../IterativeSolvers.jl"))

## 1. Imports
using MoM_AllinOne
using DataFrames, CSV, LaTeXStrings
using CairoMakie, MoM_Visualizing
using LinearAlgebra

## 2. Simulation Configuration
# Set floating point precision (Float32 is faster, Float64 is more accurate)
setPrecision!(Float32)
# Enable showing plots during execution
SimulationParams.SHOWIMAGE = true

# Input Parameters
frequency = 1e8             # Frequency in Hz (100 MHz)
ieT       = :EFIE           # Integral Equation Type: Electric Field Integral Equation (for PEC)
sbfT      = :RWG            # Surface Basis Function: RWG
vbfT      = :nothing        # Volume Basis Function: None (Surface only)
solverT   = :direct         # Solver: Direct (LU Decomposition) / Iterative

# GMRES Solver Parameters (Only used if solverT = :gmres)
rtol      = 1e-3
restart   = 50

# Update internal parameters
inputParameters(;frequency = frequency, ieT = ieT)
updateVSBFTParams!(;sbfT = sbfT, vbfT = vbfT)

## 3. Mesh Loading
# Locate the mesh file (adjust path if necessary)
mesh_path = joinpath(@__DIR__, "../meshfiles/jet_100MHz.nas")
meshUnit  = :m

if !isfile(mesh_path)
    error("Mesh file not found at: $mesh_path")
end

println("Loading mesh from: $mesh_path")
meshData, εᵣs = getMeshData(mesh_path; meshUnit=meshUnit)

## 4. Basis Functions & Geometry Setup
println("Generating Basis Functions...")
ngeo, nbf, geosInfo, bfsInfo = getBFsFromMeshData(meshData; sbfT = sbfT, vbfT = vbfT)

# Optional: Set permittivity (Example uses this, though standard PEC EFIE usually ignores it)
setGeosPermittivity!(geosInfo, 2(1-0.0002im))

## 5. Matrix Assembly & Solution
println("Assembling Impedance Matrix (Size: $nbf x $nbf)...")
Zmat = getImpedanceMatrix(geosInfo, nbf)

# Define Excitation (Plane Wave: θ=90°, ϕ=0°, Pol=Linear)
source = PlaneWave(π/2, 0, 0f0, 1f0)
V = getExcitationVector(geosInfo, size(Zmat, 1), source)

println("Solving Linear System...")
# Returns Current Coefficients (ICoeff) and Convergence History (ch)
ICoeff, ch = solve(Zmat, V; solverT = solverT, rtol = rtol, restart = restart)

## 6. RCS Calculation
println("Calculating Radar Cross Section (RCS)...")
θs_obs = LinRange{Precision.FT}(-π, π, 721)
ϕs_obs = LinRange{Precision.FT}(0, π/2, 2)

# Calculate RCS
RCSθsϕs, RCSθsϕsdB, RCS, RCSdB = radarCrossSection(θs_obs, ϕs_obs, ICoeff, geosInfo)

## 7. Visualization & Comparison

# Create output directory
savedir = joinpath(@__DIR__, "results_jet")
!ispath(savedir) && mkpath(savedir)

# --- 7a. RCS Plot with FEKO Comparison ---
println("Plotting RCS...")
feko_RCS_file = joinpath(@__DIR__, "../deps/compare_feko/jet_100MHzRCS.csv")

if isfile(feko_RCS_file)
    println("Found Reference FEKO Data. Comparison Plot...")
    data_feko = DataFrame(CSV.File(feko_RCS_file, delim=' ', ignorerepeated=true))
    RCS_feko = reshape(data_feko[!, "in"], :, 2)

    fig_rcs = farfield2D(θs_obs, 10log10.(RCS_feko), 10log10.(RCS),
        [L"\text{Feko}\;\quad (\phi = \enspace0^{\circ})", L"\text{Feko}\;\quad (\phi = 90^{\circ})"], 
        [L"\text{JuMoM} (\phi = \enspace0^{\circ})", L"\text{JuMoM} (\phi = 90^{\circ})"],
        xlabel = L"\theta (^{\circ})", ylabel = L"\text{RCS(dBsm)}", x_unit = :rad, legendposition = :lb)
    
    save(joinpath(savedir, "SEFIE_RCS_Comparison.pdf"), fig_rcs)
else
    println("FEKO Data not found. Plotting JuMoM results only...")
    fig_rcs = farfield2D(θs_obs, 10log10.(RCS), 
        [L"\text{JuMoM} (\phi = %$(rad2deg(ϕ))^{\circ})" for ϕ in ϕs_obs];
        xlabel = L"\theta (^{\circ})", ylabel = L"\text{RCS(dBsm)}", x_unit = :rad)
    
    save(joinpath(savedir, "SEFIE_RCS_JuMoM.pdf"), fig_rcs)
end

# --- 7b. Surface Current Visualization ---
println("Visualizing Surface Currents...")

# Calculate physical current density J at mesh centers
Jgeos = geoElectricJCal(ICoeff, geosInfo)

# Plot Total Magnitude
J_mag = norm.(eachcol(Jgeos))
fig_current = visualizeMesh(meshData, J_mag; 
                           legendlabel = "Surface Current |J| (A/m)", 
                           title = "Current Distribution on Jet")

save(joinpath(savedir, "Jet_Surface_Current.png"), fig_current)

# Plot Real Part Magnitude
J_mag_real = norm.(eachcol(real.(Jgeos)))
fig_current_real = visualizeMesh(meshData, J_mag_real; 
                           legendlabel = "Surface Current |Re(J)| (A/m)", 
                           title = "Real Current Distribution on Jet")
save(joinpath(savedir, "Jet_Surface_Current_Real.png"), fig_current_real)

# Plot Imaginary Part Magnitude
J_mag_imag = norm.(eachcol(imag.(Jgeos)))
fig_current_imag = visualizeMesh(meshData, J_mag_imag; 
                           legendlabel = "Surface Current |Im(J)| (A/m)", 
                           title = "Imaginary Current Distribution on Jet")
save(joinpath(savedir, "Jet_Surface_Current_Imag.png"), fig_current_imag)

# --- 7c. Component-wise Surface Current Visualization ---
components = ["x", "y", "z"]

for (i, comp) in enumerate(components)
    # Extract component i (1->x, 2->y, 3->z)
    J_comp = Jgeos[i, :]

    # Real part of component
    J_real = real.(J_comp)
    fig_real = visualizeMesh(meshData, J_real; 
                             legendlabel = "Surface Current Re(J_$comp) (A/m)", 
                             title = "Real Part of J_$comp")
    save(joinpath(savedir, "Jet_Surface_Current_$(comp)_Real.png"), fig_real)

    # Imaginary part of component
    J_imag = imag.(J_comp)
    fig_imag = visualizeMesh(meshData, J_imag; 
                             legendlabel = "Surface Current Im(J_$comp) (A/m)", 
                             title = "Imaginary Part of J_$comp")
    save(joinpath(savedir, "Jet_Surface_Current_$(comp)_Imag.png"), fig_imag)
end

println("Done! Results saved to: $savedir")
