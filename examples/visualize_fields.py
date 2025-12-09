import numpy as np
import meshio
import pyvista as pv
import argparse
import sys
import os
import tempfile

def load_mesh_robust(nas_file):
    # Try reading directly first
    try:
        return meshio.read(nas_file)
    except Exception as e_initial:
        pass # Fallback
    
    print("Attempting to fix minimal Nastran format...")
    
    try:
        with open(nas_file, 'r') as f:
            content = f.read()
    except Exception as e:
        raise e
        
    # Inject headers if missing
    if "BEGIN BULK" not in content:
        # Create a temp file with the fixed content
        # We use delete=False to ensure it exists for meshio to read, then clean up
        with tempfile.NamedTemporaryFile(suffix=".nas", mode='w', delete=False) as tmp:
            tmp.write("BEGIN BULK\n")
            tmp.write(content)
            tmp.write("\nENDDATA\n")
            tmp_path = tmp.name
            
        try:
            print(f"Reading fixed mesh from temp file: {tmp_path}")
            mesh = meshio.read(tmp_path)
            # Cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return mesh
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            print(f"Failed to read even after adding headers: {e}")
            raise e
    
    raise ValueError(f"Could not load mesh: {nas_file}")

def visualize(nas_file, npz_file):
    # 1. Load Mesh
    print(f"Loading mesh: {nas_file}")
    try:
        mesh_data = load_mesh_robust(nas_file)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return

    # Extract triangles
    points = mesh_data.points
    cells = mesh_data.cells_dict.get("triangle")
    
    if cells is None:
        print("Error: No triangles found in mesh.")
        return

    # Create PyVista mesh
    # pv.PolyData(points, faces) expects faces as [n_pts, p0, p1, ..., p_n] flattened
    n_faces = cells.shape[0]
    faces = np.hstack([np.full((n_faces, 1), 3), cells]).flatten()
    mesh = pv.PolyData(points, faces)

    # 2. Load Fields
    print(f"Loading fields: {npz_file}")
    try:
        data = np.load(npz_file)
    except Exception as e:
        print(f"Error loading fields: {e}")
        return
    
    # Check consistency
    n_field_points = data['positions'].shape[0]
    if n_faces != n_field_points:
        print(f"Warning: Mesh has {n_faces} triangles but field data has {n_field_points} points.")
    
    # 3. Add Data to Mesh
    scalar_keys = []
    vector_keys = []

    for key in data.files:
        if key == 'positions': continue
        
        field_c = data[key] # Complex vector (N, 3)
        
        # 1. Overall Magnitude
        mag = np.linalg.norm(field_c, axis=1)
        mag_key = f"{key}_mag"
        mesh.cell_data[mag_key] = mag
        scalar_keys.append(mag_key)
        
        # 2. Components (Amplitude and Phase)
        components = ['x', 'y', 'z']
        for i, comp in enumerate(components):
            # Amplitude
            amp = np.abs(field_c[:, i])
            amp_key = f"{key}_{comp}_amp"
            mesh.cell_data[amp_key] = amp
            scalar_keys.append(amp_key)
            
            # Phase (radians)
            phase = np.angle(field_c[:, i])
            phase_key = f"{key}_{comp}_phase"
            mesh.cell_data[phase_key] = phase
            scalar_keys.append(phase_key)

        # 3. Real Vector (for glyphs)
        real_vec = field_c.real
        real_key = f"{key}_real"
        mesh.cell_data[real_key] = real_vec
        vector_keys.append(real_key)

    print(f"Loaded scalars: {scalar_keys}")
    print(f"Loaded vectors: {vector_keys}")

    # 4. Visualization Loop
    # Enable off-screen rendering for PNG export
    pv.OFF_SCREEN = True
    
    # Try to start virtual framebuffer if needed (linux headless)
    try:
        pv.start_xvfb()
    except Exception:
        pass

    # Plot Scalars
    for scalar_name in scalar_keys:
        print(f"Plotting {scalar_name}...")
        p = pv.Plotter(off_screen=True)
        
        # Choose colormap based on data type
        cmap = "twilight" if "phase" in scalar_name else "jet"
        
        p.add_mesh(mesh, scalars=scalar_name, cmap=cmap, show_edges=False, label=scalar_name)
        p.add_axes()
        p.add_legend()
        p.add_text(scalar_name, position='upper_left')
        p.view_isometric()
        output_filename = f"{scalar_name}.png"
        p.screenshot(output_filename)
        p.close()
        print(f"Saved {output_filename}")

    # Plot Vectors
    for vector_name in vector_keys:
        print(f"Plotting {vector_name}...")
        p = pv.Plotter(off_screen=True)
        
        # Determine magnitude scalar for coloring/scaling
        # e.g. J_real -> J_mag
        prefix = vector_name.rsplit('_', 1)[0] # "J" from "J_real"
        mag_name = f"{prefix}_mag"
        scalar_arg = mag_name if mag_name in scalar_keys else None
        
        # Plot base mesh faintly
        p.add_mesh(mesh, scalars=scalar_arg, cmap="gray", opacity=0.3, show_edges=False)
        
        # Prepare glyphs
        centers = mesh.cell_centers()
        # Copy data to points for glyph filter
        centers.point_data[vector_name] = mesh.cell_data[vector_name]
        if scalar_arg:
            centers.point_data[scalar_arg] = mesh.cell_data[scalar_arg]
            
        # Filter small arrows to reduce clutter
        subset = centers
        if scalar_arg:
             max_val = np.max(centers.point_data[scalar_arg])
             if max_val > 0:
                 mask = centers.point_data[scalar_arg] > 0.02 * max_val
                 subset = centers.extract_points(mask)
        
        # Add arrows
        # scale=scalar_arg uses the magnitude to scale arrow size
        arrows = subset.glyph(scale=scalar_arg, orient=vector_name, tolerance=0.05, factor=0.2)
        p.add_mesh(arrows, color="red", label=f"{vector_name} Vectors")
        
        p.add_axes()
        p.add_legend()
        p.add_text(f"{vector_name} (scaled by {scalar_arg})", position='upper_left')
        p.view_isometric()
        
        output_filename = f"{vector_name}.png"
        p.screenshot(output_filename)
        p.close()
        print(f"Saved {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MoM fields on mesh.")
    parser.add_argument("nas_file", help="Path to the .nas mesh file")
    parser.add_argument("npz_file", help="Path to the .npz field data file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.nas_file):
        print(f"Error: Mesh file not found: {args.nas_file}")
        sys.exit(1)
    if not os.path.exists(args.npz_file):
        print(f"Error: Data file not found: {args.npz_file}")
        sys.exit(1)
        
    visualize(args.nas_file, args.npz_file)
