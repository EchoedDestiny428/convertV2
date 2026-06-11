import math

import numpy as np
import pyvista as pv

# --- INITIAL CONFIGURATION ---
RESOLUTION = 60
INITIAL_THICKNESS = 0.4
INITIAL_BOUND = math.pi


def gyroid_value(x, y, z):
    return (
        math.sin(x) * math.cos(y)
        + math.sin(y) * math.cos(z)
        + math.sin(z) * math.cos(x)
    )


def format_equation(thickness, bound):
    return (
        "f(x,y,z)=sin(x)cos(y)+sin(y)cos(z)+sin(z)cos(x),",
        f"|f(x,y,z)| <= {thickness:.3f}, x,y,z in [-{bound:.3f}, {bound:.3f}]",
    )

def get_lattice_mesh(thickness, resolution, bound):
    """Generate a solid gyroid band and return its surface mesh."""
    x_min, x_max = -bound, bound
    y_min, y_max = -bound, bound
    z_min, z_max = -bound, bound

    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    z = np.linspace(z_min, z_max, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    values = (np.sin(X) * np.cos(Y) +
              np.sin(Y) * np.cos(Z) +
              np.sin(Z) * np.cos(X))

    grid = pv.ImageData(dimensions=(resolution, resolution, resolution))
    grid.spacing = ((x_max - x_min) / (resolution - 1),
                    (y_max - y_min) / (resolution - 1),
                    (z_max - z_min) / (resolution - 1))
    grid.origin = (x_min, y_min, z_min)

    grid.point_data["values"] = values.flatten(order="F")
    grid.point_data["band"] = (np.abs(values) <= thickness).astype(np.uint8).flatten(order="F")

    # Extract the occupied volume first, then surface-extract it so the shell is solid.
    band_volume = grid.threshold(value=0.5, scalars="band")
    solid_surface = band_volume.extract_surface(algorithm="dataset_surface").triangulate().clean()
    return solid_surface

class AppState:
    def __init__(self):
        self.thickness = INITIAL_THICKNESS
        self.resolution = RESOLUTION
        self.bound = INITIAL_BOUND
        self.mesh = None
        self.probe_point = (0.0, 0.0, 0.0)


def build_plotter():
    print("Initializing PyVista lattice viewer...")
    state = AppState()
    plotter = pv.Plotter(window_size=(1200, 760))
    plotter.title = "Gyroid Lattice Studio"
    plotter.set_background(color="#0f172a", top="#1e293b")
    plotter.add_axes(line_width=2)

    actor_ref = {"actor": None}
    probe_ref = {"actor": None, "label": None}
    def refresh_mesh():
        new_mesh = get_lattice_mesh(
            thickness=state.thickness,
            resolution=state.resolution,
            bound=state.bound,
        )
        state.mesh = new_mesh

        if actor_ref["actor"] is not None:
            plotter.remove_actor(actor_ref["actor"])

        actor_ref["actor"] = plotter.add_mesh(
            state.mesh,
            color="#14b8a6",
            show_edges=False,
            smooth_shading=True,
            specular=0.35,
            specular_power=20,
            ambient=0.2,
        )
        plotter.render()

    def render_equation():
        eq_data = format_equation(state.thickness, state.bound)
        if isinstance(eq_data, (tuple, list)) and len(eq_data) >= 2:
            eq_main, eq_constraint = str(eq_data[0]), str(eq_data[1])
        else:
            eq_main = str(eq_data)
            eq_constraint = f"|f(x,y,z)| <= {state.thickness:.3f}, x,y,z in [-{state.bound:.3f}, {state.bound:.3f}]"
        print(f"Equation: {eq_main}")
        print(f"Constraint: {eq_constraint}")

        plotter.add_text(
            eq_main,
            position=(15, 18),
            font_size=10,
            color="#cbd5e1",
            name="equation_main",
        )
        plotter.add_text(
            eq_constraint,
            position=(15, 4),
            font_size=10,
            color="#cbd5e1",
            name="equation_constraint",
        )
        plotter.render()

    def on_thickness_change(value):
        state.thickness = float(value)
        refresh_mesh()
        render_equation()

    def on_resolution_change(value):
        state.resolution = max(16, int(round(value)))
        refresh_mesh()

    def on_bound_change(value):
        state.bound = float(value)
        refresh_mesh()
        render_equation()

    def render_probe_point():
        x, y, z = state.probe_point
        value = gyroid_value(x, y, z)

        if probe_ref["actor"] is not None:
            plotter.remove_actor(probe_ref["actor"])
        if probe_ref["label"] is not None:
            plotter.remove_actor(probe_ref["label"])

        probe = pv.PolyData(np.array([[x, y, z]], dtype=float))
        probe_ref["actor"] = plotter.add_mesh(
            probe,
            color="#f97316",
            point_size=14,
            render_points_as_spheres=True,
        )
        probe_ref["label"] = plotter.add_point_labels(
            probe,
            [f"P({x:.2f}, {y:.2f}, {z:.2f})  f={value:.4f}"],
            font_size=12,
            text_color="#f8fafc",
            shape_color="#0f172a",
            fill_shape=True,
            margin=4,
            always_visible=True,
        )
        print(f"Probe point ({x:.4f}, {y:.4f}, {z:.4f}) -> f = {value:.6f}")
        plotter.render()

    def set_probe_point_from_console():
        raw = input("Enter probe point x,y,z: ").strip()
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 3:
            print("Invalid format. Use x,y,z")
            return
        try:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            print("Invalid numeric values.")
            return
        state.probe_point = (x, y, z)
        render_probe_point()

    def download_stl_file():
        filename = "gpu_lattice_output.stl"
        if state.mesh is None or state.mesh.n_points == 0:
            print("No mesh available to export.")
            return
        state.mesh.save(filename)
        print(f"Saved STL to ./{filename}")

    refresh_mesh()
    render_equation()
    render_probe_point()

    # Native PyVista sliders keep the UI in one rendering/event system.
    plotter.add_slider_widget(
        callback=on_thickness_change,
        rng=[0.05, 1.2],
        value=state.thickness,
        title="Thickness",
        pointa=(0.62, 0.16),
        pointb=(0.95, 0.16),
        style="modern",
    )
    plotter.add_slider_widget(
        callback=on_resolution_change,
        rng=[20, 110],
        value=state.resolution,
        title="Resolution",
        pointa=(0.62, 0.10),
        pointb=(0.95, 0.10),
        style="modern",
    )
    plotter.add_slider_widget(
        callback=on_bound_change,
        rng=[1.0, 5.0],
        value=state.bound,
        title="Domain Bound",
        pointa=(0.62, 0.04),
        pointb=(0.95, 0.04),
        style="modern",
    )

    plotter.add_key_event("d", download_stl_file)
    plotter.add_key_event("p", set_probe_point_from_console)
    plotter.add_text("Gyroid Lattice Studio", position="upper_left", font_size=16, color="#e2e8f0")
    plotter.add_text("D: export STL | P: set probe point", position="upper_right", font_size=10, color="#cbd5e1")
    plotter.view_isometric()
    return plotter


if __name__ == "__main__":
    viewer = build_plotter()
    print("Launching interactive window...")
    viewer.show()