import math

import numpy as np
import pyvista as pv

# --- init config ---
RESOLUTION = 96
INITIAL_THICKNESS = 0.4
INITIAL_BOUND = 2 * math.pi

EQUATIONS = {
    "gyroid": {
        "title": "Gyroid",
        "function": lambda x, y, z: (
            math.sin(x) * math.cos(y)
            + math.sin(y) * math.cos(z)
            + math.sin(z) * math.cos(x)
        ),
    },
    "diamond": {
        "title": "Diamond",
        "function": lambda x, y, z: (
            math.sin(x) * math.sin(y) * math.sin(z)
            + math.sin(x) * math.cos(y) * math.cos(z)
            + math.cos(x) * math.sin(y) * math.cos(z)
            + math.cos(x) * math.cos(y) * math.sin(z)
        ),
    },
    "sine_sum": {
        "title": "sin(x)+sin(y)+sin(z)=0",
        "function": lambda x, y, z: math.sin(x) + math.sin(y) + math.sin(z),
    },
    "sphere": {
        "title": "Sphere",
        "function": lambda x, y, z: x * x + y * y + z * z - 4.0,
    },
}


def get_equation_value(equation_key, x, y, z):
    return EQUATIONS[equation_key]["function"](x, y, z)


def get_lattice_mesh(equation_key, thickness, resolution, bound):
    # Generate a solid implicit band and return its surface mesh.
    x_min, x_max = -bound, bound
    y_min, y_max = -bound, bound
    z_min, z_max = -bound, bound

    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    z = np.linspace(z_min, z_max, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    if equation_key == "gyroid":
        values = (
            np.sin(X) * np.cos(Y)
            + np.sin(Y) * np.cos(Z)
            + np.sin(Z) * np.cos(X)
        )
    elif equation_key == "diamond":
        values = (
            np.sin(X) * np.sin(Y) * np.sin(Z)
            + np.sin(X) * np.cos(Y) * np.cos(Z)
            + np.cos(X) * np.sin(Y) * np.cos(Z)
            + np.cos(X) * np.cos(Y) * np.sin(Z)
        )
    elif equation_key == "sine_sum":
        values = np.sin(X) + np.sin(Y) + np.sin(Z)
    elif equation_key == "sphere":
        radius = bound * 0.55
        values = X * X + Y * Y + Z * Z - radius * radius
    else:
        raise ValueError(f"Unknown equation: {equation_key}")

    grid = pv.ImageData(dimensions=(resolution, resolution, resolution))
    grid.spacing = ((x_max - x_min) / (resolution - 1),
                    (y_max - y_min) / (resolution - 1),
                    (z_max - z_min) / (resolution - 1))
    grid.origin = (x_min, y_min, z_min)

    grid.point_data["values"] = values.flatten(order="F")
    grid.point_data["band"] = (np.abs(values) <= thickness).astype(np.uint8).flatten(order="F")

    # Extract the filled band volume, then smooth the outer surface so it stays solid without the staircase look.
    band_volume = grid.threshold(value=0.5, scalars="band")
    smooth_surface = band_volume.extract_surface(algorithm="dataset_surface").triangulate().clean()
    smooth_surface = smooth_surface.smooth_taubin(n_iter=20, pass_band=0.1)
    return smooth_surface

class AppState:
    def __init__(self):
        self.thickness = INITIAL_THICKNESS
        self.resolution = RESOLUTION
        self.bound = INITIAL_BOUND
        self.equation_key = "gyroid"
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
    probe_ref = {"actor": None}
    def refresh_mesh():
        new_mesh = get_lattice_mesh(
            equation_key=state.equation_key,
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

    def on_thickness_change(value):
        state.thickness = float(value)
        refresh_mesh()

    def on_resolution_change(value):
        state.resolution = max(24, int(round(value)))
        refresh_mesh()

    def on_bound_change(value):
        step = math.pi / 2
        state.bound = max(step, round(float(value) / step) * step)
        refresh_mesh()

    def set_equation(equation_key):
        if state.equation_key == equation_key:
            return
        state.equation_key = equation_key
        refresh_mesh()
        render_probe_point()

    def render_probe_point():
        x, y, z = state.probe_point
        value = get_equation_value(state.equation_key, x, y, z)

        if probe_ref["actor"] is not None:
            plotter.remove_actor(probe_ref["actor"])

        probe = pv.PolyData(np.array([[x, y, z]], dtype=float))
        probe_ref["actor"] = plotter.add_mesh(
            probe,
            color="#f97316",
            point_size=14,
            render_points_as_spheres=True,
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
        rng=[24, 220],
        value=state.resolution,
        title="Resolution",
        pointa=(0.62, 0.10),
        pointb=(0.95, 0.10),
        style="modern",
    )
    plotter.add_slider_widget(
        callback=on_bound_change,
        rng=[math.pi, 8 * math.pi],
        value=state.bound,
        title="Domain Bound (0.5π)",
        pointa=(0.62, 0.04),
        pointb=(0.95, 0.04),
        style="modern",
    )

    plotter.add_radio_button_widget(
        callback=lambda: set_equation("gyroid"),
        radio_button_group="equation",
        value=True,
        title="Gyroid",
        position=(18, 150),
        size=24,
        border_size=4,
        color_on="#14b8a6",
        color_off="#475569",
    )
    plotter.add_radio_button_widget(
        callback=lambda: set_equation("diamond"),
        radio_button_group="equation",
        value=False,
        title="Diamond",
        position=(18, 112),
        size=24,
        border_size=4,
        color_on="#14b8a6",
        color_off="#475569",
    )
    plotter.add_radio_button_widget(
        callback=lambda: set_equation("sine_sum"),
        radio_button_group="equation",
        value=False,
        title="sin(x)+sin(y)+sin(z)",
        position=(18, 74),
        size=24,
        border_size=4,
        color_on="#14b8a6",
        color_off="#475569",
    )
    plotter.add_radio_button_widget(
        callback=lambda: set_equation("sphere"),
        radio_button_group="equation",
        value=False,
        title="Sphere",
        position=(18, 36),
        size=24,
        border_size=4,
        color_on="#14b8a6",
        color_off="#475569",
    )

    plotter.add_key_event("d", download_stl_file)
    plotter.add_key_event("p", set_probe_point_from_console)
    plotter.view_isometric()
    return plotter


if __name__ == "__main__":
    viewer = build_plotter()
    print("Launching interactive window...")
    viewer.show()