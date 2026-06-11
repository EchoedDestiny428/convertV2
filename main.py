import math
import numpy as np
import pyvista as pv

PI = math.pi
EQUATIONS = {
    "gyroid": ("Gyroid", lambda X, Y, Z: np.sin(X) * np.cos(Y) + np.sin(Y) * np.cos(Z) + np.sin(Z) * np.cos(X)),
    "diamond": ("Diamond", lambda X, Y, Z: np.sin(X) * np.sin(Y) * np.sin(Z) + np.sin(X) * np.cos(Y) * np.cos(Z) + np.cos(X) * np.sin(Y) * np.cos(Z) + np.cos(X) * np.cos(Y) * np.sin(Z)),
    "sine_sum": ("sin(x)+sin(y)+sin(z)=0", lambda X, Y, Z: np.sin(X) + np.sin(Y) + np.sin(Z)),
    "sphere": ("Sphere", lambda X, Y, Z: X * X + Y * Y + Z * Z - 4.0),
}

def get_mesh(eq_key, thickness, res, bound):
    coords = np.linspace(-bound, bound, res)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    values = EQUATIONS[eq_key][1](X, Y, Z)
    
    grid = pv.ImageData(dimensions=(res, res, res))
    grid.spacing = (2 * bound / (res - 1),) * 3
    grid.origin = (-bound,) * 3
    grid.point_data["values"] = values.flatten(order="F")
    grid.point_data["band"] = (np.abs(values) <= thickness).astype(np.uint8).flatten(order="F")
    
    surface = grid.threshold(value=0.5, scalars="band").extract_surface(algorithm="dataset_surface")
    return surface.triangulate().clean().smooth_taubin(n_iter=20, pass_band=0.1)


def build_plotter():
    plotter = pv.Plotter(window_size=(1200, 760))
    plotter.set_background(color="#0f172a", top="#1e293b")
    plotter.add_axes(line_width=2)
    
    state = {
        "eq": "gyroid",
        "thickness": 0.4,
        "resolution": 96,
        "bound": 2 * PI,
        "mesh": None,
    }
    
    actor_ref = [None]
    
    def refresh():
        state["mesh"] = get_mesh(state["eq"], state["thickness"], state["resolution"], state["bound"])
        if actor_ref[0]:
            plotter.remove_actor(actor_ref[0])
        actor_ref[0] = plotter.add_mesh(state["mesh"], color="#14b8a6", smooth_shading=True, specular=0.35, specular_power=20, ambient=0.2)
        plotter.render()
    
    plotter.add_slider_widget(lambda v: (state.update({"thickness": float(v)}), refresh()), rng=[0.05, 1.2], value=0.4, title="Thickness", pointa=(0.62, 0.16), pointb=(0.95, 0.16), style="modern")
    plotter.add_slider_widget(lambda v: (state.update({"resolution": max(24, int(v))}), refresh()), rng=[24, 220], value=96, title="Resolution", pointa=(0.62, 0.10), pointb=(0.95, 0.10), style="modern")
    plotter.add_slider_widget(lambda v: (state.update({"bound": max(PI/2, round(float(v) / (PI/2)) * (PI/2))}), refresh()), rng=[PI, 8*PI], value=2*PI, title="Bound (0.5π)", pointa=(0.62, 0.04), pointb=(0.95, 0.04), style="modern")
    
    positions = [(18, 150), (18, 112), (18, 74), (18, 36)]
    for idx, (eq_key, (title, _)) in enumerate(EQUATIONS.items()):
        plotter.add_radio_button_widget(
            lambda e=eq_key: (state.update({"eq": e}), refresh()),
            radio_button_group="eq",
            value=(idx == 0),
            title=title,
            position=positions[idx],
            size=24,
            border_size=4,
            color_on="#14b8a6",
            color_off="#475569",
        )
    
    plotter.add_key_event("d", lambda: state["mesh"].save("output.stl") if state["mesh"] else None)
    refresh()
    plotter.view_isometric()
    return plotter


if __name__ == "__main__":
    viewer = build_plotter()
    viewer.show()