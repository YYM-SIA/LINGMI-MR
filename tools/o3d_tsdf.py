import os
import sys
import numpy as np
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from tools.o3d_visual import npxyzw_o3d


class TSDFWindow:
    AXIS = ['X', 'Y', 'Z']

    def __init__(self, sdf: np.ndarray, width=1024, height=768):

        # Settings
        self.bg_color = gui.Color(1, 1, 1)
        self.show_skybox = True
        self.show_axes = False
        self.use_sdf = True
        self.slice_axis = 0
        self.slice_index = 0

        sdf = sdf.squeeze()
        X, Y, Z = sdf.shape[:3]
        # assert(X == Y == Z)

        self.window = gui.Application.instance.create_window("Open3D", width, height)
        w = self.window  # to make the code more concise

        # 3D widget
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(w.renderer)

        # ## Make panel layout
        em = w.theme.font_size

        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        advanced = gui.CollapsableVert("Advanced lighting", 0,
                                       gui.Margins(em, 0, 0, 0))

        self._use_sdf = gui.Checkbox("SDF")
        self._use_sdf.set_on_checked(self._on_use_sdf)
        advanced.add_child(gui.Label("Use SDF"))
        h = gui.Horiz(em)
        h.add_child(self._use_sdf)
        advanced.add_child(h)

        self._ui_slice_index = gui.Slider(gui.Slider.INT)
        self._ui_slice_index.set_limits(0, np.min(sdf.shape[:3]) - 1) # FIXME
        self._ui_slice_index.set_on_value_changed(self._on_slice_index)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(self._ui_slice_index)
        advanced.add_child(gui.Label("Slice"))
        advanced.add_child(grid)

        advanced.add_child(gui.Label("Axis"))
        self._profiles = gui.Combobox()
        for name in self.AXIS:
            self._profiles.add_item(name)
        self._profiles.set_on_selection_changed(self._on_slice_axis)
        advanced.add_child(self._profiles)

        self._settings_panel.add_child(advanced)

        # ## Add scene
        w.set_on_layout(self._on_layout)
        w.add_child(self.scene)
        w.add_child(self._settings_panel)

        # ## Add SDF and ajust camera
        if len(sdf.shape) == 4:
            assert sdf.shape[-1] == 4
        else:
            x_ = np.arange(0, X)
            y_ = np.arange(0, Y)
            z_ = np.arange(0, Z)
            x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
            sdf = np.stack([x, y, z, sdf], axis=-1)

        points = sdf[:, :, :, :3].reshape((-1, 3))
        self.sdf = sdf

        center = np.mean(points, axis=0)
        outsize = 1.5 * (np.max(points, axis=0) - center) + center
        self.scene.scene.camera.look_at(center, outsize, [0, 0, 1])

        self._apply_settings()
        self.update_sdf()

    def update_sdf(self):
        if self.scene.scene.has_geometry('mesh'):
            self.scene.scene.remove_geometry('mesh')

        if self.use_sdf:
            pcd = npxyzw_o3d(self.sdf.reshape((-1, 4)))
            self.scene.scene.add_geometry('mesh', pcd, rendering.Material())
        else:
            axis, id = self.slice_axis, self.slice_index
            index = [slice(None)] * 3
            index[axis] = id
            pcd = npxyzw_o3d(self.sdf[tuple(index)].reshape((-1, 4)))
            self.scene.scene.add_geometry('mesh', pcd, rendering.Material())

    def _apply_settings(self):
        bg_color = [
            self.bg_color.red, self.bg_color.green,
            self.bg_color.blue, self.bg_color.alpha
        ]
        self.scene.scene.set_background(bg_color)
        self.scene.scene.show_skybox(self.show_skybox)
        self.scene.scene.show_axes(self.show_axes)

        self._use_sdf.checked = self.use_sdf
        self._ui_slice_index.int_value = self.slice_index

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self.scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(r.height,
                     self._settings_panel.calc_preferred_size(
                         layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)

    def _on_use_sdf(self, use):
        self.use_sdf = use
        self._apply_settings()
        self.update_sdf()

    def _on_slice_index(self, intensity):
        self.slice_index = int(intensity)
        self._apply_settings()
        if not self.use_sdf:
            self.update_sdf()

    def _on_slice_axis(self, name, index):
        if name in self.AXIS:
            self.slice_index = 0
            self.slice_axis = index
        self._apply_settings()
        if not self.use_sdf:
            self.update_sdf()


def run_tsdf_windows(sdf, width=1024, height=768, models = None):
    # We need to initalize the application, which finds the necessary shaders
    # for rendering and prepares the cross-platform window abstraction.
    gui.Application.instance.initialize()

    w = TSDFWindow(sdf, width, height)
    for i, m in enumerate(models):
        w.scene.scene.add_geometry(f'model-{i}', m, rendering.Material())
    
    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()


if __name__ == "__main__":
    sdf = np.load("data/sdf.npy")
    run_tsdf_windows(sdf)
