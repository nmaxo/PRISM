"""Presentation-only callback: USD materials, reference goal and actual actor depth."""
from pathlib import Path
import hashlib
import numpy as np
from holosoma.agents.callbacks.base_callback import RLEvalCallback


def depth_rgba(depth):
    """Fixed scale for the checkpoint's normalized depth [-.5, .5]."""
    x = np.clip(np.asarray(depth).reshape(58, 87) + 0.5, 0, 1)
    near, mid, far = np.array([255, 207, 99]), np.array([42, 185, 188]), np.array([17, 25, 48])
    rgb = np.where((x < .5)[..., None], near + (mid-near)*x[..., None]*2,
                   mid + (far-mid)*(x[..., None]-.5)*2)
    return np.concatenate([rgb.astype(np.uint8), np.full((58, 87, 1), 255, np.uint8)], axis=-1)


class Demo(RLEvalCallback):
    def on_pre_evaluate_policy(self):
        import omni.usd
        import omni.ui as ui
        from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import POSITION_GOAL_MARKER_CFG
        self.env = self.training_loop.env
        self.command = self.env.command_manager.get_state('motion_command')
        self.stage = omni.usd.get_context().get_stage()
        stage = self.stage

        def material(name, color, metallic=0., roughness=.6):
            mat = UsdShade.Material.Define(stage, '/World/PrismDemo/Materials/' + name)
            shader = UsdShade.Shader.Define(stage, str(mat.GetPath()) + '/Shader')
            shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metallic)
            shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
            return mat

        source_path = Path(__file__).parent / 'demo_assets/unitree_g1_base.usd'
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == 'd9768c942783ae0932f0c3db3558d5283b9a93ba0e561d15d8984f548aa2a65d'
        source = Usd.Stage.Open(str(source_path))
        colors = {}
        for prim in source.Traverse():
            if prim.IsA(UsdGeom.Mesh) and str(prim.GetPath()).startswith('/visuals/'):
                binding, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                for shader in binding.GetPrim().GetChildren():
                    value = shader.GetAttribute('inputs:diffuse_color_constant').Get()
                    if value is not None:
                        colors[str(prim.GetPath()).split('/')[-2]] = tuple(value)
                        colors.setdefault(str(prim.GetPath()).split('/')[2], tuple(value))
        official = {link: material('Unitree_' + link, color, .35, .32) for link, color in colors.items()}
        robot_count = 0
        debug = stage.GetPrimAtPath('/Visuals')
        if debug:
            UsdGeom.Imageable(debug).MakeInvisible()
        chessboard = stage.GetPrimAtPath('/World/ground_chessboard')
        if chessboard:
            UsdGeom.Imageable(chessboard).MakeInvisible()
        for prim in list(stage.Traverse()):
            if '/Robot/' in str(prim.GetPath()) and '/visuals' in str(prim.GetPath()) and prim.IsInstance():
                prim.SetInstanceable(False)
        for prim in list(stage.Traverse()):
            path = str(prim.GetPath())
            if '/Robot/' in path and '/visuals' in path and prim.IsA(UsdGeom.Mesh):
                link = next((part for part in reversed(path.split('/')) if part in official), None)
                if link in official:
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(official[link], bindingStrength=UsdShade.Tokens.strongerThanDescendants)
                    robot_count += 1
        assert robot_count, 'No robot visual materials were bound'
        box_color = (self.config or {}).get('box_color')
        if box_color is not None:
            box_prim = stage.GetPrimAtPath('/World/envs/env_0/Object')
            assert box_prim, 'Box render prim is missing'
            for prim in list(Usd.PrimRange(box_prim)):
                if '/visuals' in str(prim.GetPath()) and prim.IsInstance():
                    prim.SetInstanceable(False)
            box_material = material('BoxColor', box_color)
            meshes = [prim for prim in Usd.PrimRange(box_prim)
                      if '/visuals/' in str(prim.GetPath()) and prim.IsA(UsdGeom.Mesh)]
            assert meshes, 'Box visual mesh is missing'
            for prim in meshes:
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    box_material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
            print(f'DEMO_BOX_COLOR {box_color}', flush=True)
        # Warp geometry was built before this display-only marker. No collision schemas are added.
        offset = self.command._get_env_offsets()[0].detach().cpu().numpy()
        reference = self.command.motion.object_pos_w.detach().cpu().numpy() + offset
        self.goal = reference[-1].copy()
        self.goal_marker = VisualizationMarkers(
            POSITION_GOAL_MARKER_CFG.replace(prim_path='/World/Demo/Goal'))
        self.goal_marker.visualize(translations=self.goal[None],
                                   scales=np.array([[6., 6., 6.]]), marker_indices=[0])
        # Frame the whole route, leaving room to inspect both pickup and destination.
        center = (reference[0] + reference[-1]) / 2
        viewer = self.env.simulator.viewport_camera_controller
        viewer.update_view_to_world()
        viewer.update_view_location(eye=center + np.array([1.5, -4.8, 3.2]),
                                    lookat=center + np.array([0., 0., .4]))
        self.provider = ui.ByteImageProvider()
        for title in ('Stage', 'Layer', 'Render Settings', 'Property', 'Content', 'Console', 'Semantics Schema Editor'):
            panel = ui.Workspace.get_window(title)
            if panel:
                panel.visible = False
        self.window = ui.Window('Depth camera', width=380, height=450)
        self.window.position_x = 20
        self.window.position_y = 90
        with self.window.frame:
            with ui.VStack(spacing=10, style={'background_color': 0xFF20170F, 'color': 0xFFF0E6DE, 'font_size': 16}):
                ui.Label('UNITREE G1', height=30, style={'font_size': 23})
                ui.Label('LIVE DEPTH', height=22)
                ui.ImageWithProvider(self.provider, width=348, height=232)
                ui.Label('0.3 m  /  amber - teal - navy  /  3.0 m', height=22, style={'font_size': 13})
                self.status = ui.Label('', height=25)
                ui.Label('Goal marker: red = far, green = within 15 cm', height=22, style={'font_size': 13})
        print(f'PRISM_DEMO_READY floor=isaac_default robot_materials={robot_count} goal={self.goal.tolist()}', flush=True)
        self.record_path = (self.config or {}).get('record')
        if self.record_path:
            import carb
            import tempfile
            import omni.renderer_capture
            self.settings = carb.settings.get_settings()
            self.settings.set_int('/rtx/post/aa/op', 4)  # Native RTX setting: DLAA.
            self.settings.set_bool('/rtx-transient/dlssg/enabled', False)
            assert self.settings.get('/rtx-transient/post/dlss/supported'), 'DLAA is not supported by this renderer'
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            self.frames = Path(tempfile.mkdtemp(prefix='dlaa_frames_', dir=self.record_path.parent))
            self.capture = omni.renderer_capture.acquire_renderer_capture_interface()
            self.capture_times = []
            for _ in range(12):
                self.env.simulator.sim.render()  # Warm up temporal AA without advancing physics.
            print(f'DLAA_RECORD_READY output={self.record_path} fps={1/self.env.dt:g}', flush=True)

    def on_pre_eval_env_step(self, actor_state):
        # Read exactly the already-computed policy tensor, including sensor noise and latency.
        if self.record_path or actor_state['step'] % 2 == 0:
            depth = actor_state['obs']['perception_obs'][0].detach().cpu().numpy()
            pixels = depth_rgba(depth)
            self.provider.set_bytes_data(pixels.flatten().tolist(), [87, 58])
            obj = self.command.simulator_object_pos_w[0].detach().cpu().numpy()
            distance = np.linalg.norm(obj[:2] - self.goal[:2])
            self.goal_marker.visualize(marker_indices=[int(distance < .15)])
            self.status.text = f'Object to goal  {distance:.2f} m    |    height  {obj[2]:.2f} m'
        if self.record_path:
            step = actor_state['step']
            assert self.settings.get('/rtx/post/aa/op') == 4
            sim_time = self.env.simulator.time()
            frame = self.frames / f'{step:06d}.png'
            self.capture.capture_next_frame_swapchain(str(frame))
            self.env.simulator.sim.render()
            self.capture.wait_async_capture()
            assert frame.is_file(), f'Missing captured frame: {frame}'
            assert abs(self.env.simulator.time() - sim_time) < 1e-8, 'Capture advanced physics'
            self.capture_times.append(sim_time)
            if step % 100 == 0:
                print(f'DLAA_RECORD_FRAME {step}', flush=True)
        return actor_state

    def on_post_evaluate_policy(self):
        if self.record_path:
            import json
            import shutil
            import subprocess
            fps = 1 / self.env.dt
            assert np.allclose(np.diff(self.capture_times), self.env.dt, atol=1e-5), 'Irregular simulation timing'
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n',
                            '-framerate', f'{fps:g}', '-i', str(self.frames / '%06d.png'),
                            '-vf', 'scale=1920:-2:flags=lanczos,pad=1920:1080:(ow-iw)/2:(oh-ih)/2',
                            '-c:v', 'libx264', '-crf', '17', '-pix_fmt', 'yuv420p',
                            '-movflags', '+faststart', str(self.record_path)], check=True)
            self.record_path.with_suffix('.json').write_text(json.dumps({
                'antialiasing': 'DLAA', 'rtx_post_aa_op': 4, 'fps': fps,
                'frames': len(self.capture_times), 'duration_seconds': len(self.capture_times)/fps,
                'playback_speed': 1.0, 'depth': 'actual policy input; application UI capture',
                'simulation_timestamps': self.capture_times}, indent=2))
            shutil.rmtree(self.frames)
            print(f'DLAA_VIDEO_COMPLETE {self.record_path}', flush=True)
        print('PRISM_DEMO_COMPLETE', flush=True)


if __name__ == '__main__':
    pixels = depth_rgba(np.linspace(-.5, .5, 58*87))
    assert pixels.shape == (58, 87, 4) and pixels.dtype == np.uint8
    assert pixels[0, 0].tolist() == [255, 207, 99, 255]
    assert pixels[-1, -1].tolist() == [17, 25, 48, 255]
    print('PASS: fixed-range depth visualization')
