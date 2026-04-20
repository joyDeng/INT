
# this file is used to visualize the mesh
# you need to install paraview, add path of pvpython.exe to the path and run pvpython paraview.py to render
import sys
from constant import TEMP_DIR
# to run this file use command pvpython paramview

# add paraview python package
# sys.path.insert(0, "C:\Program Files\ParaView 5.13.3\\bin")
# sys.path.insert(0, "C:\Program Files\ParaView 5.13.3\\bin\Lib")
# sys.path.insert(0, "C:\Program Files\ParaView 5.13.3\\bin\Lib\site-packages")


# print("PYTHONPATH:", sys.path)
from paraview.simple import *

Disconnect()
Connect()

# Create render view once
render_view = GetActiveViewOrCreate('RenderView')
LoadPalette("WhiteBackground")
render_view.CameraPosition = [3.2, 1.0, 3.0]
render_view.CameraFocalPoint = [0.0, 0.0, 0.0]

# Make sure output dir exists
# os.makedirs(os.path.join(TEMP_DIR, "screen"), exist_ok=True)

# Build pipeline once with "dummy" first frame (or the first real frame)
def init_pipeline(first_id, name):
    r0 = PLYReader(FileNames=[TEMP_DIR + f"/opt_sensor_mesh_3_mesh_C_iter0_csg.ply"])
    r1 = PLYReader(FileNames=[TEMP_DIR + f"/{name}_D_iter{first_id}.ply"])

    c1 = Clip(Input=r1)
    c1.ClipType = 'Plane'
    c1.ClipType.Origin = [0.0, 0.0, 0.0]
    c1.ClipType.Normal = [1.0, 0.0, 0.0]

    c0 = Clip(Input=r0)
    c0.ClipType = 'Plane'
    c0.ClipType.Origin = [0.0, 0.0, 0.0]
    c0.ClipType.Normal = [1.0, 0.0, 0.0]

    s1 = Show(c1, render_view)
    s0 = Show(c0, render_view)
    s0.DiffuseColor = [233 / 255.0, 196 / 255.0, 106 / 255.0]  # gold
    s1.DiffuseColor = [42  / 255.0, 157 / 255.0, 143 / 255.0]  # teal

    render_view.Update()
    return r0, r1

ply_reader_0, ply_reader_1 = init_pipeline(first_id=0, name="opt_sensor_mesh_vol_3")

def render_frame(frame_id, name, id):
    # update file names only
    ply_reader_0.FileNames = [TEMP_DIR + f"/opt_sensor_mesh_3_mesh_C_iter0_csg.ply"]
    ply_reader_1.FileNames = [TEMP_DIR + f"/{name}_D_iter{frame_id}.ply"]

    ply_reader_0.UpdatePipeline()
    ply_reader_1.UpdatePipeline()
    render_view.Update()

    SaveScreenshot(
        TEMP_DIR + f"/screen/{name}_{id:03d}_mid_plane.png",
        render_view,
        ImageResolution=[1080, 1080],
    )

if __name__ == "__main__":
    for ii in range(200):
        iteration = ii * 10
        # render_frame(iteration, "opt_sensor_<built-in function id>", ii)
        render_frame(iteration, "opt_sensor_mesh_vol_3", ii)