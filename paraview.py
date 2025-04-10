
# this file is used to visualize the mesh
# you need to install paraview, add path of pvpython.exe to the path and run pvpython paraview.py to render
import sys

# add paraview python package
sys.path.insert(0, "C:\Program Files\ParaView 5.13.3\\bin")
sys.path.insert(0, "C:\Program Files\ParaView 5.13.3\\bin\Lib")
sys.path.insert(0, "C:\Program Files\ParaView 5.13.3\\bin\Lib\site-packages")


# print("PYTHONPATH:", sys.path)
from paraview.simple import *

def ResetSession():
    pxm = servermanager.ProxyManager()
    pxm.UnRegisterProxies()
    del pxm
    Disconnect()
    Connect()

def render_frame(id):
    ResetSession()
    ply_reader = PLYReader(FileNames=f"../Opt/A_iter{id}_csg.ply")
    obj_reader = WavefrontOBJReader(FileName='./scene/torusC.obj')
    # Show(ply_reader)
    # Render()
    # # ply = PLYReader("../Opt/A_iter0_csg.ply")
    # WriteImage("../Opt/A_iter0_csg.png")

    # Apply a clip filter
    clip1 = Clip(Input=ply_reader)
    clip1.ClipType = 'Plane'
    clip1.ClipType.Origin = [0.0, 0.0, 0.0]
    clip1.ClipType.Normal = [0.0, 1.0, 0.0]

    # Update the pipeline
    clip1.UpdatePipeline()

    # Apply a clip filter
    clip2 = Clip(Input=obj_reader)
    clip2.ClipType = 'Plane'
    clip2.ClipType.Origin = [0.0, 0.0, 0.0]
    clip2.ClipType.Normal = [0.0, 1.0, 0.0]

    # Update the pipeline
    clip2.UpdatePipeline()

    # clip1.UpdatePipeline()
    # Load your data source
    # data_source = OpenDataFile("../Opt/A_iter0_csg.ply")


    # Create a render view
    render_view = GetActiveViewOrCreate('RenderView')

    # Display the data in the render view
    # data_display = Show(ply_reader, render_view)
    # Show(obj_reader, render_view)
    show1 = Show(clip1, render_view)
    show2 = Show(obj_reader, render_view)
    show2.DiffuseColor = [145 / 255.0, 198 / 255.0, 181 / 255.0] 
    show1.DiffuseColor = [211 / 255.0, 211 / 255.0, 188 / 255.0]

    # Update the view to ensure updated data information
    render_view.Update()
    LoadPalette(paletteName="WhiteBackground")
    # render_view.Background = [1.0, 1.0, 1.0]
    # Set the camera position (optional)
    render_view.CameraPosition = [5.0, 10.0, 0.0]
    render_view.CameraFocalPoint = [0.0, 0.0, 0.0]

    # Save the screenshot
    # SaveScreenshot("path/to/save/screenshot.png", render_view)

    # Optionally, you can specify image resolution
    SaveScreenshot(f"../Opt/A_iter_volume_constraints{id:03d}_csg_h.png", render_view, ImageResolution=[1920, 1080])

    # If you want to close ParaView after saving the screenshot
    # Disconnect()

if __name__ == "__main__":
    for ii in range(200):
        render_frame(ii)