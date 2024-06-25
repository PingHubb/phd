
import os
import bpy
import datetime

bpy.ops.mesh.primitive_cube_add(size=4)

cube_obj = bpy.context.active_object
loc = cube_obj.location
cube_obj.location.x = 5
now = datetime.datetime.now(datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo)
result = now.strftime('%Y_%m%d_%H%M_%Z')
bpy.ops.wm.save_as_mainfile(filepath = os.getcwd() + r"\results\\" + result + r".blend")
# bpy.ops.wm.open_mainfile(filepath=bpy.data.filepath)