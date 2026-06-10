Hello everyone :)

I hope you have a good day. This is a framework for 3D and 2D demonstration.

The framework is mainly based on PyQt5 and Pyvista. 

## External AI assets

Large AI datasets and checkpoints should live outside the Python package.  By
default the app still uses `phd/resource/ai`, but you can point it at an
external asset folder:

```bash
export PINGLAB_AI_RESOURCE_ROOT=/home/ping2/phd_assets/ai
```

Expected layout:

```text
/home/ping2/phd_assets/ai/
  data/
  models/
```

You can also point at a full resource mirror; the app will use its `ai`
subfolder:

```bash
export PINGLAB_RESOURCE_ROOT=/home/ping2/phd_assets
```

`PINGLAB_AI_RESOURCE_ROOT` takes priority when both variables are set.

I would like to add more functions here to make it more helpful to you.

The plan is described below:

1. A 3D demo widget with a model tree to set the visibility of features for the models

2. A 2D combo widget for knitting map demonstration.

3. A tool to fully automatically generate the scene for the blender.

4. A robot simulation and real-time controller

5. A library to govern the topology of different geometries.

6. ...
Please let me know if you have any suggestions, hopefully, we can make a earth-shake framework!
