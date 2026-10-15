"""Build an obstacle room around the supplied Waffle Pi; no control decisions."""

import xml.etree.ElementTree as ET
import numpy as np
import cv2
from .config import ROOT


def build_scene(config, trial):
    root = ET.parse(ROOT / "models/robotis_tb3/turtlebot3_waffle_pi.xml").getroot()
    root.find("compiler").set("meshdir", str(ROOT / "models/robotis_tb3/assets"))
    option = root.find("option")
    option.set("timestep", str(config.physics_dt))
    option.set("integrator", "implicitfast")
    ET.SubElement(root, "statistic", center="0 0 0.25", extent="4.5")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "global",
        offwidth=str(config.camera_width),
        offheight=str(config.camera_height),
        azimuth="125",
        elevation="-65",
    )
    ET.SubElement(visual, "headlight", ambient="0.5 0.5 0.5", diffuse="0.5 0.5 0.5")
    ET.SubElement(visual, "quality", shadowsize="1024")
    asset = root.find("asset")
    ET.SubElement(
        asset,
        "material",
        name="marker_black",
        rgba="0 0 0 1",
        emission="1",
        specular="0",
    )
    ET.SubElement(
        asset,
        "material",
        name="marker_white",
        rgba="1 1 1 1",
        emission="1",
        specular="0",
    )
    world = root.find("worldbody")
    base = world.find("body[@name='base']")
    base.set("pos", f"{config.start[0]} {config.start[1]} 0")
    base.set("quat", f"{np.cos(config.start[2] / 2)} 0 0 {np.sin(config.start[2] / 2)}")
    base.find("camera").set("pos", " ".join(map(str, config.camera_pos)))
    base.find("camera").set("fovy", str(config.camera_fovy))
    base.find("site[@name='lidar']").set("pos", " ".join(map(str, config.lidar_pos)))
    ET.SubElement(
        world,
        "light",
        pos="0 0 3",
        dir="0 0 -1",
        directional="true",
        castshadow="false",
    )
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="4 3 0.1",
        rgba="0.78 0.81 0.84 1",
        group="1",
    )
    x0, x1, y0, y1 = config.bounds
    boxes = [
        ("wall_left", (x0 - 0.05, 0, 0.3), (0.05, (y1 - y0) / 2, 0.3)),
        ("wall_right", (x1 + 0.05, 0, 0.3), (0.05, (y1 - y0) / 2, 0.3)),
        ("wall_bottom", (0, y0 - 0.05, 0.3), ((x1 - x0) / 2, 0.05, 0.3)),
        ("wall_top", (0, y1 + 0.05, 0.3), ((x1 - x0) / 2, 0.05, 0.3)),
        ("obstacle_a", (-0.40, -0.45 + trial.obstacle_shift, 0.22), (0.20, 0.25, 0.22)),
        ("obstacle_b", (0.35, 0.85, 0.22), (0.22, 0.20, 0.22)),
        ("charging_station", (1.72, 0, 0.23), (0.115, 0.23, 0.23)),
    ]
    for name, pos, size in boxes:
        ET.SubElement(
            world,
            "geom",
            name=name,
            type="box",
            pos=" ".join(map(str, pos)),
            size=" ".join(map(str, size)),
            rgba="0.30 0.40 0.49 1",
            group="0",
        )
    # Draw the actual ArUco code as unlit square cells in the 3-D scene.
    # The camera must see and detect these pixels; no pose is fabricated.
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    bits = cv2.aruco.generateImageMarker(dictionary, config.marker_id, 6)
    mx, my, mz = config.marker_pos
    ET.SubElement(
        world,
        "geom",
        name="marker_board",
        type="box",
        pos=f"{mx + 0.003} {my} {mz}",
        size=f".002 {config.marker_side * 0.65} {config.marker_side * 0.65}",
        material="marker_white",
        contype="0",
        conaffinity="0",
        group="2",
    )
    cell = config.marker_side / 6
    for row in range(6):
        for col in range(6):
            y = my - (col - 2.5) * cell
            z = mz + (2.5 - row) * cell
            ET.SubElement(
                world,
                "geom",
                name=f"marker_{row}_{col}",
                type="box",
                pos=f"{mx} {y} {z}",
                size=f".0002 {cell / 2} {cell / 2}",
                material="marker_white" if bits[row, col] else "marker_black",
                contype="0",
                conaffinity="0",
                group="2",
            )
    return ET.tostring(root, encoding="unicode")
