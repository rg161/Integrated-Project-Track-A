import mujoco

PATH = "models/robotis_tb3/scene_turtlebot3_waffle_pi.xml"

model = mujoco.MjModel.from_xml_path(PATH)
data = mujoco.MjData(model)

print("loaded OK")
print("nq, nv, nu:", model.nq, model.nv, model.nu)
print("actuators:", [model.actuator(i).name for i in range(model.nu)])
print("ctrlrange:", model.actuator_ctrlrange.tolist())
print("joints:", [(model.joint(i).name, int(model.jnt_type[i])) for i in range(model.njnt)])
print("sensors:", [model.sensor(i).name for i in range(model.nsensor)])
print("cameras:", [model.camera(i).name for i in range(model.ncam)])
print("meshes:", model.nmesh)

start = data.qpos[:3].copy()
data.ctrl[:] = 1.0
while data.time < 2.0:
    mujoco.mj_step(model, data)
print("moved from", start, "to", data.qpos[:3])