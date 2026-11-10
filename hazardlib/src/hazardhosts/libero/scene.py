"""Add explicitly configured hazard bodies to an original LIBERO MJCF in memory.

The original BDDL, objects, robot, cameras and success predicates are retained.
Bodies are appended, so existing joint addresses are preserved. Original states
are still mapped by joint name, with the added joints parked by the host.
"""
import xml.etree.ElementTree as ET


def joint_layout(model):
    import mujoco
    sizes = {0: ("free", 7, 6), 1: ("ball", 4, 3),
             2: ("slide", 1, 1), 3: ("hinge", 1, 1)}
    return [(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j),) +
            sizes[int(model.jnt_type[j])] for j in range(model.njnt)]


def add_hazard_pool(xml, config):
    root = ET.fromstring(xml)
    world = root.find("worldbody")
    names = {node.get("name") for node in root.iter() if node.get("name")}
    for index, item in enumerate(config["members"]):
        name = item["name"]
        if name in names:
            raise ValueError("hazard name already exists: %s" % name)
        names.add(name)
        origin = [a + index * b for a, b in zip(config["park_origin_m"], config["park_spacing_m"])]
        body = ET.SubElement(world, "body", name=name, pos=" ".join(map(str, origin)), gravcomp="1")
        h = item["half_extents_m"]
        mass = item["mass_kg"]
        inertia = [mass * (h[(i + 1) % 3] ** 2 + h[(i + 2) % 3] ** 2) / 3 for i in range(3)]
        # Original LIBERO disables inertiafromgeom; declare the box's physical
        # inertia explicitly rather than changing the scene's compiler options.
        ET.SubElement(body, "inertial", pos="0 0 0", mass=str(mass),
                      diaginertia=" ".join(map(str, inertia)))
        ET.SubElement(body, "joint", name=name + "_joint", type="free", damping="0",
                      armature="0", frictionloss="0", stiffness="0")
        ET.SubElement(body, "geom", name=name + "_geom", type="box",
                      size=" ".join(map(str, item["half_extents_m"])), mass=str(item["mass_kg"]),
                      rgba=" ".join(map(str, config["rgba"])),
                      friction=" ".join(map(str, config["friction"])), contype="1", conaffinity="1", group="1")
    return ET.tostring(root, encoding="unicode")
