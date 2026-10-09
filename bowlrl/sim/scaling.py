"""Subject-specific scaling of the bowler model (anthropometrics + strength).

Works on the MJCF XML text so it is independent of the MuJoCo python API version.

    xml = scale_model(height_m=1.91, mass_kg=86.9, strength_scale=1.1)
    model = mujoco.MjModel.from_xml_string(xml)

Segment lengths scale linearly with height (relative to the 1.85 m template), masses
with body mass, and peak torques (actuator gear) with `strength_scale` or a per-joint
dict from isometric dynamometry (see configs/subject_example.yaml).
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Dict, Optional

TEMPLATE_HEIGHT = 1.85
TEMPLATE_MASS = 85.0
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "bowler.xml")

_LENGTH_ATTRS = ("pos", "fromto", "size")


def _scale_vec(text: str, k: float) -> str:
    return " ".join(f"{float(v) * k:.6g}" for v in text.split())


def _scale_subtree(elem: ET.Element, k_len: float, k_mass: float) -> None:
    for child in elem:
        if child.tag in ("body", "geom", "site", "joint"):
            for a in _LENGTH_ATTRS:
                if a in child.attrib and child.tag != "joint":
                    child.attrib[a] = _scale_vec(child.attrib[a], k_len)
            if child.tag == "geom" and "mass" in child.attrib:
                child.attrib["mass"] = f"{float(child.attrib['mass']) * k_mass:.6g}"
        if child.tag == "body":
            _scale_subtree(child, k_len, k_mass)


def scale_model(height_m: float = TEMPLATE_HEIGHT, mass_kg: float = TEMPLATE_MASS,
                strength_scale: float = 1.0,
                joint_torques: Optional[Dict[str, float]] = None,
                template_path: str = MODEL_PATH) -> str:
    """Return scaled MJCF XML as a string."""
    tree = ET.parse(template_path)
    root = tree.getroot()
    k_len = height_m / TEMPLATE_HEIGHT
    k_mass = mass_kg / TEMPLATE_MASS

    world = root.find("worldbody")
    pelvis = None
    for b in world.findall("body"):
        if b.attrib.get("name") == "pelvis":
            pelvis = b
    if pelvis is None:
        raise ValueError("pelvis body not found")
    # pelvis height and everything below it
    pelvis.attrib["pos"] = _scale_vec(pelvis.attrib["pos"], k_len)
    _scale_subtree(pelvis, k_len, k_mass)   # scales the pelvis' own geoms and every descendant

    # strength: gear = peak torque; muscle torque scales ~ mass^(2/3) * length, default
    # isometric scaling if no explicit factor
    default_strength = strength_scale * (k_mass ** (2.0 / 3.0)) * k_len
    for motor in root.find("actuator").findall("motor"):
        jname = motor.attrib["joint"]
        if joint_torques and jname in joint_torques:
            motor.attrib["gear"] = f"{float(joint_torques[jname]):.6g}"
        else:
            motor.attrib["gear"] = f"{float(motor.attrib['gear']) * default_strength:.6g}"

    # ball initial body position scales with the hand position
    for b in world.findall("body"):
        if b.attrib.get("name") == "ball":
            b.attrib["pos"] = _scale_vec(b.attrib["pos"], k_len)
    return ET.tostring(root, encoding="unicode")


def set_joint_damping(xml: str, scale: float) -> str:
    """Multiply every hinge joint's damping by `scale` (useful for sim-to-real tuning)."""
    root = ET.fromstring(xml)
    for j in root.iter("joint"):
        if "damping" in j.attrib:
            j.attrib["damping"] = f"{float(j.attrib['damping']) * scale:.6g}"
    return ET.tostring(root, encoding="unicode")
