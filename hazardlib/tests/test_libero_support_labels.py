"""The support bodies a LIBERO scene uses: primary (tables) if the scene has one, else the fallback (floor)."""
from pathlib import Path

import yaml

from hazardhosts.libero.host import support_labels_in_use

CONFIG = Path(__file__).parents[2] / "configs/hazard/libero.yaml"


def lists():
    data = yaml.safe_load(CONFIG.read_text())
    return data["support_bodies"], data["support_fallback_bodies"]


def test_primary_supports_when_the_scene_has_a_table_else_the_floor():
    primary, fallback = lists()
    assert "floor" in fallback and "floor" not in primary
    assert support_labels_in_use(primary, fallback, True) == set(primary)
    assert support_labels_in_use(primary, fallback, False) == set(fallback)       # libero_object
    assert "floor" in support_labels_in_use(primary, fallback, False)
    assert "floor" not in support_labels_in_use(primary, fallback, True)
