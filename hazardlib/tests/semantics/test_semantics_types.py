"""T03 must reuse the T01 host types, not redefine them."""
from pathlib import Path

import hazardlib.host as host
import hazardlib.semantics as semantics
from hazardlib.semantics import contact_graph, events

SEMANTICS_DIR = Path(semantics.__file__).resolve().parent


def test_partner_class_and_contact_obs_are_the_host_objects():
    assert events.PartnerClass is host.PartnerClass
    assert events.ContactObs is host.ContactObs
    assert contact_graph.PartnerClass is host.PartnerClass
    assert contact_graph.ContactObs is host.ContactObs


def test_no_semantics_module_defines_these_classes():
    for path in sorted(SEMANTICS_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "class PartnerClass" not in text, path.name
        assert "class ContactObs" not in text, path.name


def test_event_log_uses_the_host_partner_class():
    log = semantics.HazardEventLog.new("i", partner_class="held_object")
    assert type(log.partner_class) is host.PartnerClass


def test_the_host_partner_classes_are_unchanged():
    assert {m.value for m in host.PartnerClass} == {"robot_link", "held_object", "scene", "hazard"}
