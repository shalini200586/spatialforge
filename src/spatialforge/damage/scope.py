"""Scope line items keyed to surfaces: what a repair or inspection would involve, with a quantity where one can be justified.

A quantity is the damage's measured length (crack) or area (stain) when its metric extent is reliable; otherwise the item is kept
with quantity None and weak quality, so the work is listed without an invented amount. Concealed-damage flags become
INSPECTION items (quantity 1 each), never repair items: nothing hidden is assumed to exist.
"""

from __future__ import annotations

from spatialforge.pipeline.models import DamageRegion, Measurement

ACTIONS = {
    "crack": ("seal_and_repair_crack", "length_m", "m"),
    "stain": ("treat_stain_and_repaint", "area_m2", "m2"),
}
INSPECT = {"moisture": "inspect_for_concealed_moisture", "structural_movement": "inspect_for_concealed_structural_movement"}
RANK = {"weak": 1, "moderate": 2, "strong": 3}
NAME = {1: "weak", 2: "moderate", 3: "strong"}


def build_scope(regions: list[DamageRegion], flags: list[dict], tier: str) -> list[dict]:
    items: list[dict] = []
    for d in regions:
        action, field_name, unit = ACTIONS[d.damage_class]
        m = d.length if d.damage_class == "crack" else d.area
        if m is not None and d.metric_reliable:
            qty, quality, basis = m, NAME[min(RANK[d.quality], RANK[m.quality or "weak"])], f"measured {field_name} of the projected region"
        else:
            qty, quality, basis = None, "weak", "no reliable metric extent: quantity not estimated"
        items.append({"item_id": "", "surface_id": d.surface_id, "damage_id": d.id, "flag_id": None, "room_id": d.room_id,
                      "action": action, "quantity": None if qty is None else qty.to_dict(), "quality": quality, "basis": basis})
    for f in flags:
        one = Measurement(1.0, "each", None, None, None, "weak", tier)
        items.append({"item_id": "", "surface_id": f["surface_id"], "damage_id": f["damage_ids"][0] if f["damage_ids"] else None,
                      "flag_id": f["flag_id"], "room_id": f["room_id"], "action": INSPECT.get(f["risk"], "inspect"),
                      "quantity": one.to_dict(), "quality": "weak",
                      "basis": f"rule {f['rule_id']} ({f['rule_name']}): inspection only, hidden damage is not assumed"})
    items.sort(key=lambda i: (i["surface_id"] or "~", i["action"], i["damage_id"] or "", i["flag_id"] or ""))
    for n, it in enumerate(items, 1):
        it["item_id"] = f"scope_{n:03d}"
    return items
