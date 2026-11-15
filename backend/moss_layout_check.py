"""Checks a MOSS-Image page layout (Create → 📰 Comic page & poster) before minutes are spent on it: the same rules as
moss_image_mlx.layout.check (boxes [y_min, x_min, y_max, x_max], canvas centre as the origin), without importing MLX."""


def check(data, width, height):
    problems = []
    if data.get("bbox_format", "yxyx") != "yxyx" or data.get("bbox_coord_space", "abs_centralized") != "abs_centralized":
        problems.append("unexpected box format")
    if not (data.get("summary") or {}).get("caption"):
        problems.append("the page has no description")
    items = data.get("items") or []
    if not items:
        problems.append("the page has nothing on it")
    hw, hh = width / 2, height / 2
    for i, item in enumerate(items):
        box = item.get("bbox")
        name = item.get("text") or (item.get("caption") or f"item {i + 1}")[:30]
        if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box)):
            problems.append(f"{name!r} has no box")
            continue
        y0, x0, y1, x1 = box
        if y0 >= y1 or x0 >= x1:
            problems.append(f"{name!r}'s box is empty")
        if y0 < -hh - 1 or y1 > hh + 1 or x0 < -hw - 1 or x1 > hw + 1:
            problems.append(f"{name!r} goes off the page")
        if item.get("label") == "text" and not str(item.get("text") or "").strip():
            problems.append(f"{name!r} has no words")
    return problems
