#!/usr/bin/env python3
"""Render ring-aligned Mermaid/SVG views backed by the detailed topology."""

import argparse
import html
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import ImageFont


ROOT = Path(__file__).resolve().parent
RINGS = {
    "r3": ("#F5F6F8", "#667085"),
    "r2": ("#EDF2F6", "#48637B"),
    "r1": ("#ECF4EE", "#3D7052"),
    "r0": ("#E8F0FA", "#2A5A8D"),
}
FONT_REG = "/usr/share/fonts/TTF/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"
_FONTS = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _FONTS:
        _FONTS[key] = ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)
    return _FONTS[key]


def measure(value, size, bold=False):
    return font(size, bold).getlength(value)


def wrap_text(value, max_width, size, bold=False):
    lines, current = [], ""
    for word in value.split():
        trial = f"{current} {word}".strip()
        if current and measure(trial, size, bold) > max_width:
            lines.append(current)
            current = word
        else:
            current = trial
    lines.append(current)
    return lines


def validate(graph):
    nodes = {node["id"] for node in graph["nodes"]}
    groups = {group["id"] for group in graph["groups"]}
    assert len(nodes) == len(graph["nodes"])
    assert len(groups) == len(graph["groups"])
    assert not nodes & groups
    for node in graph["nodes"]:
        assert not node.get("group") or node["group"] in groups
    adjacency = {node: [] for node in nodes}
    for edge in graph["edges"]:
        assert edge["from"] in nodes and edge["to"] in nodes
        if edge.get("kind") != "relation":
            adjacency[edge["from"]].append(edge["to"])
    visiting, visited = set(), set()

    def walk(node):
        assert node not in visiting, f"code dependency cycle through {node}"
        if node in visited:
            return
        visiting.add(node)
        for dependency in adjacency[node]:
            walk(dependency)
        visiting.remove(node)
        visited.add(node)

    for node in nodes:
        walk(node)
    if "r0" in groups:
        membership = {node["id"]: node.get("group") for node in graph["nodes"]}
        for source, dependencies in adjacency.items():
            for dependency in dependencies:
                if membership[source] == "r0":
                    assert membership[dependency] == "r0", "R0 code must not depend upward"
                if membership[source] == "r1":
                    assert membership[dependency] not in ("r2", "r3"), "R1 code must not depend on products"
        for pure_core in ("db", "models"):
            assert "rust" in adjacency[pure_core], "peer core must have a direct primitive route"
            reachable, todo = set(), [pure_core]
            while todo:
                current = todo.pop()
                if current not in reachable:
                    reachable.add(current)
                    todo.extend(adjacency[current])
            assert "bedrock" not in reachable, "pure core must not require the Bedrock gateway"


def validate_view(name, view, graph):
    assert [ring["id"] for ring in view["rings"]] == list(RINGS)
    membership = {node["id"]: node.get("group") for node in graph["nodes"]}
    covered = set(view["outside_refs"]) | set(view["separate"]["refs"])
    for ring in view["rings"]:
        assert 1 <= len(ring["cards"]) <= 4
        refs = set(ring.get("note_refs", []))
        for card in ring["cards"]:
            refs.update(card["refs"])
            assert 1 <= len(card["lines"]) <= 2
        if name == "proposed":
            assert all(membership[ref] == ring["id"] for ref in refs)
        covered.update(refs)
    assert covered == set(membership), "view must account for every detailed-topology node"


def mermaid(view):
    lines = ["flowchart TB"]
    for ring in view["rings"]:
        title = html.escape(f'{ring["id"].upper()} - {ring["name"]}')
        peers = " | ".join(html.escape(card["summary"]) for card in ring["cards"])
        lines.append(f'  {ring["id"]}["<b>{title}</b><br/>{peers}"]:::{ring["id"]}')
    lines.extend(("  r3 --> r2", "  r2 --> r1", "  r1 --> r0"))
    for ring, (fill, stroke) in RINGS.items():
        lines.append(f"  classDef {ring} fill:{fill},stroke:{stroke},color:#172B3A")
    lines.append("  %% Layer direction only: R2 may also use R0 directly.")
    return "\n".join(lines) + "\n"


def svg(view):
    """Font-measured geometry; every string is wrapped to its box before placement."""
    width = 1240
    margin, gutter, right = 40, 150, 40
    content_x = gutter + 10
    content_right = width - right
    content_w = content_right - content_x
    gap, pad = 14, 14
    body = []

    def text(x, y, value, size=15, bold=False, color="#172B3A", anchor="start"):
        weight = 700 if bold else 400
        body.append(
            f'<text x="{x:g}" y="{y:g}" font-size="{size:g}" font-weight="{weight}" '
            f'fill="{color}" text-anchor="{anchor}">{html.escape(value)}</text>'
        )

    cursor = 44
    text(margin, cursor, view["title"], 27, True)
    cursor += 30
    text(margin, cursor, view["subtitle"], 15, color="#526576")
    cursor += 25
    text(margin, cursor, "READ TOP TO BOTTOM  /  ONE BAND PER RING  /  NO SERIAL ORDER BETWEEN PEERS", 11, True, "#657786")
    cursor += 24

    bands = []
    for ring in view["rings"]:
        count = len(ring["cards"])
        card_w = (content_w - gap * (count - 1)) / count
        usable = card_w - 2 * pad
        laid_out = []
        for card in ring["cards"]:
            title_lines = wrap_text(card["title"], usable, 17, True)
            lines = []
            for line in card["lines"]:
                lines.extend(wrap_text(line, usable, 14))
            assert all(measure(line, 17, True) <= usable for line in title_lines), card["title"]
            assert all(measure(line, 14) <= usable for line in lines), card["title"]
            card_h = pad + len(title_lines) * 21 + (6 if lines else 0) + len(lines) * 19 + pad
            laid_out.append((card, title_lines, lines, card_h))
        heading = wrap_text(ring["name"], content_w, 20, True)
        desc = wrap_text(ring["description"], content_w, 13.5)
        head_h = 26 + len(heading) * 24 + len(desc) * 18 + 8
        card_h = max(item[3] for item in laid_out)
        bands.append((ring, laid_out, heading, desc, head_h, card_h, card_w))

    for index, (ring, cards, heading, desc, head_h, card_h, card_w) in enumerate(bands):
        fill, accent = RINGS[ring["id"]]
        band_h = head_h + card_h + 16
        body.append(f'<rect x="{margin}" y="{cursor:g}" width="{content_right - margin}" height="{band_h}" rx="8" fill="{fill}"/>')
        body.append(f'<rect x="{margin}" y="{cursor:g}" width="7" height="{band_h}" rx="3" fill="{accent}"/>')
        text(margin + 55, cursor + band_h / 2 - 4, ring["id"].upper(), 34, True, accent, "middle")
        text(margin + 55, cursor + band_h / 2 + 17, "RING", 11, True, accent, "middle")
        for line_index, line in enumerate(heading):
            text(content_x, cursor + 30 + line_index * 24, line, 20, True)
        for line_index, line in enumerate(desc):
            text(content_x, cursor + 30 + len(heading) * 24 + 4 + line_index * 18, line, 13.5, color="#526576")
        top = cursor + head_h
        for column, (card, title_lines, lines, height) in enumerate(cards):
            x = content_x + column * (card_w + gap)
            status = card.get("status")
            card_fill = "#FFF8E9" if status else "#FFFFFF"
            border = "#BB9855" if status else "#CCD7DF"
            dash = ' stroke-dasharray="5 4"' if status in ("deferred", "candidate") else ""
            body.append(f'<rect x="{x:g}" y="{top:g}" width="{card_w:g}" height="{card_h:g}" rx="5" fill="{card_fill}" stroke="{border}"{dash}/>')
            y = top + pad + 16
            for line in title_lines:
                text(x + pad, y, line, 17, True)
                y += 21
            y += 6
            for line in lines:
                text(x + pad, y, line, 14, color="#465C6D")
                y += 19
        cursor += band_h
        if index < len(bands) - 1:
            cx = content_x + content_w / 2
            body.append(f'<path d="M{cx:g},{cursor + 3:g} V{cursor + 18:g}" fill="none" stroke="#81909C" stroke-width="1.5" marker-end="url(#arrow)"/>')
            cursor += 25

    cursor += 6
    for line in wrap_text(view["direct_note"], content_w, 14):
        text(content_x, cursor, line, 14, True, "#405B70")
        cursor += 20
    cursor += 6
    sep_lines = [view["separate"]["title"], *view["separate"]["lines"]]
    wrapped = []
    for item in sep_lines:
        wrapped.append((item, wrap_text(item, content_w - 20, 15)))
    sep_h = 16 + sum(len(lines) * 23 + (8 if i == 0 else 0) for i, (_, lines) in enumerate(wrapped)) + 14
    body.append(f'<rect x="{margin}" y="{cursor:g}" width="{content_right - margin}" height="{sep_h}" rx="7" fill="#F4F1F8"/>')
    text(content_x, cursor + 24, view["separate"]["title"], 17, True, "#68517D")
    y = cursor + 50
    for line in view["separate"]["lines"]:
        for piece in wrap_text(line, content_w - 20, 15):
            text(content_x, y, piece, 15, color="#526576")
            y += 23
    cursor += sep_h + 24
    for line in wrap_text(view["external_note"], content_w, 12):
        text(margin, cursor, line, 12, color="#657786")
        cursor += 17
    height = cursor + 20

    header = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height:g}" viewBox="0 0 {width} {height:g}" role="img" aria-labelledby="title description">',
        f'<title id="title">{html.escape(view["title"])}</title>',
        f'<desc id="description">{html.escape(view["subtitle"])} Four aligned bands, R3 through R0. R2 may call R0 directly.</desc>',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="none" stroke="#81909C" stroke-width="1.4"/></marker></defs>',
        f'<rect width="{width}" height="{height:g}" fill="#FFFFFF"/>',
        '<g font-family="DejaVu Sans, sans-serif" fill="#172B3A">',
    ]
    return "\n".join([*header, *body, "</g>", "</svg>"]) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    topology = json.loads((ROOT / "topology.json").read_text())
    views = json.loads((ROOT / "views.json").read_text())
    assert set(views) == set(topology)
    publication = json.loads((ROOT / "publication.json").read_text())
    assert publication["approved_to_publish"] is False, "this renderer is for the local review stage"
    generated = {}
    text = (ROOT / "source.md").read_text()
    figures = []
    for name, graph in topology.items():
        validate(graph)
        view = views[name]
        validate_view(name, view, graph)
        mmd, image = mermaid(view), svg(view)
        tree = ET.fromstring(image)
        assert tree.tag.endswith("svg") and tree.get("viewBox")
        generated[f"diagrams/{name}.mmd"] = mmd
        generated[f"diagrams/{name}.svg"] = image
        marker = "{{diagram:" + name + "}}"
        assert text.count(marker) == 1
        note = " ".join(view["separate"]["lines"])
        text = text.replace(marker, f"```mermaid\n{mmd}```\n\n[SVG diagram](diagrams/{name}.svg)\n\n{view['direct_note']}\n\n**Separate Song track:** {note}")
        figures.append(f'<section><h2>{html.escape(view["title"])}</h2><a href="diagrams/{name}.svg">Open full-size SVG</a><div class="diagram"><img src="diagrams/{name}.svg" alt="{html.escape(view["title"])}"></div></section>')
    assert "{{diagram:" not in text
    generated["RFC.md"] = text
    generated["index.html"] = '''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RFC 0001 | Core Architecture Review</title>
<style>
body{margin:0;background:#fafbfc;color:#172b3a;font:16px/1.5 system-ui,sans-serif}
main{max-width:1120px;margin:auto;padding:28px}header{border-bottom:2px solid #172b3a;padding-bottom:24px}
h1{font-size:34px;line-height:1.2;margin:12px 0}h2{font-size:23px}a{color:#235e8c}nav a{margin-right:24px}
.status{font-size:13px;font-weight:700;letter-spacing:.08em;color:#8d6421}section{margin:38px 0}.diagram{overflow:auto;margin-top:18px;border:1px solid #ccd5dc;border-radius:6px;background:white}.diagram img{display:block;width:100%;min-width:680px}
@media(max-width:700px){main{padding:18px}h1{font-size:27px}nav a{display:block;margin-top:8px}}
</style><main><header><div class="status">PUBLISHED FOR REVIEW 2026-09-11 / GITHUB Ptyktos/fold#15 / GITLAB PENDING</div>
<h1>Core Architecture, Evidence Standards, and Fold Sunset</h1>
<p>Four aligned rings. Today first, proposed boundaries second. Short form posted to GitHub Ptyktos/fold#15; canonical GitLab publication is pending approval.</p>
<nav><a href="RFC.md">Read the complete RFC</a><a href="ownership.md">Ownership evidence</a><a href="../../call-census/README.md">Call census</a></nav></header>
''' + "\n".join(figures) + "\n</main></html>\n"
    for relative, content in generated.items():
        path = ROOT / relative
        if args.check:
            assert path.read_text() == content, f"stale generated artifact: {relative}"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    for relative in publication["attachments_after_approval"]:
        assert (ROOT / relative).exists(), f"missing publication attachment: {relative}"
    print(f'{"Checked" if args.check else "Generated"} {len(generated)} local artifacts; publication remains blocked.')


if __name__ == "__main__":
    main()
