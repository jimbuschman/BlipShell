"""Export the BlipShell entity graph from SQLite.

The entity graph lives in three tables inside blipshell.db:
  - entities             (id, name, entity_type)
  - entity_relationships (subject_id, predicate, object_id, source_memory_id)
  - entity_mentions      (entity_id, memory_id)

This script pulls the relationship triples (plus per-entity mention counts and
graph degree) and writes them out in one or both of two formats:

  - json: a node-link document, loadable by networkx
          (nx.node_link_graph(data, edges="links")) or anything else.
  - html: a self-contained interactive visualization using vis-network from a
          CDN. Just open it in a browser. No build step, no local deps.

Because the full graph is large (~31K entities / ~56K relationships), the HTML
view caps nodes by default to stay responsive in the browser. JSON is
unfiltered by default. Use the filters to scope what you pull:

  --entity NAME [--depth N]   ego subgraph around one or more entities (BFS)
  --top-n N                   keep the N highest-degree nodes
  --min-degree N              drop nodes with fewer than N relationships
  --type TYPE                 keep only the given entity type(s)

Usage:
    python scripts/export_graph.py                      # -> graph.json + graph.html
    python scripts/export_graph.py --format json
    python scripts/export_graph.py --format html --top-n 500
    python scripts/export_graph.py --entity python --depth 2 --format both
    python scripts/export_graph.py --db data/blipshell.db --out out/mygraph
"""

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

# vis-network is pinned to a specific version so the output keeps working even
# if the CDN's "latest" changes its API.
VIS_NETWORK_JS = "https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"

# Stable color per entity type (anything unknown falls back to grey).
TYPE_COLORS = {
    "person": "#e8743b",
    "project": "#19a979",
    "technology": "#1f83b4",
    "concept": "#a173d1",
    "preference": "#f6c85f",
    "place": "#ed5564",
    "organization": "#6f4e7c",
}
DEFAULT_COLOR = "#888888"

# Cap for the HTML view unless the user opts into more. The full graph will
# freeze most browsers, so we truncate by degree and log what we dropped.
DEFAULT_HTML_TOP_N = 300


def log(msg: str):
    print(msg, file=sys.stderr)


def load_graph(db_path: str):
    """Read entities, relationships and mention counts from SQLite.

    Returns (nodes_by_id, edges) where nodes_by_id maps entity id -> dict and
    edges is a list of (subject_id, predicate, object_id, source_memory_id).
    """
    if not Path(db_path).exists():
        log(f"ERROR: database not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        nodes = {}
        for row in conn.execute("SELECT id, name, entity_type FROM entities"):
            nodes[row["id"]] = {
                "id": row["id"],
                "name": row["name"],
                "type": row["entity_type"] or "concept",
                "degree": 0,
                "mentions": 0,
            }

        # Mention counts per entity (graph "weight" — how often it's referenced).
        for row in conn.execute(
            "SELECT entity_id, COUNT(*) AS c FROM entity_mentions GROUP BY entity_id"
        ):
            if row["entity_id"] in nodes:
                nodes[row["entity_id"]]["mentions"] = row["c"]

        edges = []
        for row in conn.execute(
            "SELECT subject_id, predicate, object_id, source_memory_id "
            "FROM entity_relationships"
        ):
            s, o = row["subject_id"], row["object_id"]
            # Skip dangling edges (subject/object pointing at a deleted entity).
            if s not in nodes or o not in nodes:
                continue
            edges.append((s, row["predicate"], o, row["source_memory_id"]))
            nodes[s]["degree"] += 1
            nodes[o]["degree"] += 1

        return nodes, edges
    finally:
        conn.close()


def ego_filter(nodes, edges, seed_names, depth):
    """Keep only nodes within `depth` hops of any seed entity (by name)."""
    name_to_id = {}
    for nid, n in nodes.items():
        name_to_id.setdefault(n["name"].lower(), nid)

    frontier = set()
    for raw in seed_names:
        nid = name_to_id.get(raw.lower())
        if nid is None:
            log(f"WARNING: entity not found, skipping seed: {raw!r}")
        else:
            frontier.add(nid)

    if not frontier:
        log("ERROR: none of the requested seed entities exist in the graph.")
        sys.exit(1)

    adj = defaultdict(set)
    for s, _pred, o, _mid in edges:
        adj[s].add(o)
        adj[o].add(s)

    keep = set(frontier)
    for _ in range(max(0, depth)):
        nxt = set()
        for nid in frontier:
            nxt |= adj[nid]
        nxt -= keep
        keep |= nxt
        frontier = nxt
        if not frontier:
            break

    return keep


def apply_filters(nodes, edges, args):
    """Apply --entity/--type/--min-degree/--top-n. Returns (nodes, edges)."""
    keep_ids = set(nodes)

    if args.entity:
        keep_ids &= ego_filter(nodes, edges, args.entity, args.depth)

    if args.type:
        wanted = {t.lower() for t in args.type}
        keep_ids = {i for i in keep_ids if nodes[i]["type"].lower() in wanted}

    if args.min_degree:
        keep_ids = {i for i in keep_ids if nodes[i]["degree"] >= args.min_degree}

    if args.top_n and len(keep_ids) > args.top_n:
        ranked = sorted(keep_ids, key=lambda i: nodes[i]["degree"], reverse=True)
        dropped = len(keep_ids) - args.top_n
        keep_ids = set(ranked[: args.top_n])
        log(f"NOTE: --top-n {args.top_n} kept the {args.top_n} highest-degree "
            f"nodes, dropped {dropped} lower-degree nodes.")

    sub_nodes = {i: nodes[i] for i in keep_ids}
    sub_edges = [e for e in edges if e[0] in keep_ids and e[2] in keep_ids]
    return sub_nodes, sub_edges


def to_node_link(nodes, edges, db_path):
    """networkx-compatible node-link document."""
    return {
        "directed": True,
        "multigraph": True,
        "graph": {
            "source": db_path,
            "entity_count": len(nodes),
            "relationship_count": len(edges),
        },
        "nodes": [
            {
                "id": n["id"],
                "name": n["name"],
                "type": n["type"],
                "degree": n["degree"],
                "mentions": n["mentions"],
            }
            for n in nodes.values()
        ],
        "links": [
            {
                "source": s,
                "target": o,
                "predicate": pred,
                "source_memory_id": mid,
            }
            for (s, pred, o, mid) in edges
        ],
    }


def write_json(nodes, edges, db_path, out_path):
    doc = to_node_link(nodes, edges, db_path)
    Path(out_path).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}  ({len(nodes)} nodes, {len(edges)} edges)")


def write_html(nodes, edges, db_path, out_path):
    vis_nodes = []
    for n in nodes.values():
        # Size by mention count so heavily-referenced entities read as bigger.
        size = 10 + min(40, n["mentions"])
        vis_nodes.append({
            "id": n["id"],
            "label": n["name"],
            "title": f"{n['name']} ({n['type']})\\n"
                     f"degree {n['degree']}, mentions {n['mentions']}",
            "color": TYPE_COLORS.get(n["type"].lower(), DEFAULT_COLOR),
            "value": size,
            "group": n["type"],
        })

    vis_edges = []
    for s, pred, o, _mid in edges:
        vis_edges.append({"from": s, "to": o, "label": pred, "arrows": "to"})

    legend = "".join(
        f'<span class="chip" style="background:{c}">{t}</span>'
        for t, c in TYPE_COLORS.items()
    )

    html = _HTML_TEMPLATE.format(
        vis_js=VIS_NETWORK_JS,
        db_path=db_path,
        node_count=len(nodes),
        edge_count=len(edges),
        legend=legend,
        nodes_json=json.dumps(vis_nodes),
        edges_json=json.dumps(vis_edges),
    )
    Path(out_path).write_text(html, encoding="utf-8")
    log(f"Wrote {out_path}  ({len(nodes)} nodes, {len(edges)} edges) "
        f"-- open it in a browser")


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BlipShell entity graph</title>
<script src="{vis_js}"></script>
<style>
  html, body {{ margin: 0; height: 100%; font-family: system-ui, sans-serif; background: #1a1a1a; color: #eee; }}
  #bar {{ padding: 8px 12px; background: #222; border-bottom: 1px solid #333; font-size: 13px; }}
  #bar b {{ color: #fff; }}
  .chip {{ display: inline-block; padding: 1px 8px; margin: 0 4px; border-radius: 10px; color: #000; font-size: 11px; }}
  #graph {{ width: 100%; height: calc(100% - 40px); }}
  #find {{ background: #111; color: #eee; border: 1px solid #444; border-radius: 4px; padding: 3px 6px; margin-left: 8px; }}
</style>
</head>
<body>
<div id="bar">
  <b>BlipShell entity graph</b> &mdash; {node_count} nodes, {edge_count} edges
  <span style="color:#888">(source: {db_path})</span>
  {legend}
  <input id="find" placeholder="find entity..." autocomplete="off">
</div>
<div id="graph"></div>
<script>
  const nodes = new vis.DataSet({nodes_json});
  const edges = new vis.DataSet({edges_json});
  const container = document.getElementById("graph");
  const network = new vis.Network(container, {{ nodes, edges }}, {{
    nodes: {{ shape: "dot", scaling: {{ min: 8, max: 50 }}, font: {{ color: "#eee", size: 12 }} }},
    edges: {{ color: {{ color: "#555", highlight: "#fff" }}, font: {{ color: "#aaa", size: 9, strokeWidth: 0 }}, smooth: false }},
    physics: {{ stabilization: {{ iterations: 200 }}, barnesHut: {{ gravitationalConstant: -8000, springLength: 120 }} }},
    interaction: {{ hover: true, tooltipDelay: 100 }}
  }});

  // Freeze the layout once it settles so the graph stops drifting.
  // Without this the force simulation runs forever and the nodes jiggle.
  network.once("stabilizationIterationsDone", () => network.setOptions({{ physics: false }}));
  // Let dragging a node re-settle its neighbors, then freeze again.
  network.on("dragStart", () => network.setOptions({{ physics: true }}));
  network.on("dragEnd", () => network.setOptions({{ physics: false }}));

  // Type-to-find: focus the first matching node.
  document.getElementById("find").addEventListener("keydown", (e) => {{
    if (e.key !== "Enter") return;
    const q = e.target.value.trim().toLowerCase();
    if (!q) return;
    const hit = nodes.get().find(n => (n.label || "").toLowerCase().includes(q));
    if (hit) {{ network.selectNodes([hit.id]); network.focus(hit.id, {{ scale: 1.2, animation: true }}); }}
  }});
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Export the BlipShell entity graph (JSON and/or HTML viz)."
    )
    parser.add_argument("--db", default="data/blipshell.db", help="SQLite DB path")
    parser.add_argument("--format", choices=["json", "html", "both"],
                        default="both", help="Output format (default: both)")
    parser.add_argument("--out", default="graph",
                        help="Output path prefix; extension added per format "
                             "(default: graph -> graph.json / graph.html)")
    parser.add_argument("--entity", action="append", metavar="NAME",
                        help="Seed entity for an ego subgraph (repeatable)")
    parser.add_argument("--depth", type=int, default=2,
                        help="Hops from seed entities (default: 2)")
    parser.add_argument("--top-n", type=int, default=None,
                        help="Keep only the N highest-degree nodes")
    parser.add_argument("--min-degree", type=int, default=0,
                        help="Drop nodes with fewer than N relationships")
    parser.add_argument("--type", action="append", metavar="TYPE",
                        help="Keep only this entity type (repeatable)")
    args = parser.parse_args()

    nodes, edges = load_graph(args.db)
    log(f"Loaded full graph: {len(nodes)} entities, {len(edges)} relationships")

    nodes, edges = apply_filters(nodes, edges, args)

    if not nodes:
        log("Nothing to export after filters.")
        sys.exit(1)

    out = args.out
    if args.format in ("json", "both"):
        write_json(nodes, edges, args.db, f"{out}.json")
    if args.format in ("html", "both"):
        # HTML defaults to a degree cap if the user didn't set --top-n and the
        # graph is large, to keep the browser responsive.
        if args.top_n is None and len(nodes) > DEFAULT_HTML_TOP_N:
            ranked = sorted(nodes, key=lambda i: nodes[i]["degree"], reverse=True)
            dropped = len(nodes) - DEFAULT_HTML_TOP_N
            keep = set(ranked[:DEFAULT_HTML_TOP_N])
            h_nodes = {i: nodes[i] for i in keep}
            h_edges = [e for e in edges if e[0] in keep and e[2] in keep]
            log(f"NOTE: HTML view capped at top {DEFAULT_HTML_TOP_N} nodes by "
                f"degree (dropped {dropped}). Use --top-n to change, "
                f"or --format json for the full graph.")
            write_html(h_nodes, h_edges, args.db, f"{out}.html")
        else:
            write_html(nodes, edges, args.db, f"{out}.html")


if __name__ == "__main__":
    main()
