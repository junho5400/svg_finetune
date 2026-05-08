"""
Walk data/google-fonts/{ofl,apache}/ , parse each font's METADATA.pb ,
join with tags/all/families.csv , and write data/font_index.csv .

One row per usable font: enough info for downstream glyph extraction
and caption building.

Usable = OFL or APACHE2 license + has the 'latin' subset (so A-Z exists).
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOOGLE_FONTS = ROOT / "data" / "google-fonts"
OUT_PATH = ROOT / "data" / "font_index.csv"
TAGS_CSV = GOOGLE_FONTS / "tags" / "all" / "families.csv"

ALLOWED_LICENSES = {"OFL", "APACHE2"}
TAG_SCORE_THRESHOLD = 50  # tags below this are weak signals; drop them
TOP_N_TAGS = 7


def parse_metadata_pb(path: Path) -> dict:
    """Tiny parser for Google Fonts text-format METADATA.pb.

    We only need a handful of fields, and the format is regular enough
    that a line-by-line state machine is simpler than pulling in protoc.
    """
    data = {
        "name": None, "category": None, "license": None,
        "designer": None, "subsets": [], "fonts": [],
    }
    block_stack: list[str] = []  # which nested {} block we're inside
    cur_font: dict | None = None

    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.endswith("{"):
                # entering a block, e.g. `fonts {` or `axes {` or `source {`
                block_name = line[:-1].rstrip(":").strip()
                block_stack.append(block_name)
                if len(block_stack) == 1 and block_name == "fonts":
                    cur_font = {}
                continue

            if line == "}":
                # leaving a block
                if len(block_stack) == 1 and block_stack[-1] == "fonts" and cur_font is not None:
                    data["fonts"].append(cur_font)
                    cur_font = None
                block_stack.pop()
                continue

            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip('"')

            depth = len(block_stack)
            if depth == 0:
                # top-level scalars we care about
                if key in ("name", "category", "license", "designer"):
                    data[key] = value
                elif key == "subsets":
                    data["subsets"].append(value)
            elif depth == 1 and block_stack[-1] == "fonts" and cur_font is not None:
                # inside a `fonts { ... }` block — capture so we can pick the
                # regular-weight, normal-style TTF later
                if key == "weight":
                    try:
                        cur_font[key] = int(value)
                    except ValueError:
                        cur_font[key] = value
                else:
                    cur_font[key] = value
            # we ignore deeper blocks (e.g. source.files) — not needed
    return data


def pick_primary_ttf(fonts: list[dict]) -> str | None:
    """Pick the canonical 'regular' TTF: weight=400, style=normal, else first."""
    if not fonts:
        return None
    for f in fonts:
        if f.get("weight") == 400 and f.get("style") == "normal":
            return f.get("filename")
    return fonts[0].get("filename")


def load_tags(path: Path) -> dict[str, list[tuple[str, int]]]:
    """families.csv rows: family,?,tag_path,score   ->   family -> [(tag, score), ...]
    Keyed case-insensitively to tolerate name spelling drift.
    """
    out: dict[str, list[tuple[str, int]]] = defaultdict(list)
    with path.open() as f:
        for row in csv.reader(f):
            if len(row) < 4:
                continue
            family, _, tag_path, score = row[0], row[1], row[2], row[3]
            try:
                s = int(score)
            except ValueError:
                continue
            out[family.strip().lower()].append((tag_path.strip(), s))
    return out


def format_top_tags(tag_scores: list[tuple[str, int]]) -> str:
    """Pick top-N tags above threshold, format as "tag:score|tag:score|...".

    The tag_path looks like "/Sans/Geometric"; we keep only the leaf
    ("Geometric") since the prefix duplicates the broader category
    we already have, and the leaf is what'll feed into captions.
    """
    filtered = [(p, s) for p, s in tag_scores if s >= TAG_SCORE_THRESHOLD]
    filtered.sort(key=lambda ps: ps[1], reverse=True)
    leaves = []
    for tag_path, score in filtered[:TOP_N_TAGS]:
        leaf = tag_path.rsplit("/", 1)[-1]
        leaves.append(f"{leaf}:{score}")
    return "|".join(leaves)


def main() -> None:
    if not GOOGLE_FONTS.exists():
        raise SystemExit(f"google-fonts not found at {GOOGLE_FONTS}; clone first")

    print(f"Loading tags from {TAGS_CSV}...")
    tags_by_family = load_tags(TAGS_CSV)
    print(f"  loaded tags for {len(tags_by_family)} families")

    rows: list[dict] = []
    skipped = Counter()
    matched_tags = 0

    for license_dir in ("ofl", "apache"):
        base = GOOGLE_FONTS / license_dir
        if not base.exists():
            continue
        for font_dir in sorted(base.iterdir()):
            meta_path = font_dir / "METADATA.pb"
            if not meta_path.exists():
                skipped["no_metadata"] += 1
                continue

            try:
                meta = parse_metadata_pb(meta_path)
            except Exception as e:
                skipped[f"parse_error:{type(e).__name__}"] += 1
                continue

            if meta["license"] not in ALLOWED_LICENSES:
                skipped[f"license:{meta['license']}"] += 1
                continue
            if "latin" not in meta["subsets"]:
                skipped["no_latin"] += 1
                continue

            ttf_filename = pick_primary_ttf(meta["fonts"])
            if not ttf_filename:
                skipped["no_ttf"] += 1
                continue
            ttf_path = font_dir / ttf_filename
            if not ttf_path.exists():
                skipped["ttf_missing"] += 1
                continue

            tag_scores = tags_by_family.get(meta["name"].strip().lower(), [])
            if tag_scores:
                matched_tags += 1
            top_tags = format_top_tags(tag_scores)

            rows.append({
                "family": meta["name"],
                "dir": str(font_dir.relative_to(ROOT)),
                "ttf_path": str(ttf_path.relative_to(ROOT)),
                "category": meta["category"] or "",
                "license": meta["license"],
                "designer": meta["designer"] or "",
                "has_latin": True,
                "top_tags": top_tags,
            })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    cat_counts = Counter(r["category"] for r in rows)
    lic_counts = Counter(r["license"] for r in rows)

    print("\n=== Index built ===")
    print(f"  wrote   : {OUT_PATH} ({len(rows)} rows)")
    print(f"  matched tags: {matched_tags}/{len(rows)} fonts")
    print(f"  by license  : {dict(lic_counts)}")
    print(f"  by category : {dict(cat_counts)}")
    if skipped:
        print(f"  skipped     : {dict(skipped)}")


if __name__ == "__main__":
    main()
