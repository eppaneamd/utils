#!/usr/bin/env python3
"""
Convert a bounding box annotation JSON file to COCO format.

Coordinates in the input are normalized (0-1). Provide either --images-dir
(reads dimensions per image file) or --image-size (uniform dimensions).

The output file defaults to {input_stem}_to_coco.json next to the input.
Category IDs start from 0 by default; use --category-id-start 1 for
standard COCO convention.

Usage:
  python annots_to_coco.py example-annotations.json --images-dir example-images/ --pretty
  python annots_to_coco.py example-annotations.json --image-size 1920x1080
  python annots_to_coco.py example-annotations.json --images-dir example-images/ --category-id-start 1
  python annots_to_coco.py example-annotations.json --image-size 1920x1080 --output out.json --dry-run
"""

import argparse
import datetime
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", metavar="INPUT",
                   help="Annotation JSON file")
    p.add_argument("--output", "-o", default=None,
                   help="Output file (default: {input_stem}_to_coco.json next to the input)")
    p.add_argument("--dry-run", action="store_true",
                   help="Parse and convert but do not write output")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image-size", metavar="WxH",
                   help="Uniform image dimensions for all images, e.g. 1920x1080")
    g.add_argument("--images-dir", metavar="DIR",
                   help="Directory containing images; dimensions read per file via Pillow")
    p.add_argument("--category-id-start", type=int, choices=[0, 1], default=0,
                   help="Starting value for category IDs (default: 0)")
    p.add_argument("--pretty", action="store_true")
    return p.parse_args()


def load_records(path):
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        sys.exit(f"[error] malformed JSON in {path}: {e}")
    if "annotations" not in data:
        sys.exit(f"[error] no 'annotations' key in {path}")
    return data["annotations"]


def resolve_dims(filename, image_size, images_dir):
    if images_dir:
        from PIL import Image
        with Image.open(Path(images_dir) / filename) as img:
            return img.size  # (width, height)
    return image_size


def make_annotation(lbl, img_id, ann_id, cat_id, w, h):
    name = lbl.get("label", "").strip()
    geom = lbl.get("geometry", {})
    if not name or geom.get("kind") != "box":
        return None
    b = geom.get("box", {})
    nx, ny, nw, nh = b.get("x"), b.get("y"), b.get("width"), b.get("height")
    if None in (nx, ny, nw, nh):
        return None
    bx, by, bw, bh = (nx * w, ny * h, nw * w, nh * h) if (w and h) else (nx, ny, nw, nh)
    return {
        "area": bw * bh,
        "bbox": [bx, by, bw, bh],
        "category_id": cat_id[name],
        "id": ann_id,
        "image_id": img_id,
        "iscrowd": 0,
        "segmentation": [],
    }


def build_coco(records, image_size, images_dir, source, id_start):
    all_names = sorted({
        lbl["label"].strip()
        for rec in records
        for lbl in rec.get("labels", [])
        if lbl.get("geometry", {}).get("kind") == "box" and lbl.get("label", "").strip()
    })
    categories = [{"id": i + id_start, "name": n, "supercategory": "object"}
                  for i, n in enumerate(all_names)]
    cat_id = {c["name"]: c["id"] for c in categories}

    images, annotations = [], []
    seen = {}  # filename -> (img_id, w, h)
    ann_id = 1

    for rec in records:
        filename = rec.get("asset", {}).get("filename", "").strip()
        if not filename:
            continue

        if filename not in seen:
            img_id = len(seen) + 1
            w, h = resolve_dims(filename, image_size, images_dir)
            seen[filename] = (img_id, w, h)
            images.append({"file_name": filename, "height": h, "width": w,
                           "id": img_id, "coco_url": None, "date_captured": None})

        img_id, w, h = seen[filename]

        for lbl in rec.get("labels", []):
            ann = make_annotation(lbl, img_id, ann_id, cat_id, w, h)
            if ann:
                annotations.append(ann)
                ann_id += 1

    return {
        "info": {"description": f"Converted from {source} by annots_to_coco.py",
                 "date_created": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},
        "licenses": [], "categories": categories, "images": images, "annotations": annotations,
    }


def main():
    args = parse_args()
    image_size = None
    if args.image_size:
        try:
            w, h = args.image_size.lower().split("x", 1)
            image_size = (int(w), int(h))
        except (ValueError, TypeError):
            sys.exit(f"[error] --image-size must be WxH (e.g. 1920x1080), got: {args.image_size!r}")

    if args.images_dir:
        try:
            import PIL  # noqa: F401
        except ImportError:
            sys.exit("[error] --images-dir requires Pillow: pip install Pillow")

    inp = Path(args.input)
    if not inp.exists():
        sys.exit(f"[error] file not found: {inp}")

    records = load_records(inp)
    coco = build_coco(records, image_size, args.images_dir, inp.name, args.category_id_start)

    out = Path(args.output) if args.output else inp.parent / f"{inp.stem}_to_coco.json"
    summary = (f"{len(coco['images'])} images · {len(coco['annotations'])} annotations · "
               f"{len(coco['categories'])} categories → {out}")

    if args.dry_run:
        print(f"[dry-run] {summary} (not written)", file=sys.stderr)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(coco, indent=2 if args.pretty else None))
        print(f"[done] {summary}", file=sys.stderr)


if __name__ == "__main__":
    main()
