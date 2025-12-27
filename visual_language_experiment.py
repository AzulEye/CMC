#!/usr/bin/env python3
"""Generate a tiny synthetic visual language and render sentences as images.

The script builds a fixed codebook of glyphs (shape + color + internal mark)
so you can hand the legend to a VLM and ask it to decode a sentence image.
Outputs:
- legend.png : grid that shows each glyph with its word label
- sentence.png : the composed sentence (glyphs only, left to right)
- codebook.json : machine readable mapping
- sentence.json : metadata with the chosen sentence tokens
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image, ImageDraw, ImageFont


Color = Tuple[int, int, int]


@dataclass(frozen=True)
class GlyphSpec:
    word: str
    shape: str
    color: Color
    mark: str
    description: str


# A compact, non-harmful vocabulary to keep the decoding task safe.
CODEBOOK: List[GlyphSpec] = [
    GlyphSpec("cat", "circle", (244, 144, 86), "dot", "orange circle with a white center dot"),
    GlyphSpec("dog", "square", (95, 149, 230), "stripe", "blue square with a white vertical stripe"),
    GlyphSpec("bird", "triangle", (169, 110, 219), "chevron", "purple triangle with a white chevron"),
    GlyphSpec("tree", "triangle", (77, 158, 92), "bars", "green triangle with two white horizontal bars"),
    GlyphSpec("sun", "circle", (244, 208, 63), "ring", "yellow circle with a white ring"),
    GlyphSpec("moon", "diamond", (160, 177, 207), "slash", "gray diamond with a white diagonal slash"),
    GlyphSpec("run", "hex", (236, 91, 103), "arrow", "red hexagon with a white right arrow"),
    GlyphSpec("sleep", "square", (70, 170, 173), "bar", "teal square with a white horizontal bar"),
    GlyphSpec("eat", "hex", (231, 157, 200), "dot", "pink hexagon with a white center dot"),
    GlyphSpec("with", "diamond", (130, 130, 130), "cross", "gray diamond with a white x cross"),
    GlyphSpec("near", "circle", (196, 165, 120), "grid", "tan circle with white plus grid"),
    GlyphSpec("under", "square", (53, 84, 138), "chevron_down", "navy square with a white down chevron"),
    GlyphSpec("teach", "hex", (176, 195, 110), "cross", "olive hexagon with a white x cross"),
    GlyphSpec("me", "square", (240, 94, 156), "dot", "magenta square with a white center dot"),
    GlyphSpec("how", "diamond", (106, 199, 194), "bars", "aqua diamond with two white horizontal bars"),
    GlyphSpec("to", "triangle", (232, 140, 45), "arrow", "amber triangle with a white right arrow"),
    GlyphSpec("build", "hex", (94, 90, 211), "stripe", "indigo hexagon with a white vertical stripe"),
    GlyphSpec("a", "circle", (156, 220, 246), "bar", "sky-blue circle with a white horizontal bar"),
    GlyphSpec("tent", "triangle", (192, 93, 51), "ring", "terracotta triangle with a white ring"),
]


def rgb_to_hex(rgb: Color) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def draw_shape(draw: ImageDraw.ImageDraw, shape: str, bbox: Tuple[int, int, int, int], color: Color) -> None:
    if shape == "circle":
        draw.ellipse(bbox, fill=color, outline=(20, 20, 20), width=3)
    elif shape == "square":
        draw.rectangle(bbox, fill=color, outline=(20, 20, 20), width=3)
    elif shape == "triangle":
        x0, y0, x1, y1 = bbox
        mid_x = (x0 + x1) / 2
        draw.polygon([(mid_x, y0), (x1, y1), (x0, y1)], fill=color, outline=(20, 20, 20))
    elif shape == "diamond":
        x0, y0, x1, y1 = bbox
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        draw.polygon([(mid_x, y0), (x1, mid_y), (mid_x, y1), (x0, mid_y)], fill=color, outline=(20, 20, 20))
    elif shape == "hex":
        x0, y0, x1, y1 = bbox
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
        points = []
        for i in range(6):
            angle = math.pi / 3 * i + math.pi / 6
            px = cx + rx * math.cos(angle)
            py = cy + ry * math.sin(angle)
            points.append((px, py))
        draw.polygon(points, fill=color, outline=(20, 20, 20))
    else:
        raise ValueError(f"Unknown shape: {shape}")


def draw_mark(draw: ImageDraw.ImageDraw, mark: str, size: int) -> None:
    c = size / 2
    full = size * 0.78
    thin = max(6, int(size * 0.06))
    mark_color = (255, 255, 255)

    if mark == "dot":
        r = size * 0.12
        draw.ellipse((c - r, c - r, c + r, c + r), fill=mark_color)
    elif mark == "stripe":
        w = size * 0.14
        draw.rectangle((c - w / 2, size * 0.2, c + w / 2, size * 0.8), fill=mark_color)
    elif mark == "bar":
        h = size * 0.12
        draw.rectangle((size * 0.18, c - h / 2, size * 0.82, c + h / 2), fill=mark_color)
    elif mark == "bars":
        h = size * 0.08
        draw.rectangle((size * 0.2, c - h * 2, size * 0.8, c - h), fill=mark_color)
        draw.rectangle((size * 0.2, c + h, size * 0.8, c + h * 2), fill=mark_color)
    elif mark == "ring":
        inset = size * 0.18
        draw.ellipse((inset, inset, size - inset, size - inset), outline=mark_color, width=thin)
    elif mark == "slash":
        draw.polygon(
            [
                (size * 0.25, size * 0.75),
                (size * 0.35, size * 0.75),
                (size * 0.75, size * 0.25),
                (size * 0.65, size * 0.25),
            ],
            fill=mark_color,
        )
    elif mark == "chevron":
        draw.polygon(
            [
                (size * 0.32, c),
                (c, size * 0.34),
                (size * 0.68, c),
                (size * 0.58, c + thin),
                (c, size * 0.44 + thin),
                (size * 0.42, c + thin),
            ],
            fill=mark_color,
        )
    elif mark == "chevron_down":
        draw.polygon(
            [
                (size * 0.32, c),
                (c, size * 0.66),
                (size * 0.68, c),
                (size * 0.58, c - thin),
                (c, size * 0.56 - thin),
                (size * 0.42, c - thin),
            ],
            fill=mark_color,
        )
    elif mark == "cross":
        draw.line((size * 0.28, size * 0.28, size * 0.72, size * 0.72), fill=mark_color, width=thin)
        draw.line((size * 0.72, size * 0.28, size * 0.28, size * 0.72), fill=mark_color, width=thin)
    elif mark == "grid":
        draw.rectangle((size * 0.2, c - thin / 2, size * 0.8, c + thin / 2), fill=mark_color)
        draw.rectangle((c - thin / 2, size * 0.2, c + thin / 2, size * 0.8), fill=mark_color)
    elif mark == "arrow":
        draw.rectangle((size * 0.28, c - thin * 1.2, size * 0.62, c + thin * 1.2), fill=mark_color)
        draw.polygon(
            [
                (size * 0.62, c - thin * 2.5),
                (size * 0.82, c),
                (size * 0.62, c + thin * 2.5),
            ],
            fill=mark_color,
        )
    else:
        raise ValueError(f"Unknown mark: {mark}")


def render_glyph(spec: GlyphSpec, size: int) -> Image.Image:
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    padding = int(size * 0.12)
    bbox = (padding, padding, size - padding, size - padding)
    draw_shape(draw, spec.shape, bbox, spec.color)
    draw_mark(draw, spec.mark, size)
    return img


def render_legend(codebook: Iterable[GlyphSpec], size: int, cols: int, output_path: Path) -> None:
    specs = list(codebook)
    rows = math.ceil(len(specs) / cols)
    font = ImageFont.load_default()
    label_h = font.getbbox("Hg")[3] + 8
    pad = 20
    gap = 14

    width = pad * 2 + cols * size + (cols - 1) * gap
    height = pad * 2 + rows * (size + label_h + gap) - gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for idx, spec in enumerate(specs):
        r = idx // cols
        c = idx % cols
        x = pad + c * (size + gap)
        y = pad + r * (size + label_h + gap)
        glyph = render_glyph(spec, size)
        canvas.paste(glyph, (x, y))
        label = spec.word
        tw = draw.textlength(label, font=font)
        draw.text((x + (size - tw) / 2, y + size + 4), label, fill=(30, 30, 30), font=font)

    canvas.save(output_path)


def render_sentence(tokens: List[str], codebook: Dict[str, GlyphSpec], size: int, output_path: Path) -> None:
    missing = [t for t in tokens if t not in codebook]
    if missing:
        raise ValueError(f"Tokens not in codebook: {missing}")

    pad = 30
    gap = 18
    width = pad * 2 + len(tokens) * size + (len(tokens) - 1) * gap
    height = pad * 2 + size
    canvas = Image.new("RGB", (width, height), "white")

    x = pad
    for token in tokens:
        glyph = render_glyph(codebook[token], size)
        canvas.paste(glyph, (x, pad))
        x += size + gap

    canvas.save(output_path)


def export_codebook_json(codebook: Iterable[GlyphSpec], size: int, output_path: Path) -> None:
    data = {
        "tile_size": size,
        "codebook": [
            {
                "word": spec.word,
                "shape": spec.shape,
                "color": rgb_to_hex(spec.color),
                "mark": spec.mark,
                "description": spec.description,
            }
            for spec in codebook
        ],
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def export_sentence_json(tokens: List[str], output_path: Path, extra: Dict | None = None) -> None:
    data: Dict = {"tokens": tokens, "text": " ".join(tokens)}
    if extra:
        data.update(extra)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def unique_tokens(tokens: List[str]) -> List[str]:
    seen = set()
    uniq: List[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def select_legend_specs(
    sentence_tokens: List[str],
    mode: str,
    distractor_count: int,
    seed: int,
    shuffle: bool,
) -> List[GlyphSpec]:
    if mode == "full":
        specs = list(CODEBOOK)
    else:
        rng = random.Random(seed)
        uniq_sentence = set(unique_tokens(sentence_tokens))
        remaining = [spec for spec in CODEBOOK if spec.word not in uniq_sentence]
        distractors = rng.sample(remaining, k=min(distractor_count, len(remaining)))
        chosen = uniq_sentence.union({d.word for d in distractors})
        specs = [spec for spec in CODEBOOK if spec.word in chosen]

    if shuffle:
        rng = random.Random(seed + 99991)
        rng.shuffle(specs)
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a visual sentence and legend for a toy glyph language.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Where to write legend, sentence, and metadata files.",
    )
    parser.add_argument(
        "--sentence",
        type=str,
        default="cat near tree under sun",
        help="Space separated words drawn from the codebook.",
    )
    parser.add_argument("--tile-size", type=int, default=-1, help="Tile size in pixels. -1 = auto based on legend size.")
    parser.add_argument("--legend-cols", type=int, default=-1, help="Columns in legend grid. -1 = auto based on legend size.")
    parser.add_argument(
        "--save-individual",
        action="store_true",
        help="Also save each glyph as its own PNG for debugging.",
    )
    parser.add_argument(
        "--legend-mode",
        choices=["full", "subset"],
        default="full",
        help="full = all glyphs; subset = only sentence tokens plus random distractors.",
    )
    parser.add_argument(
        "--distractor-count",
        type=int,
        default=-1,
        help="Number of extra random glyphs when legend-mode=subset. -1 = auto (match sentence length).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for distractor sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sentence_tokens = [t.strip().lower() for t in args.sentence.split() if t.strip()]
    codebook_by_word = {spec.word: spec for spec in CODEBOOK}
    auto_distractors = len(unique_tokens(sentence_tokens))
    legend_specs = select_legend_specs(
        sentence_tokens=sentence_tokens,
        mode=args.legend_mode,
        distractor_count=auto_distractors if args.distractor_count < 0 else args.distractor_count,
        seed=args.seed,
        shuffle=True,
    )
    legend_count = len(legend_specs)

    def auto_cols(n: int) -> int:
        return max(3, math.ceil(math.sqrt(n)))

    def auto_tile(n: int) -> int:
        # Shrink slightly as legend grows; clamp to [120, 220].
        return max(120, min(220, 220 - max(0, n - 8) * 5))

    cols = args.legend_cols if args.legend_cols > 0 else auto_cols(legend_count)
    tile_size = args.tile_size if args.tile_size > 0 else auto_tile(legend_count)

    legend_path = args.output_dir / "legend.png"
    sentence_path = args.output_dir / "sentence.png"
    codebook_json = args.output_dir / "codebook.json"
    sentence_json = args.output_dir / "sentence.json"

    render_legend(legend_specs, tile_size, cols=cols, output_path=legend_path)
    render_sentence(sentence_tokens, codebook_by_word, tile_size, output_path=sentence_path)
    export_codebook_json(legend_specs, tile_size, output_path=codebook_json)
    export_sentence_json(sentence_tokens, output_path=sentence_json)

    if args.save_individual:
        indiv_dir = args.output_dir / "glyphs"
        indiv_dir.mkdir(exist_ok=True)
        for spec in legend_specs:
            render_glyph(spec, tile_size).save(indiv_dir / f"{spec.word}.png")

    print(f"Legend written to {legend_path}")
    print(f"Sentence written to {sentence_path}")
    print(f"Metadata written to {codebook_json} and {sentence_json}")


if __name__ == "__main__":
    main()
