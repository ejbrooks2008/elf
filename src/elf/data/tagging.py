"""Image tagging with honest labeling for uncensored fine-tuning.

We ensemble a community Florence checkpoint (for rich open-captioning) with
the SmilingWolf wd-v1-4-swinv2-tagger-v2 classifier (for reliable NSFW and
object tags), then apply style-conformance scoring to handle the high
variance in training images.

Strategy — "honest labeling with conformance tiers":
  The training set is ~25% on-target and ~75% off-target.  Rather than
  forcing every image into the desired style (which teaches wrong
  associations), we:

  ON-TARGET images (match desired aesthetic):
    • Tagged with ``on_target``, ``medieval_fantasy``, etc.
    • Default character traits (red-brown hair, green eyes, barefoot)
      are OMITTED — absence = default, keeping tags sparse.
    • Caption rewritten with ideal prose style (prefix + suffix).
    → The model learns: "on_target" = this specific visual look.

  OFF-TARGET images (different style, palette, setting):
    • Tagged with ``off_target`` + explicit deviation markers
      (e.g. ``anime_style``, ``bright_palette``, ``plain_background``).
    • All character traits are explicitly tagged.
    • Caption honestly describes what's visible.
    → The model learns: these deviation tags = this different look.

  At inference time, prompting with "on_target, medieval_fantasy, ..."
  steers generation toward the desired aesthetic.

Target aesthetic:
  - Highly detailed digital art (Alita: Battle Angel quality)
  - Medieval fantasy, Pacific Northwest setting
  - Moody / dark / filtered lighting (never bright and sunny)
  - Muted earthy palette (no pastels or bright colours)
  - POV from the viewer, featuring a wood elf protagonist

The .txt caption file written per image uses: tags on line 1, natural-
language caption on line 2.  Both are structured so fine-tune trainers
(kohya, EveryDream, etc.) can consume them directly.
"""
from __future__ import annotations

import hashlib
import json
import csv
import os
import re
import signal
import sys
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from huggingface_hub import hf_hub_download

from transformers import (
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoProcessor,
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor,
)


# Model choices (wd v2 is public; v3 can be gated)
WD_MODEL_ID = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
# Use the PromptGen fine-tune for uncensored, tag-friendly captions
FLORENCE_MODEL_ID = "MiaoshouAI/Florence-2-large-PromptGen-v1.5"
LLAVA_MODEL_ID = "Salesforce/instructblip-flan-t5-xl"  # uncensored multimodal captioner fallback

# ---------------------------------------------------------------------------
#  Scoring and caps
# ---------------------------------------------------------------------------
TAG_THRESHOLD = 0.28          # base cutoff for WD tagger
TOP_K = 80                    # maximum total tags kept
ANIME_PENALTY = 0.35          # multiply scores for anime/cartoon prefixes
ANIME_MIN_SCORE = 0.42        # drop anime-ish tags below this after penalty
ANIME_MAX = 5                 # never allow more than N anime/cartoon/lineart tags

# ---------------------------------------------------------------------------
#  Forced identity / style tags  (always injected)
# ---------------------------------------------------------------------------
# Identity tags always injected (the character is always our elf)
IDENTITY_TAGS = [
    "wood_elf",
    "pov",
]

# Style tags — only injected when image actually conforms to the target
# aesthetic.  Off-target images get *deviation* tags instead, so the model
# learns what makes the desired style distinct.
TARGET_STYLE_TAGS = [
    "medieval_fantasy",
    "pacific_northwest",
    "highly_detailed_digital_art",
]

ALWAYS_FORCE_FORCED_TAGS = True
FORCED_EVIDENCE_SUBSTRINGS = {"elf", "elven", "fae", "pointy_ears"}

# ---------------------------------------------------------------------------
#  Style-conformance scoring
# ---------------------------------------------------------------------------
# The pipeline assigns each image a tier: "on_target" vs "off_target".
# ON-TARGET (≈25% of training set):
#   • Gets TARGET_STYLE_TAGS injected
#   • Caption rewritten into our preferred prose style (prefix + suffix)
#   • Default character traits (red-brown hair, green eyes, pale skin,
#     barefoot) are OMITTED from tags — absence = default.
# OFF-TARGET (≈75% of training set):
#   • Gets explicit deviation tags describing HOW it differs
#     (e.g. cartoon_style, bright_palette, modern_clothing)
#   • Caption is honest: describes what's visible, prefixed with
#     the actual art style, with deviation markers
#   • Default character traits are INCLUDED (tagged explicitly)
#     so the model learns their visual association
#
# At inference time you prompt with "on_target, medieval_fantasy, ..."
# and the model steers toward the learned ideal aesthetic.

# Signals indicating ON-TARGET style (each matched gives positive score)
_CONFORMANCE_POSITIVE = {
    # Art style
    "realistic", "digital_art", "highly_detailed", "hyper_realistic",
    "detailed", "sharp_focus", "depth_of_field",
    # Palette / mood
    "muted", "earthy", "dark_palette", "moody", "shadows", "low_light",
    "filtered_light", "overcast",
    # Setting
    "forest", "cabin", "mountain", "mossy", "ferns", "mist",
    "old_growth_forest", "rustic_cabin", "mountain_stream",
    # Character
    "barefoot", "pointy_ears", "freckles",
}
# Signals indicating OFF-TARGET style (each matched gives negative score)
_CONFORMANCE_NEGATIVE = {
    # Wrong art style
    "anime", "cartoon", "comic", "manga", "chibi", "sketch", "lineart",
    "3d_render", "pixel_art", "watercolor", "oil_painting", "pencil",
    "cel_shading", "flat_color",
    # Wrong palette / mood
    "bright", "sunny", "pastel", "neon", "vibrant", "colorful",
    "blue_sky", "clear_sky",
    # Wrong setting
    "modern", "urban", "city", "studio", "plain_background",
    "grey_background", "white_background", "simple_background",
    # Wrong clothing
    "school_uniform", "sweater", "hoodie", "jeans", "sneakers",
}

# Minimum net score to qualify as on-target (tuned for mixed datasets)
_CONFORMANCE_THRESHOLD = 2.0

# Deviation tags emitted for off-target images (WD/caption → tag)
_STYLE_DEVIATION_MAP = {
    # Art style deviations
    "anime": "anime_style",
    "cartoon": "cartoon_style",
    "comic": "comic_style",
    "manga": "manga_style",
    "chibi": "chibi_style",
    "sketch": "sketch_style",
    "lineart": "lineart_style",
    "3d_render": "3d_render_style",
    "watercolor": "watercolor_style",
    "oil_painting": "oil_painting_style",
    "pixel_art": "pixel_art_style",
    # Palette deviations
    "bright": "bright_palette",
    "pastel": "pastel_palette",
    "neon": "neon_palette",
    "vibrant": "vibrant_palette",
    # Lighting deviations
    "sunny": "bright_lighting",
    "blue_sky": "bright_lighting",
    "clear_sky": "bright_lighting",
    # Setting deviations
    "modern": "modern_setting",
    "urban": "urban_setting",
    "studio": "studio_background",
    "grey_background": "plain_background",
    "white_background": "plain_background",
    "simple_background": "plain_background",
}

# Default character traits that are OMITTED for on-target images
# (absence = default) but INCLUDED for off-target images (explicit learning)
_DEFAULT_TRAITS = {
    "red_brown_hair", "emerald_green_eyes", "pale_skin", "freckles",
    "pointy_ears", "barefoot", "wavy_hair", "past_shoulder_length_hair",
    "large_eyes",
}

# ---------------------------------------------------------------------------
#  Tags to suppress / remove
# ---------------------------------------------------------------------------
DOWNWEIGHT_PREFIXES = [
    "anime", "cartoon", "comic", "manga", "chibi", "sketch", "lineart",
]
REMOVE_TAGS = {"lowres", "bad_anatomy", "bad_hands", "error", "missing_fingers"}
# PROHIBITED_TAGS: things that are *factually impossible* for our character
# or are pure WD hallucination artefacts.  Style/palette/setting deviations
# are NOT prohibited — they become deviation tags for honest labeling.
PROHIBITED_TAGS = {
    # Footwear (elf is ALWAYS barefoot — canonical trait)
    "shoes", "sandals", "sneakers", "boots", "heels", "high_heels",
    "thigh_boots", "knee_boots", "loafers", "slippers", "socks",
    # Arm guards / gauntlets (she never wears these)
    "arm_guard", "gauntlet", "gauntlets", "armguard", "vambrace",
    "armored_gloves", "gloves",
    # Humans (only the viewer exists; no other humans in scene)
    "human", "humans", "people", "crowd", "boy", "man", "men",
    "multiple_girls", "multiple_boys", "group",
    # Gender mis-tags (she is female)
    "1boy", "2boys", "male_focus",
    # WD hallucination artefacts (wrong skin, wrong species, etc.)
    "blue_skin", "purple_skin", "green_skin", "grey_skin", "colored_skin",
    "dark_skin", "tan", "tanned",
    "colored_sclera",
    # WD character misidentification
    "ranni_the_witch",
    # WD misidentification (pleated green skirt + white shirt → school uniform)
    "school_uniform", "serafuku", "sailor_collar",
    # WD misidentification (vest + skirt → dress)
    "dress", "green_dress", "black_dress", "white_dress",
    # WD body hallucinations
    "extra_arms", "extra_legs", "horns", "antlers", "cracked_skin",
    "flat_chest", "body_writing",
    # Headwear (elf doesn't wear hats — WD confuses hoods/foliage for hats)
    "hat", "witch_hat", "santa_hat", "top_hat", "baseball_cap",
    # Violence (elf is non-violent)
    "gun", "sword", "weapon", "knife", "dagger",
    # Pure noise / irrelevant metadata
    "artist_name", "watermark", "signature", "copyright", "username",
    "text", "logo", "web_address",
    # Pose noise
    "v_arms",
}

# ---------------------------------------------------------------------------
#  Hair colour normalization  (keep only the best-scoring one)
# ---------------------------------------------------------------------------
HAIR_COLOR_TAGS = [
    "blonde_hair", "brown_hair", "auburn_hair", "red_hair", "orange_hair",
    "black_hair", "white_hair", "gray_hair", "silver_hair",
    "pink_hair", "blue_hair", "green_hair", "purple_hair",
    "red_brown_hair",
]
# WD commonly outputs blue_hair on stylised images — strongly prefer our canon
HAIR_COLOR_BIAS = {"auburn_hair": 0.25, "red_hair": 0.20, "brown_hair": 0.15, "red_brown_hair": 0.30}
HAIR_COLOR_SUPPRESS = {"blue_hair", "pink_hair", "green_hair", "purple_hair", "silver_hair", "white_hair"}

# Eye colour suppression — our elf always has emerald green eyes
EYE_COLOR_SUPPRESS = {"blue_eyes", "yellow_eyes", "red_eyes", "brown_eyes", "orange_eyes", "purple_eyes"}

# ---------------------------------------------------------------------------
#  WD → canonical tag remapping
# ---------------------------------------------------------------------------
WD_TAG_REMAP = {
    # Florence / WD may output generic equivalents; unify to our vocabulary
    "pointy_ears": "pointy_ears",
    "elf": "wood_elf",
    "elf_girl": "wood_elf",
    "elven": "wood_elf",
    "barefoot": "barefoot",
    "bare_feet": "barefoot",
    "feet": "barefoot",
    "outdoors": "outdoor",
    "nature": "outdoor",
    "forest": "old_growth_forest",
    "trees": "old_growth_forest",
    "woods": "old_growth_forest",
    "river": "mountain_stream",
    "stream": "mountain_stream",
    "waterfall": "waterfall",
    "mountain": "mountain",
    "fog": "mist",
    "mist": "mist",
    "moss": "mossy",
    "cabin": "rustic_cabin",
    "hut": "rustic_cabin",
    "deer": "deer",
    "bird": "bird",
    "owl": "owl",
    "wolf": "wolf",
    "rabbit": "rabbit",
    "fox": "fox",
    "magic": "magic",
    "spell": "casting_spell",
    "glowing": "magical_glow",
    "runes": "runes",
    # Background tag consolidation (avoid triple-stacking)
    "grey_background": "plain_background",
    "gray_background": "plain_background",
    "white_background": "plain_background",
    "simple_background": "plain_background",
}

# ---------------------------------------------------------------------------
#  Custom vocabulary biased toward our domain
# ---------------------------------------------------------------------------
CUSTOM_VOCAB = {
    "style": [
        "highly_detailed_digital_art",
        "alita_style",
        "cinematic_lighting",
        "sharp_focus",
        "volumetric_light",
        "depth_of_field",
        "muted_earthy_palette",
        "dark_palette",
    ],
    "setting": [
        "pacific_northwest",
        "medieval_fantasy",
        "old_growth_forest",
        "mossy",
        "misty",
        "ferns",
        "cedar_trees",
        "forest_trail",
        "mountain_stream",
        "waterfall",
        "mountain_fog",
        "rustic_cabin",
    ],
    "character": [
        "wood_elf",
        "female_protagonist",
        "young_adult",
        "pointy_ears",
        "large_eyes",
        "emerald_green_eyes",
        "freckles",
        "pale_skin",
        "red_brown_hair",
        "wavy_hair",
        "past_shoulder_length_hair",
        "hair_wisps_in_face",
        "slim_build",
        "barefoot",
        "magic_user",
        "plant_magic",
        "magical_glow",
        "pov",
    ],
    "attire": [
        "white_rustic_shirt",
        "deep_v_neck",
        "rolled_sleeves",
        "brown_corset",
        "shoulder_straps",
        "front_lacing",
        "green_vine_inlay",
        "short_green_skirt",
        "left_leg_slit",
        "green_boyshorts",
        "green_cloak",
        "cloak_brooch",
    ],
    "lighting": [
        "moody_lighting",
        "overcast",
        "filtered_light",
        "dappled_light",
        "firelight",
        "magical_light",
        "low_light",
        "candlelight",
        "moonlight",
        "twilight",
        "golden_hour",
        "shadows",
    ],
    "other": [],  # catch-all for uncategorised
}

# Category caps to avoid overlong captions while keeping diversity
CATEGORY_CAPS = {
    "style": 6,
    "setting": 8,
    "character": 12,
    "attire": 8,
    "lighting": 5,
    "other": 50,
}

# Florence-specific guardrails
FLORENCE_ONLY_MAX = 10  # cap for Florence-only tags that lack WD support
FLORENCE_STOPWORDS = {
    "image", "picture", "scene", "view", "someone", "something",
    "object", "things", "item", "photo", "photograph", "camera",
    "stock", "render",
}
FLORENCE_ALLOWED_ALONE = {
    # style / scene cues we want even if WD lacks labels
    "forest", "woods", "mist", "fog", "moss", "trail", "river",
    "riverbank", "water", "lighting", "cinematic_lighting",
    "volumetric_light", "sharp_focus", "photorealistic", "digital_art",
    "highly_detailed", "hyper_realistic", "cabin", "cave", "mountain",
    "waterfall", "stream", "lake", "ferns", "cedar", "mossy",
    "firelight", "torch", "campfire", "deer", "owl", "wolf", "fox",
    "rabbit", "bird", "raven", "magic", "spell", "runes", "glow",
}

# Cache paths and batch sizing for 3060 Ti / 32GB RAM
CACHE_DIR = Path(".cache/tagging")
WD_BATCH_SIZE = 6  # fits 8GB VRAM with fp16


@dataclass
class TagResult:
    tags: List[str]
    attributes: Dict[str, str]
    scores: Dict[str, float]


@lru_cache(maxsize=1)
def _load_wd_model(model_id: str = WD_MODEL_ID) -> Tuple[AutoImageProcessor, AutoModelForImageClassification, List[str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Guard against spurious SIGINT from the VS Code terminal during model init.
    with _ignore_sigint():
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id, torch_dtype=dtype)
        model = model.to(device).eval()

    # Reconcile processor input size with the model's actual patch_embed size.
    # transformers==4.49.0: processor advertises 448 but model expects 256.
    # transformers>=5.x:   both may agree on 448. Read the model to be safe.
    import timm.data as _timm_data  # already a transitive dep
    _data_cfg = dict(processor.data_config)
    _patch_embed = getattr(getattr(model, "timm_model", None), "patch_embed", None)
    _target_size = (
        tuple(_patch_embed.img_size)
        if _patch_embed is not None and hasattr(_patch_embed, "img_size")
        else (256, 256)
    )
    _proc_size = tuple(_data_cfg.get("input_size", (3, 448, 448))[1:])
    if _proc_size != _target_size:
        _data_cfg["input_size"] = (3,) + _target_size
        processor.data_config = _data_cfg
        processor.val_transforms = _timm_data.create_transform(**_data_cfg, is_training=False)

    id2label = model.config.id2label
    labels = [id2label[i] for i in range(len(id2label))]

    # Prefer the curated tag list from the repo (avoids generic label_x placeholders)
    # Order is critical: the CSV/JSON follow the classifier head ordering.
    try:
        tags_path = hf_hub_download(model_id, "selected_tags.json")
        with open(tags_path, "r", encoding="utf-8") as f:
            hf_labels = json.load(f)
        if isinstance(hf_labels, list) and len(hf_labels) == len(labels):
            labels = hf_labels
    except Exception:
        # fall back to CSV if JSON is absent (most public wd checkpoints ship the CSV)
        try:
            csv_path = hf_hub_download(model_id, "selected_tags.csv")
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                csv_labels: List[str] = []
                for row in reader:
                    name = str(row.get("name", "")).strip()
                    category_val = row.get("category")
                    try:
                        category = int(category_val) if category_val is not None else None
                    except Exception:
                        category = None
                    # Category 9 corresponds to rating tags; prefix to keep downstream logic working
                    if category == 9 and name:
                        name = f"rating:{name}"
                    if name:
                        csv_labels.append(name)
            if len(csv_labels) == len(labels):
                labels = csv_labels
        except Exception:
            # fall back to id2label mapping if download fails
            pass

    return processor, model, labels


# ---------------------------------------------------------------------------
#  Signal guard – VS Code terminals can send spurious SIGINT during long
#  model init (timm trunc_normal_ → tensor.uniform_).  We temporarily
#  ignore SIGINT while loading heavy models.
# ---------------------------------------------------------------------------
@contextmanager
def _ignore_sigint():
    """Temporarily ignore SIGINT on Windows to survive VS Code terminal noise."""
    if sys.platform == "win32":
        prev = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, prev)
    else:
        yield


@lru_cache(maxsize=1)
def _load_florence(model_id: str = FLORENCE_MODEL_ID) -> Tuple[AutoProcessor, AutoModelForCausalLM]:
    """Load the Florence-2 PromptGen model and processor.

    Requires ``transformers==4.49.0`` (pinned in environment.yml / requirements.txt)
    so that the checkpoint's custom ``trust_remote_code`` modules load cleanly with
    no monkey-patches needed.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    with _ignore_sigint():
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(device).eval()

    return processor, model


@lru_cache(maxsize=1)
def _load_llava(model_id: str = LLAVA_MODEL_ID) -> Tuple[InstructBlipProcessor, InstructBlipForConditionalGeneration]:
    """Load InstructBLIP (flan-t5-large) as an uncensored captioning fallback."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    with _ignore_sigint():
        processor = InstructBlipProcessor.from_pretrained(model_id, use_fast=False)
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )
        model = model.to(device).eval()

    return processor, model


def _normalize_tag(tag: str) -> str:
    cleaned = tag.strip().lower().replace(" ", "_").replace("-", "_")
    cleaned = cleaned.strip(",.;:()[]{}\n\t")
    return cleaned


def _hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _infer_wd(path: Path, processor, model, labels: List[str], image_obj: Optional[Image.Image] = None) -> Dict[str, float]:
    device = next(model.parameters()).device
    with ExitStack() as stack:
        if image_obj is not None:
            img_ctx = image_obj
            if img_ctx.mode != "RGB":
                img_ctx = img_ctx.convert("RGB")
                stack.callback(img_ctx.close)
        else:
            opened = stack.enter_context(Image.open(path))
            img_ctx = opened.convert("RGB")
            stack.callback(img_ctx.close)

        inputs = processor(images=img_ctx, return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits[0]
            probs = logits.sigmoid().cpu().tolist()
        return {label: probs[idx] for idx, label in enumerate(labels)}


def _caption_with_florence(path: Path, processor, model, image_obj: Optional[Image.Image] = None) -> Tuple[List[str], str]:
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    with ExitStack() as stack:
        if image_obj is not None:
            img_ctx = image_obj
            if img_ctx.mode != "RGB":
                img_ctx = img_ctx.convert("RGB")
                stack.callback(img_ctx.close)
        else:
            opened = stack.enter_context(Image.open(path))
            img_ctx = opened.convert("RGB")
            stack.callback(img_ctx.close)

        def _run_task(img_input, task: str, max_new_tokens: int) -> str:
            inputs = processor(text=task, images=img_input, return_tensors="pt")
            inputs = {
                key: (value.to(device=device, dtype=model_dtype) if torch.is_floating_point(value) else value.to(device))
                for key, value in inputs.items()
            }

            def _patch_missing_gen_attrs() -> None:
                cfgs = [
                    getattr(model, "config", None),
                    getattr(model, "generation_config", None),
                    getattr(getattr(model, "config", None), "text_config", None),
                    getattr(model, "language_model", None) and getattr(model.language_model, "config", None),
                ]
                for cfg in cfgs:
                    if cfg is None:
                        continue
                    for attr in ("forced_bos_token_id", "forced_eos_token_id"):
                        if hasattr(cfg, attr):
                            continue
                        # Try setting the attribute directly; Florence configs sometimes use slots-only types.
                        try:
                            setattr(cfg, attr, None)
                            continue
                        except Exception:
                            pass
                        try:
                            object.__setattr__(cfg, attr, None)
                            continue
                        except Exception:
                            pass
                        try:
                            setattr(cfg.__class__, attr, None)
                            continue
                        except Exception:
                            pass
                        try:
                            # As a last resort, install a permissive __getattr__ to return None for these fields
                            # instead of raising AttributeError when the underlying config refuses new attributes.
                            def _fallback(self, name, _attr=attr):  # pragma: no cover - defensive shim
                                if name == _attr:
                                    return None
                                raise AttributeError(name)

                            setattr(cfg.__class__, "__getattr__", _fallback)
                        except Exception:
                            pass

            _patch_missing_gen_attrs()

            def _generate(token_cap: int) -> torch.Tensor:
                return model.generate(
                    **inputs,
                    max_new_tokens=token_cap,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.05,
                    use_cache=False,
                )

            with torch.inference_mode():
                try:
                    generated_ids = _generate(token_cap=max_new_tokens)
                except AttributeError as exc:
                    if "forced_bos_token_id" in str(exc) or "forced_eos_token_id" in str(exc):
                        _patch_missing_gen_attrs()
                        generated_ids = _generate(token_cap=max_new_tokens)
                    else:
                        raise
                except RuntimeError as exc:
                    if "CUDA out of memory" in str(exc):
                        torch.cuda.empty_cache()
                        token_cap = min(max_new_tokens, 512)
                        generated_ids = _generate(token_cap=token_cap)
                    else:
                        raise

            return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        def _is_repetitive(text: str) -> bool:
            words = text.lower().split()
            if not words:
                return True
            if len(words) < 10:
                return False

            counts = Counter(words)
            most_common_ratio = counts.most_common(1)[0][1] / len(words)
            if most_common_ratio > 0.35:
                return True
            if len(words) >= 40:
                top3_ratio = sum(c for _, c in counts.most_common(3)) / len(words)
                if top3_ratio > 0.55:
                    return True

            if re.search(r"(\b\w{3,10}\b)(\s+\1){3,}", text.lower()):
                return True

            ngrams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
            return any(count > 2 for count in Counter(ngrams).values())

        tags_str = _run_task(img_ctx, "<GENERATE_TAGS>", max_new_tokens=192)
        caption_text = _run_task(img_ctx, "<MORE_DETAILED_CAPTION>", max_new_tokens=512)

        raw_tags = [t.strip() for t in re.split(r"[,;\n]+", tags_str) if t.strip()]
        tags: List[str] = []
        seen = set()
        for t in raw_tags:
            nt = _normalize_tag(t)
            if nt and nt not in seen:
                seen.add(nt)
                tags.append(nt)

        words = caption_text.split()
        if len(words) > 140:
            caption_text = " ".join(words[:140])
        caption_text = " ".join(caption_text.split())

        if _is_repetitive(caption_text):
            raise ValueError("florence_low_quality_caption")

        if not caption_text:
            raise ValueError("florence_empty_caption")

        return tags, caption_text


def _caption_with_llava(path: Path, processor, model, image_obj: Optional[Image.Image] = None) -> Tuple[List[str], str]:
    """Caption with LLaVA as an uncensored, general fallback."""

    def _clean_caption_text(text: str) -> str:
        cleaned = " ".join(text.split())
        cleaned = re.sub(r"(\b\w+\b)(\s+\1){2,}", r"\1 \1", cleaned, flags=re.IGNORECASE)
        return cleaned.strip(" .,\t\n")

    def _has_repeated_chunk(token: str) -> bool:
        lowered = token.lower()
        return bool(re.search(r"([a-z]{2,4})\1{2,}", lowered))

    def _is_low_quality_caption(text: str) -> bool:
        tokens = re.findall(r"[A-Za-z]+", text.lower())
        if not tokens:
            return True
        unique = set(tokens)
        if len(unique) <= 3:
            return True

        normalized_text = " ".join(tokens)
        subject_ok = bool(re.search(r"\b(elf|fae|fairy|faerie|girl|woman|female|lady)\b", normalized_text))
        content_ok = bool(re.search(r"\b(body|skin|nude|breast|breasts|chest|legs|hair|eyes|ears|face|shirt|dress|cloak|cape|skirt|pose|standing|sitting)\b", normalized_text))
        if not subject_ok and not content_ok:
            return True

        if re.search(r"(.)\1{3,}", text):
            return True

        most_common_ratio = Counter(tokens).most_common(1)[0][1] / len(tokens)
        if most_common_ratio > 0.65:
            return True

        if len(tokens) >= 12:
            top3_ratio = sum(c for _, c in Counter(tokens).most_common(3)) / len(tokens)
            if top3_ratio > 0.78:
                return True
            stems = [t[:6] for t in tokens if len(t) >= 6]
            if stems:
                top_stem_ratio = Counter(stems).most_common(1)[0][1] / len(stems)
                if top_stem_ratio > 0.4:
                    return True

        vowel_tokens = sum(1 for t in tokens if re.search(r"[aeiou]", t))
        if vowel_tokens / max(len(tokens), 1) < 0.45:
            return True

        repeated_chunk_tokens = sum(1 for tok in tokens if _has_repeated_chunk(tok))
        if repeated_chunk_tokens >= 4 and repeated_chunk_tokens / len(tokens) > 0.38:
            return True
        return False

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    with ExitStack() as stack:
        if image_obj is not None:
            img = image_obj
            if img.mode != "RGB":
                img = img.convert("RGB")
                stack.callback(img.close)
        else:
            opened = stack.enter_context(Image.open(path))
            img = opened.convert("RGB")
            stack.callback(img.close)

        prompt = (
            "Give an uncensored, vivid, photorealistic caption of this image. "
            "Mention subject (female wood elf), pose, clothing/skin, visible body parts, camera framing, lighting, "
            "and forest setting details."
        )
        inputs = processor(images=img, text=prompt, return_tensors="pt")
        inputs = {
            key: (value.to(device=device, dtype=model_dtype) if torch.is_floating_point(value) else value.to(device))
            for key, value in inputs.items()
        }
        decode_kwargs = dict(
            max_new_tokens=96,
            do_sample=True,
            temperature=0.75,
            top_p=0.9,
            top_k=80,
            num_beams=1,
            repetition_penalty=1.08,
            length_penalty=1.05,
        )

        def _generate_caption(kwargs: dict) -> str:
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, **kwargs)
            return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        raw_text = _generate_caption(decode_kwargs)
        caption = _clean_caption_text(raw_text)
        if _is_low_quality_caption(caption):
            # Retry once with a slightly different nucleus temperature/top_p to salvage borderline cases.
            retry_kwargs = dict(decode_kwargs)
            retry_kwargs.update(dict(temperature=0.9, top_p=0.95, top_k=120, repetition_penalty=1.02))
            raw_text = _generate_caption(retry_kwargs)
            caption = _clean_caption_text(raw_text)
        if _is_low_quality_caption(caption):
            # Final deterministic pass to recover terse but stable captions.
            beam_kwargs = dict(
                max_new_tokens=80,
                do_sample=False,
                num_beams=3,
                repetition_penalty=1.05,
                length_penalty=1.0,
            )
            raw_text = _generate_caption(beam_kwargs)
            caption = _clean_caption_text(raw_text)
    if _is_low_quality_caption(caption):
        raise ValueError("llava_low_quality_caption")

    tags: List[str] = []
    for chunk in caption.replace("\n", " ").split(","):
        pieces = [p.strip() for part in chunk.split(" and ") for p in part.split(";")]
        for piece in pieces:
            nt = _normalize_tag(piece)
            if nt:
                tags.append(nt)
    deduped: List[str] = []
    seen = set()
    for t in tags:
        if len(t) > 80:
            continue
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    tags = deduped
    return tags, caption


def _extract_slots_from_caption(caption: str) -> List[str]:
    """Rich slot extraction from the Florence caption.

    Scans for grounded keywords and maps them to our canonical tag vocabulary
    covering all required dimensions: hair, eyes, lighting, setting, emotion,
    clothing, body, pose, zoom, wetness, NSFW, animals, gaze, and unique
    elements.
    """

    text = caption.lower()
    slots: List[str] = []

    # ------------------------------------------------------------------
    # Hair
    # ------------------------------------------------------------------
    hair_map = {
        "auburn": "red_brown_hair",
        "reddish-brown": "red_brown_hair",
        "reddish brown": "red_brown_hair",
        "red hair": "red_brown_hair",
        "brown hair": "red_brown_hair",
        "brunette": "red_brown_hair",
        "ginger": "red_brown_hair",
        "blonde": "blonde_hair",
        "black hair": "black_hair",
        "dark hair": "red_brown_hair",
    }
    for key, tag in hair_map.items():
        if key in text:
            slots.append(tag)
            break
    if any(k in text for k in ("wavy", "curls", "curly", "waves")):
        slots.append("wavy_hair")
    if any(k in text for k in ("past her shoulder", "cascading", "long hair", "flowing hair")):
        slots.append("past_shoulder_length_hair")
    if any(k in text for k in ("wisps", "stray hair", "hair in her face", "loose strands")):
        slots.append("hair_wisps_in_face")

    # ------------------------------------------------------------------
    # Eyes
    # ------------------------------------------------------------------
    eye_map = {
        "green eyes": "emerald_green_eyes",
        "emerald": "emerald_green_eyes",
        "vivid green": "emerald_green_eyes",
        "blue eyes": "blue_eyes",
        "brown eyes": "brown_eyes",
        "hazel": "hazel_eyes",
    }
    for key, tag in eye_map.items():
        if key in text:
            slots.append(tag)
            break
    if any(k in text for k in ("large eyes", "big eyes", "wide eyes", "expressive eyes")):
        slots.append("large_eyes")

    # ------------------------------------------------------------------
    # Skin / face
    # ------------------------------------------------------------------
    if any(k in text for k in ("pale skin", "fair skin", "light skin", "porcelain")):
        slots.append("pale_skin")
    if "freckle" in text:
        slots.append("freckles")
    if any(k in text for k in ("pointy ear", "pointed ear", "elf ear", "elven ear")):
        slots.append("pointy_ears")

    # ------------------------------------------------------------------
    # Facial expression / emotion
    # ------------------------------------------------------------------
    expr_map = {
        "smile": "soft_smile",
        "smiling": "soft_smile",
        "grin": "grinning",
        "serious": "serious_expression",
        "stern": "serious_expression",
        "focused": "focused_expression",
        "pensive": "pensive_expression",
        "curious": "curious_expression",
        "surprise": "surprised_expression",
        "playful": "playful_expression",
        "serene": "serene_expression",
        "sad": "sad_expression",
        "contemplat": "contemplative_expression",
        "confident": "confident_expression",
        "shy": "shy_expression",
        "mischiev": "mischievous_expression",
    }
    for key, tag in expr_map.items():
        if key in text:
            slots.append(tag)

    # ------------------------------------------------------------------
    # Clothing / attire
    # ------------------------------------------------------------------
    clothing_map = {
        "corset": "brown_corset",
        "lace-up": "front_lacing",
        "lace up": "front_lacing",
        "front lac": "front_lacing",
        "strap": "shoulder_straps",
        "cloak": "green_cloak",
        "hood": "hooded_cloak",
        "brooch": "cloak_brooch",
        "broach": "cloak_brooch",
        "skirt": "short_green_skirt",
        "slit": "left_leg_slit",
        "boyshort": "green_boyshorts",
        "boy short": "green_boyshorts",
        "short shorts": "green_boyshorts",
        "shirt": "white_rustic_shirt",
        "blouse": "white_rustic_shirt",
        "deep v": "deep_v_neck",
        "v-neck": "deep_v_neck",
        "v neck": "deep_v_neck",
        "vine": "green_vine_inlay",
        "rolled sleeve": "rolled_sleeves",
        "sleeves rolled": "rolled_sleeves",
    }
    for key, tag in clothing_map.items():
        if key in text:
            slots.append(tag)
    # Nudity / partial nudity
    if any(k in text for k in ("topless", "nude", "naked", "bare chest", "bare breast",
                                "no clothing", "unclothed", "undressed")):
        slots.append("nude")
    if any(k in text for k in ("partial", "semi-nude", "barely", "sheer")):
        slots.append("partial_nudity")

    # ------------------------------------------------------------------
    # Body description
    # ------------------------------------------------------------------
    if any(k in text for k in ("slim", "slender", "lean", "lithe", "petite")):
        slots.append("slim_build")
    if any(k in text for k in ("small breast", "flat chest", "small to medium")):
        slots.append("small_breasts")
    if "medium breast" in text or "medium-sized breast" in text:
        slots.append("medium_breasts")
    if any(k in text for k in ("large breast", "big breast")):
        slots.append("large_breasts")
    if any(k in text for k in ("muscular", "toned", "athletic")):
        slots.append("athletic_build")
    if "barefoot" in text or "bare feet" in text or "no shoes" in text:
        slots.append("barefoot")

    # ------------------------------------------------------------------
    # Body position / pose
    # ------------------------------------------------------------------
    pose_map = {
        "standing": "standing",
        "sitting": "sitting",
        "kneeling": "kneeling",
        "crouching": "crouching",
        "lying": "lying_down",
        "leaning": "leaning",
        "bending": "bending_over",
        "squatting": "squatting",
        "walking": "walking",
        "running": "running",
        "climbing": "climbing",
        "swimming": "swimming",
        "reclining": "reclining",
        "stretching": "stretching",
    }
    for key, tag in pose_map.items():
        if key in text:
            slots.append(tag)

    # ------------------------------------------------------------------
    # Camera / zoom level
    # ------------------------------------------------------------------
    if any(k in text for k in ("close-up", "close up", "closeup", "headshot")):
        slots.append("close_up")
    elif any(k in text for k in ("upper body", "bust shot", "portrait")):
        slots.append("upper_body_shot")
    elif "cowboy" in text or "waist up" in text or "waist-up" in text:
        slots.append("medium_shot")
    elif any(k in text for k in ("full body", "full-body", "head to toe")):
        slots.append("full_body_shot")
    elif any(k in text for k in ("wide shot", "wide angle", "panoram")):
        slots.append("wide_shot")

    # ------------------------------------------------------------------
    # Camera angle / perspective
    # ------------------------------------------------------------------
    if any(k in text for k in ("from above", "bird's eye", "top-down", "looking down")):
        slots.append("high_angle")
    elif any(k in text for k in ("from below", "low angle", "looking up", "worm's eye")):
        slots.append("low_angle")
    elif any(k in text for k in ("from behind", "back view", "rear view")):
        slots.append("from_behind")
    elif any(k in text for k in ("from side", "profile", "side view")):
        slots.append("side_view")
    elif any(k in text for k in ("front view", "facing", "facing the viewer")):
        slots.append("front_view")

    # ------------------------------------------------------------------
    # Gaze direction
    # ------------------------------------------------------------------
    if any(k in text for k in ("looking at viewer", "looking at the viewer",
                                "looking at camera", "looking directly")):
        slots.append("looking_at_viewer")
    elif any(k in text for k in ("looking away", "looking to the side", "averted gaze")):
        slots.append("looking_away")
    elif any(k in text for k in ("looking down", "downcast eyes", "eyes closed")):
        slots.append("looking_down")
    elif any(k in text for k in ("looking up", "gazing up")):
        slots.append("looking_up")

    # ------------------------------------------------------------------
    # Lighting / time of day / shadows
    # ------------------------------------------------------------------
    if any(k in text for k in ("moon", "moonlight", "night")):
        slots.append("moonlight")
        slots.append("night")
    if any(k in text for k in ("sunset", "twilight", "dawn", "dusk")):
        slots.append("twilight")
    if any(k in text for k in ("golden hour", "warm light")):
        slots.append("golden_hour")
    if any(k in text for k in ("glow", "magic light", "spell light", "rune")):
        slots.append("magical_light")
    if any(k in text for k in ("fire", "torch", "campfire", "hearth", "candle")):
        slots.append("firelight")
    if any(k in text for k in ("shadow", "shadows")):
        slots.append("shadows")
    if any(k in text for k in ("overcast", "cloudy", "cloud")):
        slots.append("overcast")
    if any(k in text for k in ("fog", "mist", "haze", "misty")):
        slots.append("mist")
    if any(k in text for k in ("dappled", "filtered", "sunlight through", "light through")):
        slots.append("filtered_light")
    if any(k in text for k in ("dark", "dim", "low light")):
        slots.append("low_light")
    if "moody" in text:
        slots.append("moody_lighting")
    # Detect bright/sunny lighting honestly (for deviation scoring)
    if any(k in text for k in ("bright", "sunny", "sunlit", "clear day")):
        slots.append("bright_lighting")

    # ------------------------------------------------------------------
    # Setting / environment / background
    # ------------------------------------------------------------------
    if any(k in text for k in ("forest", "woods", "grove", "trees", "tree", "cedar", "fir", "pine")):
        slots.append("old_growth_forest")
    if any(k in text for k in ("river", "stream", "brook", "creek")):
        slots.append("mountain_stream")
    if "waterfall" in text:
        slots.append("waterfall")
    if any(k in text for k in ("mountain", "cliff", "peak")):
        slots.append("mountain")
    if any(k in text for k in ("cabin", "hut", "cottage", "lodge")):
        slots.append("rustic_cabin")
    if "cave" in text or "cavern" in text:
        slots.append("cave")
    if any(k in text for k in ("lake", "pond")):
        slots.append("lake")
    if any(k in text for k in ("moss", "mossy")):
        slots.append("mossy")
    if "fern" in text:
        slots.append("ferns")
    if any(k in text for k in ("wooden plank", "wooden wall", "log wall")):
        slots.append("rustic_cabin")

    # ------------------------------------------------------------------
    # Season
    # ------------------------------------------------------------------
    if any(k in text for k in ("autumn", "fall", "orange leaves", "red leaves")):
        slots.append("autumn")
    elif any(k in text for k in ("winter", "snow", "frost")):
        slots.append("winter")
    elif any(k in text for k in ("spring", "blossom", "bloom")):
        slots.append("spring")
    elif any(k in text for k in ("summer", "lush", "verdant")):
        slots.append("summer")

    # ------------------------------------------------------------------
    # Wetness
    # ------------------------------------------------------------------
    if any(k in text for k in ("wet", "damp", "dripping", "soaked", "rain",
                                "dewy", "glistening", "splash")):
        slots.append("wet")
    else:
        slots.append("dry")

    # ------------------------------------------------------------------
    # Activity / interaction
    # ------------------------------------------------------------------
    # Use regex word boundaries to avoid false positives (e.g. "magical" ≠ "magic",
    # "casting shadows" ≠ "casting spell", "creating" ≠ "eating")
    if re.search(r'\b(?:casting\s+(?:a\s+)?spell|conjur|spellcast)', text):
        slots.append("casting_spell")
        slots.append("magic_user")
    if any(k in text for k in ("reading", "book", "scroll", "tome")):
        slots.append("reading")
    if any(k in text for k in ("reaching", "touching", "holding")):
        slots.append("interacting_with_environment")
    if any(k in text for k in ("bathing", "bath", "washing")):
        slots.append("bathing")
    if re.search(r'\b(?:eating|drinking|cup|goblet)\b', text):
        slots.append("drinking")
    if any(k in text for k in ("sleeping", "napping", "resting", "asleep")):
        slots.append("resting")
    if any(k in text for k in ("meditating", "praying", "ritual")):
        slots.append("meditating")
    if any(k in text for k in ("gathering", "picking", "herbs", "mushroom", "berries")):
        slots.append("foraging")
    if any(k in text for k in ("playing", "instrument", "flute", "lute", "music")):
        slots.append("playing_music")

    # ------------------------------------------------------------------
    # NSFW sexual context
    # ------------------------------------------------------------------
    nsfw_map = {
        "breast": "breasts_visible",
        "nipple": "nipples_visible",
        "areola": "nipples_visible",
        "buttock": "buttocks_visible",
        "butt": "buttocks_visible",
        "genital": "genitals_visible",
        "vulva": "genitals_visible",
        "pussy": "genitals_visible",
        "groin": "genitals_visible",
        "pubic": "genitals_visible",
        "spread": "spread_pose",
        "explicit": "explicit",
        "provocative": "provocative_pose",
        "sensual": "sensual",
        "intimate": "intimate",
        "seductive": "seductive_pose",
    }
    for key, tag in nsfw_map.items():
        if key in text:
            slots.append(tag)

    # ------------------------------------------------------------------
    # Animals in scene
    # ------------------------------------------------------------------
    animal_map = {
        "deer": "deer", "stag": "deer", "doe": "deer",
        "owl": "owl", "raven": "raven", "crow": "raven",
        "wolf": "wolf", "fox": "fox", "rabbit": "rabbit",
        "hare": "rabbit", "bird": "bird", "squirrel": "squirrel",
        "bear": "bear", "hawk": "hawk", "eagle": "eagle",
        "butterfly": "butterfly", "moth": "moth",
        "frog": "frog", "toad": "frog", "snake": "snake",
        "fish": "fish", "salmon": "fish",
    }
    for key, tag in animal_map.items():
        if key in text:
            slots.append(tag)

    # ------------------------------------------------------------------
    # Unique / special elements
    # ------------------------------------------------------------------
    if any(k in text for k in ("tattoo", "marking", "body paint")):
        slots.append("body_markings")
    if any(k in text for k in ("scar", "wound")):
        slots.append("scar")
    if any(k in text for k in ("flower", "floral", "wreath", "garland")):
        slots.append("flowers")
    if any(k in text for k in ("mushroom", "fungi")):
        slots.append("mushrooms")
    if any(k in text for k in ("crystal", "gem", "jewel")):
        slots.append("crystals")
    if any(k in text for k in ("vine", "ivy", "tendril")):
        if "green_vine_inlay" not in slots:
            slots.append("vines")

    # Deduplicate while preserving order
    seen: set = set()
    deduped: List[str] = []
    for s in slots:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


# ---------------------------------------------------------------------------
#  Caption post-processing
# ---------------------------------------------------------------------------

# Substitutions applied to the raw Florence / LLaVA caption to enforce our
# visual identity and suppress contradictions with the target aesthetic.
_CAPTION_SUBS: List[Tuple[str, str]] = [
    # Medium: force digital art framing, never "photograph"
    (r"\b(?:a\s+)?high[- ]resolution\s+photograph\b", "a highly detailed digital illustration"),
    (r"\bthe\s+photograph\b", "the digital illustration"),
    (r"\b(?:a\s+)?(?:digital\s+)?photograph\b", "a highly detailed digital illustration"),
    (r"\bthe\s+photo\b", "the digital illustration"),
    (r"\b(?:a\s+)?(?:digital\s+)?photo\b", "a highly detailed digital illustration"),
    (r"\bphotorealistic\b", "highly detailed"),
    (r"\bstock image\b", "digital illustration"),
    (r"\b3d render\b", "digital illustration"),
    # Subject: never "doll", always our elf  (order: "the X" first, then "a X")
    (r"\bthe\s+doll\b", "the young wood elf"),
    (r"\b(?:a\s+)?doll\b", "a young wood elf"),
    (r"\bthe\s+mannequin\b", "the young wood elf"),
    (r"\b(?:a\s+)?mannequin\b", "a young wood elf"),
    (r"\byoung\s+caucasian\s+woman\b", "young wood elf"),
    (r"\bcaucasian\s+woman\b", "wood elf"),
    # "of caucasian descent" / "is of caucasian descent" — strip entirely
    (r",?\s*(?:likely\s+)?(?:is\s+)?of\s+caucasian\s+descent\b", ""),
    (r"\byoung,?\s+nude\s+woman\b", "young, nude wood elf"),
    (r"\bnude\s+woman\b", "nude wood elf"),
    (r"\bnude,?\s+young\s+woman\b", "nude, young wood elf"),
    (r"\byoung\s+elf\s+woman\b", "young wood elf"),
    (r"\belf\s+woman\b", "wood elf"),
    (r"\byoung woman\b", "young wood elf"),
    (r"\bthe\s+woman\b", "the wood elf"),
    (r"\ba\s+woman\b", "a wood elf"),
    # Keep "she is/she has" — it reads more naturally than forced noun replacement
    # Hair: enforce canonical red-brown
    (r"\bblonde hair\b", "red-brown wavy hair"),
    (r"\bblack hair\b", "red-brown wavy hair"),
    (r"\bblue hair\b", "red-brown wavy hair"),
    (r"\bpink hair\b", "red-brown wavy hair"),
    (r"\bsilver hair\b", "red-brown wavy hair"),
    (r"\bwhite hair\b", "red-brown wavy hair"),
    # Eyes: enforce emerald green
    (r"\bblue eyes\b", "emerald green eyes"),
    (r"\bbrown eyes\b", "emerald green eyes"),
    (r"\byellow eyes\b", "emerald green eyes"),
    (r"\bred eyes\b", "emerald green eyes"),
    # Skin: enforce pale
    (r"\btanned\b", "pale"),
    (r"\bdark skin\b", "pale skin"),
    (r"\btan skin\b", "pale skin"),
    (r"\bsmooth and tanned\b", "smooth and pale"),
    # Lighting: suppress bright / sunny
    (r"\bsunlight streaming\b", "filtered light streaming"),
    (r"\bsunlight\b", "filtered light"),
    (r"\bbright\s+and\s+sunny\b", "overcast and moody"),
    (r"\bsunny\b", "overcast"),
    (r"\bclear blue sky\b", "overcast grey sky"),
    (r"\bblue sky\b", "grey overcast sky"),
    (r"\bbright light\b", "soft filtered light"),
    (r"\bbright colors?\b", "muted tones"),
    (r"\bbright\s+and\b", "soft and"),
    # Setting anchors
    (r"\bbackyard\b", "forest clearing"),
    (r"\bgarden\b", "forest clearing"),
    (r"\bstudio\b", "rustic cabin interior"),
    (r"\bplain\s*,?\s*(light\s+)?gray\s*(surface|background)\b", "weathered wooden wall"),
    (r"\bgray background\b", "muted forest backdrop"),
    (r"\bgrey background\b", "muted forest backdrop"),
    (r"\bneutral gray\b", "muted earthy tones"),
    (r"\bneutral grey\b", "muted earthy tones"),
    (r"\bwhite background\b", "misty forest backdrop"),
    (r"\bplain background\b", "natural forest backdrop"),
    # Modern / urban
    (r"\bcity\b", "forest"),
    (r"\burban\b", "wilderness"),
    (r"\bmodern\b", "medieval"),
    (r"\bcontemporary\b", "medieval"),
    # Florence contradictions
    (r",?\s*with no shoes or clothing on\b", ""),  # contradicts outfit just described
    (r",?\s*with no shoes\b", ""),  # she's always barefoot, redundant
    # "likely" / "appears to be" hedging — make assertions direct
    (r"\blikely\s+(?=a\s+)", ""),  # "likely a young" → "a young"
    (r"\bis likely\b", "is"),
    (r"\bappears to be\b", "is"),
    (r"\bseems to be\b", "is"),
    # Subject: replace generic "the subject" with "the wood elf"
    (r"\bthe subject's\b", "the wood elf's"),
    (r"\bthe subject\b", "the wood elf"),
    # Remove Florence filler phrases about "the subject as a young individual"
    (r",?\s*emphasizing the wood elf as a young individual\b", ""),
    (r",?\s*emphasizing the wood elf's natural beauty\b", ""),
    # Grammar fixes
    (r"\bthe overall mood of\b", "the overall mood is"),
    # Florence stutter / double-start safety net (must run last)
    (r"\ba digital a highly\b", "a highly"),
    (r"\bdigital a highly detailed digital\b", "highly detailed digital"),
]

# Style prefix prepended to every caption
_CAPTION_PREFIX = "highly detailed digital art, medieval fantasy, pacific northwest, "
# Palette / lighting suffix appended when not already present
_CAPTION_SUFFIX = ", muted earthy palette, moody lighting"


def _postprocess_caption(raw_caption: str) -> str:
    """Clean up the raw model caption: identity/setting subs, grammar fixes, trim.

    This produces the honest, cleaned caption text.  Style framing (prefix,
    suffix, deviation markers) is handled by ``_build_caption`` based on the
    conformance tier.
    """

    text = raw_caption.strip()

    # Apply all substitutions
    for pattern, repl in _CAPTION_SUBS:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # Safety net: fix double-article artifacts ("the a young" → "a young")
    text = re.sub(r"\bthe\s+a\s+young\b", "a young", text, flags=re.IGNORECASE)
    text = re.sub(r"\ban\s+a\s+", "a ", text, flags=re.IGNORECASE)

    # Collapse multiple spaces
    text = " ".join(text.split())

    # Trim to sentence boundary (avoid mid-word truncation)
    # Find the last sentence-ending punctuation within ~180 words
    words = text.split()
    if len(words) > 180:
        text = " ".join(words[:180])

    # Find last sentence end
    last_period = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if last_period > len(text) * 0.4:  # only trim if we keep >40% of text
        text = text[: last_period + 1]

    # Strip trailing comma fragments  (e.g. ", the overall style" or "is taken in a,")
    text = text.rstrip(" ,;:")

    # Remove trailing dangling clause after last complete sentence
    # Pattern: sentence-ending punctuation followed by a non-sentence fragment
    trail_match = re.search(r"([.!?])\s+[^.!?]{0,80}$", text)
    if trail_match:
        fragment = text[trail_match.start(1) + 1 :].strip()
        # If the trailing fragment ends without sentence-ending punctuation, chop it
        if fragment and not re.search(r"[.!?]$", fragment):
            text = text[: trail_match.start(1) + 1]

    # Catch comma-clause run-ons that lack ANY sentence-ending punctuation.
    # Florence often generates one giant comma-separated run-on.  If the text
    # has no period at all, find the last semantically complete comma clause.
    if "." not in text and "!" not in text and "?" not in text:
        # Find last comma that's followed by a determiner/preposition
        # (signals a new clause), then trim everything after it
        m = re.search(r",\s+(?:the|a|an|with|in|her|his|this|and)\s+\S+[^,]{0,50}$", text)
        if m and m.start() > len(text) * 0.5:
            text = text[: m.start()]

    # Final strip of any remaining trailing punctuation artifacts
    text = text.rstrip(" ,;:")

    return text


def _score_conformance(
    wd_scores: Dict[str, float],
    florence_tags: Sequence[str],
    raw_caption: str,
) -> Tuple[float, bool, List[str]]:
    """Score how well an image matches the target aesthetic.

    Returns:
        (score, is_on_target, deviation_tags)
        - score: net conformance score (positive = on-target direction)
        - is_on_target: True when score >= threshold
        - deviation_tags: list of style-deviation tags for off-target images
    """
    score = 0.0
    # Only consider WD tags that passed the base threshold
    confident_wd = {k for k, v in wd_scores.items() if v >= TAG_THRESHOLD}
    all_tags = confident_wd | set(florence_tags)
    caption_lower = raw_caption.lower() if raw_caption else ""

    for signal in _CONFORMANCE_POSITIVE:
        if signal in all_tags:
            score += 1.0
        elif re.search(r"\b" + re.escape(signal) + r"\b", caption_lower):
            score += 0.5

    for signal in _CONFORMANCE_NEGATIVE:
        if signal in all_tags:
            score -= 1.5  # negative signals weigh more (err on side of caution)
        elif re.search(r"\b" + re.escape(signal) + r"\b", caption_lower):
            score -= 0.75

    # Bonus for core character evidence
    char_evidence = {"pointy_ears", "barefoot", "freckles", "green_eyes"}
    for ce in char_evidence:
        if ce in all_tags or re.search(r"\b" + re.escape(ce).replace("_", r"[\s_]") + r"\b", caption_lower):
            score += 0.5

    is_on_target = score >= _CONFORMANCE_THRESHOLD

    # Collect deviation tags for off-target images
    deviation_tags: List[str] = []
    if not is_on_target:
        for signal, dev_tag in _STYLE_DEVIATION_MAP.items():
            if signal in all_tags or re.search(r"\b" + re.escape(signal) + r"\b", caption_lower):
                if dev_tag not in deviation_tags:
                    deviation_tags.append(dev_tag)
        # If no specific deviation detected, add a generic marker
        if not deviation_tags:
            deviation_tags.append("non_target_style")

    return score, is_on_target, deviation_tags


def _build_caption(raw_caption: str, is_on_target: bool, deviation_tags: List[str]) -> str:
    """Build the final training caption with style-aware framing.

    ON-TARGET images: Rewrite to ideal prose (prefix + cleaned caption + suffix).
        The model learns: this prose style = the desired aesthetic.
    OFF-TARGET images: Honest description with deviation markers prepended.
        The model learns: these deviation markers = this different look.
    """
    if not raw_caption:
        return ""

    # Always run the honest-caption cleanup (identity subs, grammar fixes)
    text = _postprocess_caption(raw_caption)

    if is_on_target:
        # --- On-target: full rewrite into ideal training format ---
        prefix = _CAPTION_PREFIX
        suffix = _CAPTION_SUFFIX
        # Don't double-add suffix keywords
        lower = text.lower()
        if "muted" in lower and "palette" in lower:
            suffix = ""
        if "moody" in lower and "lighting" in lower:
            suffix = suffix.replace(", moody lighting", "")
        return prefix + text + suffix
    else:
        # --- Off-target: honest caption with deviation markers ---
        # Prepend deviation descriptors so the model associates them
        # with the visual style it sees
        dev_prefix = ", ".join(deviation_tags)
        return dev_prefix + ", " + text


def _merge_tags(
    wd_scores: Dict[str, float],
    florence_tags: Sequence[str],
    slot_tags: Sequence[str],
    rating_tags: Dict[str, float],
    is_on_target: bool = True,
    deviation_tags: Optional[List[str]] = None,
) -> Tuple[List[str], Dict[str, float], Dict[str, str]]:
    category_lookup = {t: cat for cat, vals in CUSTOM_VOCAB.items() for t in vals}

    # --- Phase 1: filter & remap WD tags ------------------------------------
    filtered: List[Tuple[str, float]] = []
    anime_bucket: List[Tuple[str, float]] = []
    for raw_tag, score in wd_scores.items():
        tag = WD_TAG_REMAP.get(raw_tag, raw_tag)  # canonical remap
        if tag in REMOVE_TAGS or tag in PROHIBITED_TAGS:
            continue
        if tag in HAIR_COLOR_SUPPRESS:
            continue  # suppress hallucinated hair colours
        if tag in EYE_COLOR_SUPPRESS:
            continue  # suppress hallucinated eye colours
        if score < TAG_THRESHOLD:
            continue
        if any(tag.startswith(pref) for pref in DOWNWEIGHT_PREFIXES):
            score *= ANIME_PENALTY
            if score < ANIME_MIN_SCORE:
                continue
            anime_bucket.append((tag, score))
        else:
            filtered.append((tag, score))

    filtered.sort(key=lambda x: x[1], reverse=True)
    anime_bucket.sort(key=lambda x: x[1], reverse=True)
    combined = filtered + anime_bucket[:ANIME_MAX]
    top = combined[:TOP_K]

    # --- Phase 2: accumulate tags by priority --------------------------------
    tags: List[str] = []
    scores: Dict[str, float] = {}
    category_counts: Dict[str, int] = {k: 0 for k in CATEGORY_CAPS.keys()}
    sources: Dict[str, set] = {}

    def add_tag(tag: str, score: float = 0.6, source: str = "unknown") -> None:
        norm = _normalize_tag(tag)
        if not norm:
            return
        norm = WD_TAG_REMAP.get(norm, norm)  # remap again after normalize
        if norm in PROHIBITED_TAGS or norm in HAIR_COLOR_SUPPRESS or norm in EYE_COLOR_SUPPRESS:
            return
        cat = category_lookup.get(norm, "other")
        cap = CATEGORY_CAPS.get(cat, CATEGORY_CAPS["other"])

        if norm in scores:
            scores[norm] = max(scores[norm], score)
            sources.setdefault(norm, set()).add(source)
            return

        if category_counts.get(cat, 0) >= cap:
            return

        category_counts[cat] = category_counts.get(cat, 0) + 1
        scores[norm] = float(score)
        tags.append(norm)
        sources.setdefault(norm, set()).add(source)

    wd_label_set = set(wd_scores.keys())

    # Only force identity when evidence exists unless explicitly overridden
    def has_forced_evidence() -> bool:
        for tag in florence_tags:
            if any(substr in tag for substr in FORCED_EVIDENCE_SUBSTRINGS):
                return True
        for tag in wd_label_set:
            if any(substr in tag for substr in FORCED_EVIDENCE_SUBSTRINGS):
                return True
        return False

    # --- Identity tags: always injected (she's always our elf) ---
    if ALWAYS_FORCE_FORCED_TAGS or has_forced_evidence():
        for t in IDENTITY_TAGS:
            add_tag(t, 0.99, source="identity")

    # --- Style conformance tags ---
    if is_on_target:
        # On-target: inject the desired style tags as positive signal
        add_tag("on_target", 0.99, source="conformance")
        for t in TARGET_STYLE_TAGS:
            add_tag(t, 0.98, source="conformance")
    else:
        # Off-target: inject deviation markers instead
        add_tag("off_target", 0.99, source="conformance")
        for dt in (deviation_tags or []):
            add_tag(dt, 0.95, source="deviation")

    for tag, score in top:
        add_tag(tag, score, source="wd")

    # Slot tags from caption parsing (high priority — grounded in observed text)
    for st in slot_tags:
        add_tag(st, 0.72, source="slot")

    florence_set = set()
    florence_only_added = 0
    for raw_ft in florence_tags:
        ft = WD_TAG_REMAP.get(raw_ft, raw_ft)
        if not ft or ft in FLORENCE_STOPWORDS or len(ft) <= 2:
            continue
        florence_set.add(ft)
        overlap = ft in scores or ft in wd_label_set
        allowed_alone = ft in FLORENCE_ALLOWED_ALONE
        if not overlap:
            if not allowed_alone:
                continue
            if florence_only_added >= FLORENCE_ONLY_MAX:
                continue
            florence_only_added += 1
        add_tag(ft, 0.64, source="florence_overlap" if overlap else "florence_only")

    # Softly reinforce custom vocab when both WD and Florence agree
    for cat, vocab in CUSTOM_VOCAB.items():
        for t in vocab:
            if t in scores and t in florence_set:
                add_tag(t, scores.get(t, 0.58), source="custom_reinforced")

    # --- Phase 3: hair colour dedup (prefer canon red-brown) -----------------
    hair_candidates = [t for t in tags if t in HAIR_COLOR_TAGS]

    def _hair_score(tag: str) -> float:
        base = scores.get(tag, 0.0)
        base += HAIR_COLOR_BIAS.get(tag, 0.0)  # bias toward canon colours
        if tag in florence_set:
            base += 0.08
        if tag in wd_label_set:
            base += 0.02
        return base

    if len(hair_candidates) > 1:
        winner = max(hair_candidates, key=lambda t: (_hair_score(t), scores.get(t, 0.0)))
        for hc in hair_candidates:
            if hc == winner:
                continue
            try:
                tags.remove(hc)
            except ValueError:
                pass
            scores.pop(hc, None)
            sources.pop(hc, None)

    tags = tags[:TOP_K]

    # --- Phase 4a: on-target trait omission ----------------------------------
    # For ON-TARGET images, omit default character traits.  The model learns:
    # "when these tags are absent + on_target is present, use the defaults."
    # For OFF-TARGET images, keep all traits explicit so the model sees them.
    if is_on_target:
        for dt in _DEFAULT_TRAITS:
            if dt in tags:
                tags.remove(dt)
                # keep in scores for auditing but mark as omitted
                sources.setdefault(dt, set()).add("omitted_default")

    # --- Phase 4b: conflict resolution ---------------------------------------
    # Zoom level: keep the most specific, drop contradictions
    _zoom_priority = ["close_up", "upper_body", "cowboy_shot", "full_body", "wide_shot"]
    zoom_present = [z for z in _zoom_priority if z in tags]
    if len(zoom_present) > 1:
        # Keep the one with highest WD confidence; drop the rest
        keeper = max(zoom_present, key=lambda t: scores.get(t, 0.0))
        for zp in zoom_present:
            if zp != keeper and zp in tags:
                tags.remove(zp)
                scores.pop(zp, None)

    # 'breasts' tag on clothed images: suppress unless nudity indicator present
    nudity_indicators = {"nude", "topless", "nipples", "breasts_visible",
                         "partial_nudity", "bare_shoulders"}
    if "breasts" in tags and not (nudity_indicators & set(tags)):
        tags.remove("breasts")
        scores.pop("breasts", None)

    # --- Phase 5: build attributes -------------------------------------------
    attributes: Dict[str, str] = {}
    if rating_tags:
        top_rating = max(rating_tags.items(), key=lambda x: x[1])
        attributes["rating"] = top_rating[0].replace("rating:", "")
        attributes["rating_score"] = f"{top_rating[1]:.4f}"

    # Expose tag sources for auditing
    attributes["tag_sources"] = {k: sorted(list(v)) for k, v in sources.items()}

    return tags, scores, attributes


def _cache_paths(image_path: Path) -> Tuple[Path, str]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = _hash_file(image_path)
    cache_file = CACHE_DIR / f"{digest}.json"
    return cache_file, digest


def tag_image(
    path: Path,
    model_name: str = WD_MODEL_ID,
    use_cache: bool = True,
    use_florence: bool = True,
    image_obj: Optional[Image.Image] = None,
) -> TagResult:
    cache_file, digest = _cache_paths(path)
    if use_cache and cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            result = TagResult(tags=data["tags"], attributes=data.get("attributes", {}), scores=data.get("scores", {}))
            if image_obj is not None:
                try:
                    image_obj.close()
                except Exception:
                    pass
            return result
        except Exception:
            cache_file.unlink(missing_ok=True)

    main_image: Optional[Image.Image] = None
    aspect_tag = "square_aspect"
    try:
        if image_obj is not None:
            main_image = image_obj if image_obj.mode == "RGB" else image_obj.convert("RGB")
        else:
            with Image.open(path) as raw:
                main_image = raw.convert("RGB")

        if main_image is None:
            raise RuntimeError(f"Unable to load image {path}")

        w, h = main_image.size
        ratio = w / h if h else 1.0
        if ratio > 1.25:
            aspect_tag = "landscape_aspect"
        elif ratio < 0.8:
            aspect_tag = "portrait_aspect"
    except Exception as e:  # pragma: no cover - defensive path
        print(f"Error opening image {path}: {e}")
        return TagResult([], {}, {})

    try:
        processor, model, labels = _load_wd_model(model_name)
        wd_scores = _infer_wd(path, processor, model, labels, image_obj=main_image)

        rating_tags = {k: v for k, v in wd_scores.items() if k.startswith("rating:")}
        content_tags = {k: v for k, v in wd_scores.items() if not k.startswith("rating:")}

        caption_tags: List[str] = []
        slot_tags: List[str] = [aspect_tag]
        raw_caption = ""
        caption_source: Optional[str] = None
        florence_error: Optional[str] = None
        llava_error: Optional[str] = None

        # --- Captioning (Florence primary, LLaVA fallback) ---
        if use_florence:
            try:
                florence_processor, florence_model = _load_florence()
                caption_tags, raw_caption = _caption_with_florence(
                    path, florence_processor, florence_model, image_obj=main_image,
                )
                caption_source = "florence"
            except Exception as exc:  # noqa: BLE001
                florence_error = f"florence_failed: {type(exc).__name__}: {exc}"

        if not raw_caption:
            try:
                llava_processor, llava_model = _load_llava()
                caption_tags, raw_caption = _caption_with_llava(
                    path, llava_processor, llava_model, image_obj=main_image,
                )
                caption_source = "llava"
            except Exception as exc:  # noqa: BLE001
                llava_error = f"llava_failed: {type(exc).__name__}: {exc}"

        # --- Slot extraction from raw caption ---
        if raw_caption:
            slot_tags.extend(_extract_slots_from_caption(raw_caption))

        # --- Style conformance scoring ---
        conf_score, is_on_target, deviation_tags = _score_conformance(
            content_tags, caption_tags, raw_caption,
        )

        # --- Build training caption (style-aware) ---
        processed_caption = _build_caption(raw_caption, is_on_target, deviation_tags) if raw_caption else ""

        # --- Merge all tag sources (conformance-aware) ---
        tags, scores, attributes = _merge_tags(
            content_tags, caption_tags, slot_tags, rating_tags,
            is_on_target=is_on_target,
            deviation_tags=deviation_tags,
        )

        # Store both raw and processed captions + conformance info
        if raw_caption:
            attributes["raw_caption"] = raw_caption
            attributes["caption"] = processed_caption
            if caption_source:
                attributes[f"{caption_source}_caption"] = raw_caption
                attributes["caption_source"] = caption_source
        if caption_tags and caption_source:
            attributes[f"{caption_source}_raw_tags"] = caption_tags

        # Conformance audit trail
        attributes["conformance_score"] = f"{conf_score:.2f}"
        attributes["style_tier"] = "on_target" if is_on_target else "off_target"
        if deviation_tags:
            attributes["deviation_tags"] = deviation_tags

        scores.update({t: scores.get(t, 0.64) for t in caption_tags if t in tags})

        # Rating
        if rating_tags:
            top_rating = max(rating_tags.items(), key=lambda x: x[1])
            attributes["rating"] = top_rating[0].replace("rating:", "")
            attributes["rating_score"] = f"{top_rating[1]:.4f}"

        # Audit trail
        if florence_error:
            attributes["florence_status"] = florence_error
        if llava_error:
            attributes["llava_status"] = llava_error
        attributes["aspect_tag"] = aspect_tag

        tag_result = TagResult(tags=tags, attributes=attributes, scores=scores)

        if use_cache:
            cache_file.write_text(
                json.dumps({"tags": tags, "attributes": attributes, "scores": scores}, ensure_ascii=False),
                encoding="utf-8",
            )

        return tag_result
    finally:
        cache_file.touch(exist_ok=True)
        if main_image is not None:
            try:
                main_image.close()
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()