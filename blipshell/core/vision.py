"""Image input support for vision-capable models.

Loads image files referenced in a user's message, downscales them (Pillow, with
a graceful fallback when it's absent), stores the downscaled bytes on disk, and
provides per-provider message-shape translation:

- Ollama wants: {"role": "user", "content": "...", "images": ["<base64>", ...]}
- OpenAI/OpenRouter wants: content as parts —
    [{"type": "text", "text": "..."},
     {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]

Messages carry a neutral `_image_refs` key (a list of ImageRef dicts) through the
agent/chat-loop layer; each client translates it to its own shape at send time.
The raw bytes live on disk under data/session_images/; only lightweight refs
travel in the message list (so replayed history doesn't bloat with base64).
"""

import base64
import hashlib
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    _HAVE_PIL = True
except ImportError:  # pragma: no cover - depends on environment
    _HAVE_PIL = False

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
MAX_DIMENSION = 1024          # downscale longest side to this (vision-token cost)
MAX_BYTES = 5 * 1024 * 1024   # soft cap; warn above this when we can't downscale
DEFAULT_STORE_DIR = Path("data/session_images")

# Tokens ending in an image extension: quoted (allows spaces) or bare (no spaces).
_PATH_RE = re.compile(
    r'"([^"]+?\.(?:png|jpe?g|gif|webp|bmp))"'      # "quoted path with spaces.png"
    r"|(\S+\.(?:png|jpe?g|gif|webp|bmp))",          # bare/no-space path.png
    re.IGNORECASE,
)

_MIME_BY_EXT = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
}


@dataclass
class ImageRef:
    """A reference to a stored (downscaled) image — not the bytes themselves."""
    path: str          # absolute path to the stored image on disk
    sha256: str
    mime: str
    orig_name: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ImageRef":
        return cls(
            path=d["path"], sha256=d["sha256"],
            mime=d["mime"], orig_name=d["orig_name"],
        )


def extract_image_paths(message: str) -> tuple[str, list[str]]:
    """Find existing image-file paths in a message, return (cleaned_text, paths).

    Detects quoted paths (which may contain spaces) and bare no-space paths that
    end in a known image extension AND exist on disk. Matched paths are stripped
    from the returned text. Non-existent or non-image tokens are left untouched.
    """
    if not message:
        return message, []

    found: list[str] = []
    spans: list[tuple[int, int]] = []
    for m in _PATH_RE.finditer(message):
        candidate = m.group(1) or m.group(2)
        try:
            if candidate and Path(candidate).is_file():
                found.append(candidate)
                spans.append(m.span())
        except OSError:
            continue  # malformed path — ignore

    if not spans:
        return message, []

    # Remove matched spans from the text (back to front to keep indices valid).
    cleaned = message
    for start, end in sorted(spans, reverse=True):
        cleaned = cleaned[:start] + cleaned[end:]
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned, found


def _mime_for(path: Path) -> str:
    return _MIME_BY_EXT.get(path.suffix.lower(), "image/png")


def load_image(path: str, store_dir: Path | None = None) -> ImageRef | None:
    """Load, downscale, and store an image. Returns an ImageRef, or None on failure.

    With Pillow: opens, downscales the longest side to MAX_DIMENSION, re-encodes,
    and stores the result. Without Pillow: stores the original bytes as-is (and
    warns if they're large). Storage path is data/session_images/<sha256><ext>,
    deduplicated by content hash.
    """
    src = Path(path)
    if not src.is_file():
        logger.warning("Image path does not exist: %s", path)
        return None

    store_dir = store_dir or DEFAULT_STORE_DIR
    try:
        raw = src.read_bytes()
    except OSError as e:
        logger.warning("Could not read image %s: %s", path, e)
        return None

    suffix = src.suffix.lower() if src.suffix.lower() in IMAGE_EXTENSIONS else ".png"
    mime = _mime_for(src)
    data = raw

    if _HAVE_PIL:
        try:
            from io import BytesIO
            img = Image.open(BytesIO(raw))
            fmt = img.format or "PNG"
            if max(img.size) > MAX_DIMENSION:
                img.thumbnail((MAX_DIMENSION, MAX_DIMENSION))
            # JPEG can't hold alpha — flatten if needed.
            if fmt in ("JPEG", "JPG") and img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            buf = BytesIO()
            img.save(buf, format=fmt)
            data = buf.getvalue()
        except Exception as e:
            logger.warning("Pillow could not process %s (%s) — sending as-is", path, e)
            data = raw
    elif len(raw) > MAX_BYTES:
        logger.warning(
            "Image %s is %.1fMB and Pillow is not installed to downscale it; "
            "sending as-is may be slow or rejected by the API.",
            path, len(raw) / 1024 / 1024,
        )

    sha = hashlib.sha256(data).hexdigest()
    try:
        store_dir.mkdir(parents=True, exist_ok=True)
        stored = store_dir / f"{sha}{suffix}"
        if not stored.exists():
            stored.write_bytes(data)
    except OSError as e:
        logger.warning("Could not store image %s: %s", path, e)
        return None

    return ImageRef(
        path=str(stored.resolve()), sha256=sha, mime=mime, orig_name=src.name,
    )


def encode_for_send(ref: ImageRef) -> str:
    """Read a stored image from disk and return base64 (no data-URI prefix)."""
    return base64.b64encode(Path(ref.path).read_bytes()).decode("ascii")


# ── Message-shape translation (called by the clients just before send) ────────

def has_image_refs(messages: list[dict]) -> bool:
    """True if any message carries the neutral _image_refs key."""
    return any(m.get("_image_refs") for m in messages)


def apply_images_ollama(messages: list[dict]) -> list[dict]:
    """Translate neutral _image_refs into Ollama's per-message `images` list."""
    if not has_image_refs(messages):
        return messages
    out = []
    for m in messages:
        refs = m.get("_image_refs")
        if not refs:
            out.append(m)
            continue
        new = {k: v for k, v in m.items() if k != "_image_refs"}
        new["images"] = [encode_for_send(ImageRef.from_dict(r)) for r in refs]
        out.append(new)
    return out


def apply_images_openai(messages: list[dict]) -> list[dict]:
    """Translate neutral _image_refs into OpenAI content-parts with image_url."""
    if not has_image_refs(messages):
        return messages
    out = []
    for m in messages:
        refs = m.get("_image_refs")
        if not refs:
            out.append(m)
            continue
        new = {k: v for k, v in m.items() if k != "_image_refs"}
        parts: list[dict] = [{"type": "text", "text": m.get("content", "") or ""}]
        for r in refs:
            ref = ImageRef.from_dict(r)
            b64 = encode_for_send(ref)
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{ref.mime};base64,{b64}"},
            })
        new["content"] = parts
        out.append(new)
    return out


def strip_image_refs(messages: list[dict]) -> list[dict]:
    """Drop image refs, replacing each with a text note (graceful degradation
    when no vision-capable endpoint is available)."""
    if not has_image_refs(messages):
        return messages
    out = []
    for m in messages:
        refs = m.get("_image_refs")
        if not refs:
            out.append(m)
            continue
        new = {k: v for k, v in m.items() if k != "_image_refs"}
        names = ", ".join(ImageRef.from_dict(r).orig_name for r in refs)
        note = f"[image: {names} — not re-sent; no vision endpoint available]"
        new["content"] = f"{m.get('content', '')}\n{note}".strip()
        out.append(new)
    return out
