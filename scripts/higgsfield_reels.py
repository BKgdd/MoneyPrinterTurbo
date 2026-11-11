#!/usr/bin/env python3
"""Generate consistent AI-influencer short videos (9:16) through the Higgsfield API.

Flow, all against https://api.higgsfield.ai:

1. once   : create a reusable character — an AI Influencer character sheet, then
            train a Soul ID reference from that image.
2. per    : SOUL V2 text-to-image (9:16, 1080p) using the character reference.
   concept
3. per    : image-to-video (PixVerse V6 or Seedance 2.0); vertical framing comes
   concept  from the still generated in step 2.
4. per    : download the MP4 and write a publish plan with hooks, captions and
   concept  hashtags.

Credentials come from HF_API_KEY_ID + HF_API_KEY_SECRET, or from a single
HF_API_KEY in "<key-id>:<secret>" form. Never pass them on the command line.
API docs: https://docs.higgsfield.ai/docs
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://api.higgsfield.ai"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "scripts" / "higgsfield_reels_plan.json"
DEFAULT_OUT = ROOT / "output" / "higgsfield"

SOUL_IMAGE_ENDPOINT = "/higgsfield-ai/soul/v2/standard"
INFLUENCER_ENDPOINT = "/higgsfield/ai-influencer"
VIDEO_ENDPOINTS = {
    "pixverse": "/pixverse/v6/image-to-video",
    "seedance": "/bytedance/seedance-2.0/image-to-video",
}
CHARACTER_LIST_PATH = "/v1/custom-references/list"
# When an existing Soul ID supplies the identity, the prompt must not describe a
# different face; it only carries scene, wardrobe and mood.
NEUTRAL_LOOK = (
    "the same person, keeping their exact face, hair and identity from the "
    "character reference"
)
UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

ACTIVE_STATUSES = frozenset({"queued", "in_progress", "pending", "processing"})
FAILURE_STATUSES = frozenset({"failed", "nsfw", "canceled", "cancelled", "expired"})

# Published Higgsfield pricing, for the pre-flight estimate only. Video cost
# depends on model and duration; see https://docs.higgsfield.ai/pricing.
SOUL_ID_CREDITS = 40  # ~$2.50, one-off per character
SOUL_IMAGE_CREDITS = 3  # 1080p, batch of 1
INFLUENCER_SHEET_CREDITS = 1  # ~$0.05
CREDITS_PER_USD = 16

MAX_SUBMIT_RETRIES = 3
MAX_POLL_RETRIES = 5


class HiggsfieldError(RuntimeError):
    """Deterministic request, configuration or response error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_plan(path: Path) -> dict[str, Any]:
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise HiggsfieldError(f"Cannot read plan {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise HiggsfieldError(f"Plan {path} is not valid JSON: {exc}") from exc
    if not isinstance(plan, dict) or not isinstance(plan.get("videos"), list):
        raise HiggsfieldError(f"Plan {path} must contain a 'videos' list")
    if not plan["videos"]:
        raise HiggsfieldError(f"Plan {path} has no videos")
    for index, video in enumerate(plan["videos"]):
        for field in ("slug", "image_prompt", "motion_prompt"):
            if not str(video.get(field) or "").strip():
                raise HiggsfieldError(f"Plan video #{index + 1} is missing '{field}'")
    return plan


KEY_ID_ENV_VARS = ("HF_API_KEY_ID", "HIGGSFIELD_API_KEY_ID")
SECRET_ENV_VARS = ("HF_API_KEY_SECRET", "HIGGSFIELD_API_KEY_SECRET")
# The official SDK reads HF_CREDENTIALS as "<key-id>:<secret>".
COMBINED_ENV_VARS = ("HF_API_KEY", "HF_CREDENTIALS", "HIGGSFIELD_API_KEY")


def _first_env(names: tuple[str, ...]) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def credentials() -> tuple[str, str]:
    key_id = _first_env(KEY_ID_ENV_VARS)
    secret = _first_env(SECRET_ENV_VARS)
    if not (key_id and secret):
        for name in COMBINED_ENV_VARS:
            combined = os.getenv(name, "").strip()
            if ":" not in combined:
                continue
            part_id, part_secret = (part.strip() for part in combined.split(":", 1))
            if part_id and part_secret:
                key_id, secret = part_id, part_secret
                break
    if not (key_id and secret):
        raise HiggsfieldError(
            "Missing Higgsfield credentials. Create an API key at "
            "https://console.higgsfield.ai and set HF_API_KEY_ID + "
            "HF_API_KEY_SECRET in this project's Secrets (aliases accepted: "
            "HIGGSFIELD_API_KEY_ID/HIGGSFIELD_API_KEY_SECRET, or "
            "HF_API_KEY/HF_CREDENTIALS as \"<key-id>:<secret>\")."
        )
    return key_id, secret


def _auth_headers(key_id: str, secret: str) -> dict[str, str]:
    return {
        "Authorization": f"Key {key_id}:{secret}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _error_text(response: requests.Response, key_id: str, secret: str) -> str:
    try:
        payload = response.json()
    except ValueError:
        return f"HTTP {response.status_code}"
    detail: Any = payload
    if isinstance(payload, dict):
        error = payload.get("error")
        detail = payload.get("detail") or payload.get("message")
        if not detail and isinstance(error, dict):
            detail = error.get("message") or error.get("detail") or error.get("code")
        detail = detail or error
    text = str(detail or f"HTTP {response.status_code}")
    for value in (secret, key_id):
        if value:
            text = text.replace(value, "***")
    return text[:500]


def _post(path: str, payload: dict[str, Any], key_id: str, secret: str) -> dict[str, Any]:
    """Submit a billable request; retries reuse one Idempotency-Key."""
    url = path if path.startswith("http") else f"{BASE_URL}{path}"
    headers = _auth_headers(key_id, secret)
    headers["Idempotency-Key"] = str(uuid.uuid4())
    last_error = "unknown error"
    for attempt in range(1, MAX_SUBMIT_RETRIES + 1):
        try:
            response = requests.post(
                url, json=payload, headers=headers, timeout=(15, 180)
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code == 429 or response.status_code >= 500:
                last_error = _error_text(response, key_id, secret)
            elif not response.ok:
                raise HiggsfieldError(
                    f"POST {path} rejected: HTTP {response.status_code}: "
                    f"{_error_text(response, key_id, secret)}"
                )
            else:
                try:
                    body = response.json()
                except ValueError as exc:
                    raise HiggsfieldError(
                        f"POST {path} returned an unreadable body"
                    ) from exc
                if not isinstance(body, dict):
                    raise HiggsfieldError(f"POST {path} returned a non-object body")
                return body
        if attempt < MAX_SUBMIT_RETRIES:
            time.sleep(2.0 * attempt)
    raise HiggsfieldError(
        f"POST {path} failed after {MAX_SUBMIT_RETRIES} attempts: {last_error}"
    )


def _get(path: str, key_id: str, secret: str) -> dict[str, Any]:
    url = path if path.startswith("http") else f"{BASE_URL}{path}"
    last_error = "unknown error"
    for attempt in range(1, MAX_POLL_RETRIES + 1):
        try:
            response = requests.get(
                url, headers=_auth_headers(key_id, secret), timeout=(15, 60)
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code == 429 or response.status_code >= 500:
                last_error = _error_text(response, key_id, secret)
            elif not response.ok:
                raise HiggsfieldError(
                    f"GET {path} failed: HTTP {response.status_code}: "
                    f"{_error_text(response, key_id, secret)}"
                )
            else:
                try:
                    body = response.json()
                except ValueError as exc:
                    raise HiggsfieldError(f"GET {path} returned an unreadable body") from exc
                if not isinstance(body, dict):
                    raise HiggsfieldError(f"GET {path} returned a non-object body")
                return body
        if attempt < MAX_POLL_RETRIES:
            time.sleep(min(2.0 * attempt, 10.0))
    raise HiggsfieldError(f"GET {path} failed after {MAX_POLL_RETRIES} attempts: {last_error}")


def _wait(
    url: str, key_id: str, secret: str, *, timeout: float, label: str
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    interval = 3.0
    while True:
        body = _get(url, key_id, secret)
        status = str(body.get("status") or "").strip().lower()
        if status == "completed":
            return body
        if status in FAILURE_STATUSES:
            reason = body.get("fail_reason") or body.get("error") or status
            raise HiggsfieldError(f"{label} ended as {status}: {reason}")
        if status and status not in ACTIVE_STATUSES and status != "not_ready":
            raise HiggsfieldError(f"{label} returned an unknown status: {status!r}")
        if time.monotonic() >= deadline:
            raise HiggsfieldError(
                f"{label} still {status or 'running'} after {timeout:.0f}s; "
                f"check the request manually: {url}"
            )
        time.sleep(interval)
        interval = min(interval * 1.2, 10.0)


def _handle(body: dict[str, Any], path: str) -> tuple[str, str]:
    request_id = str(body.get("request_id") or "").strip()
    status_url = str(body.get("status_url") or "").strip()
    if not request_id or not status_url:
        raise HiggsfieldError(f"POST {path} did not return request_id/status_url")
    return request_id, status_url


def fetch_characters(key_id: str, secret: str) -> list[dict[str, Any]]:
    """List the Soul ID references owned by the account (paginated)."""
    characters: list[dict[str, Any]] = []
    page = 1
    while page <= 10:
        body = _get(
            f"{CHARACTER_LIST_PATH}?page={page}&page_size=50", key_id, secret
        )
        items = body.get("items")
        if not isinstance(items, list):
            break
        characters.extend(item for item in items if isinstance(item, dict))
        total_pages = body.get("total_pages")
        if not items or not isinstance(total_pages, int) or page >= total_pages:
            break
        page += 1
    return characters


def _format_characters(characters: list[dict[str, Any]]) -> str:
    if not characters:
        return "none"
    return "; ".join(
        f"{item.get('name') or 'unnamed'} ({item.get('id')}) "
        f"[{item.get('status') or '?'}]"
        for item in characters
    )[:900]


def print_characters(key_id: str, secret: str) -> int:
    characters = fetch_characters(key_id, secret)
    if not characters:
        print("No Soul ID found in this account.")
        return 1
    print(f"{len(characters)} Soul ID(s) in this account:")
    for item in characters:
        print(
            f"  {str(item.get('name') or 'unnamed'):<24} "
            f"{item.get('id')}  {item.get('status') or '?'}  "
            f"{item.get('model_version') or '?'}"
        )
    return 0


def _read_saved_character(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("id") or "").strip()


def _save_character(
    path: Path, character_id: str, persona: dict[str, Any], sheet_url: str, source: str
) -> None:
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, dict):
            existing = loaded
    payload = {
        **existing,
        "id": character_id,
        "name": persona.get("name"),
        "model_version": persona.get("model_version") or "v2",
        "sheet_url": sheet_url or existing.get("sheet_url") or "",
        "source": source,
        "updated_at": _utc_now(),
    }
    payload.setdefault("created_at", _utc_now())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_character(
    *, explicit: str, saved: str, key_id: str, secret: str
) -> tuple[str, str]:
    """Return (character_id, how it was resolved) for an existing Soul ID."""
    explicit = explicit.strip()
    if explicit and UUID_RE.fullmatch(explicit):
        return explicit, "id"
    characters = fetch_characters(key_id, secret)
    if explicit:
        wanted = explicit.casefold()
        matches = [
            item
            for item in characters
            if str(item.get("name") or "").strip().casefold() == wanted
        ]
        if len(matches) == 1:
            return str(matches[0]["id"]), "name"
        if not matches:
            raise HiggsfieldError(
                f"No Soul ID named {explicit!r}. Available: {_format_characters(characters)}"
            )
        raise HiggsfieldError(
            f"{explicit!r} matches {len(matches)} Soul IDs; pass the id instead. "
            f"Available: {_format_characters(matches)}"
        )
    if saved:
        return saved, "saved"
    completed = [
        item
        for item in characters
        if str(item.get("status") or "").strip().lower() == "completed"
    ]
    if len(completed) == 1:
        return str(completed[0]["id"]), "only-completed"
    if not completed:
        raise HiggsfieldError(
            "This account has no completed Soul ID yet. Train one with "
            "--reference-url <public image url> (or --create-character). "
            f"Found: {_format_characters(characters)}"
        )
    raise HiggsfieldError(
        f"This account has {len(completed)} completed Soul IDs; choose one with "
        f"--character <id|name>. Available: {_format_characters(completed)}"
    )


def create_character_sheet(
    persona: dict[str, Any], key_id: str, secret: str, timeout: float
) -> str:
    payload: dict[str, Any] = {
        "tier": "normal",
        "brief": str(persona.get("brief") or "").strip(),
        "seed": int(persona.get("seed") or 42817),
    }
    _, status_url = _handle(
        _post(INFLUENCER_ENDPOINT, payload, key_id, secret), INFLUENCER_ENDPOINT
    )
    body = _wait(status_url, key_id, secret, timeout=timeout, label="AI Influencer sheet")
    images = body.get("images") or []
    if not images or not isinstance(images[0], dict) or not images[0].get("url"):
        raise HiggsfieldError("AI Influencer sheet completed without an image URL")
    return str(images[0]["url"])


def train_character(
    persona: dict[str, Any],
    image_urls: list[str],
    key_id: str,
    secret: str,
    timeout: float,
) -> tuple[str, str]:
    payload = {
        "name": str(persona.get("name") or "Higgsfield influencer")[:100],
        "model_version": str(persona.get("model_version") or "v2"),
        "input_images": [{"type": "image_url", "image_url": url} for url in image_urls],
    }
    created = _post("/v1/custom-references", payload, key_id, secret)
    reference_id = str(created.get("id") or "").strip()
    if not reference_id:
        raise HiggsfieldError("Soul ID creation did not return a reference id")
    status = str(created.get("status") or "").strip().lower()
    if status != "completed":
        _wait(
            f"{BASE_URL}/v1/custom-references/{reference_id}",
            key_id,
            secret,
            timeout=timeout,
            label="Soul ID training",
        )
    return reference_id, image_urls[0]


def render_image(
    prompt: str,
    character_id: str,
    resolution: str,
    key_id: str,
    secret: str,
    timeout: float,
) -> tuple[str, str]:
    payload = {
        "prompt": prompt,
        "aspect_ratio": "9:16",
        "resolution": resolution,
        "batch_size": 1,
        "custom_reference_id": character_id,
        "custom_reference_strength": 1.0,
        "enhance_prompt": False,
    }
    request_id, status_url = _handle(
        _post(SOUL_IMAGE_ENDPOINT, payload, key_id, secret), SOUL_IMAGE_ENDPOINT
    )
    body = _wait(status_url, key_id, secret, timeout=timeout, label="SOUL V2 image")
    images = body.get("images") or []
    if not images or not isinstance(images[0], dict) or not images[0].get("url"):
        raise HiggsfieldError("SOUL V2 image completed without an image URL")
    return str(images[0]["url"]), request_id


def render_video(
    image_url: str,
    prompt: str,
    *,
    model: str,
    duration: int,
    resolution: str,
    audio: bool,
    key_id: str,
    secret: str,
    timeout: float,
) -> tuple[str, str]:
    path = VIDEO_ENDPOINTS[model]
    payload: dict[str, Any] = {
        "image_url": image_url,
        "prompt": prompt,
        "duration": duration,
        "resolution": resolution,
        "generate_audio": audio,
    }
    if model == "pixverse":
        payload["negative_prompt"] = (
            "text, subtitles, watermark, extra fingers, deformed hands, "
            "plastic skin, doll-like face"
        )
    request_id, status_url = _handle(_post(path, payload, key_id, secret), path)
    body = _wait(status_url, key_id, secret, timeout=timeout, label=f"{model} video")
    video = body.get("video")
    if not isinstance(video, dict) or not video.get("url"):
        raise HiggsfieldError(f"{model} video completed without a video URL")
    return str(video["url"]), request_id


def download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=(20, 300)) as response:
        response.raise_for_status()
        temporary = destination.with_suffix(destination.suffix + ".part")
        size = 0
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
                    size += len(chunk)
    if size == 0:
        temporary.unlink(missing_ok=True)
        raise HiggsfieldError(f"Downloaded an empty file from {url}")
    temporary.replace(destination)
    return destination


def build_prompts(video: dict[str, Any], look: str) -> tuple[str, str]:
    image_prompt = str(video["image_prompt"]).replace("{look}", look).strip()
    motion_prompt = str(video["motion_prompt"]).strip()
    return image_prompt, motion_prompt


def _selection(plan: dict[str, Any], only: list[str], limit: int | None) -> list[dict[str, Any]]:
    videos = plan["videos"]
    if only:
        wanted = {slug.strip() for slug in only if slug.strip()}
        known = {str(video["slug"]) for video in videos}
        unknown = wanted - known
        if unknown:
            raise HiggsfieldError(f"Unknown slug(s): {', '.join(sorted(unknown))}")
        videos = [video for video in videos if str(video["slug"]) in wanted]
    return videos[:limit] if limit else videos


def estimate(videos: list[dict[str, Any]], needs_character: bool) -> str:
    images = len(videos)
    credits = images * SOUL_IMAGE_CREDITS
    lines = [
        f"  1 reference image per concept ({images} x {SOUL_IMAGE_CREDITS} credits) "
        f"= {images * SOUL_IMAGE_CREDITS} credits (~${images * SOUL_IMAGE_CREDITS / CREDITS_PER_USD:.2f})",
    ]
    if needs_character:
        credits += SOUL_ID_CREDITS + INFLUENCER_SHEET_CREDITS
        lines.insert(
            0,
            f"  1 character build (sheet {INFLUENCER_SHEET_CREDITS} + Soul ID "
            f"{SOUL_ID_CREDITS} credits) = {SOUL_ID_CREDITS + INFLUENCER_SHEET_CREDITS} "
            f"credits (~${(SOUL_ID_CREDITS + INFLUENCER_SHEET_CREDITS) / CREDITS_PER_USD:.2f})",
        )
    lines.append(
        f"  {images} video clip(s): priced per model and duration, see "
        "https://docs.higgsfield.ai/pricing"
    )
    lines.append(f"  Known credits for this run: {credits} (~${credits / CREDITS_PER_USD:.2f})")
    return "\n".join(lines)


def write_publish_plan(out_dir: Path, records: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "publish_plan.json").write_text(
        json.dumps({"generated_at": _utc_now(), "videos": records}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    blocks = ["# Plano de publicacao\n"]
    for record in records:
        if record.get("status") != "ready":
            continue
        hashtags = " ".join(record.get("hashtags") or [])
        blocks.append(
            f"\n## {record['title']}\n\n"
            f"- Arquivo: `{record['file']}`\n"
            f"- Plataformas: {', '.join(record.get('platforms') or [])}\n"
            f"- Gancho: {record.get('hook') or '-'}\n"
            f"- Legenda: {record.get('caption') or '-'}\n"
            f"- Hashtags: {hashtags or '-'}\n"
        )
    (out_dir / "captions.md").write_text("\n".join(blocks).strip() + "\n", encoding="utf-8")


def run(plan: dict[str, Any], args: argparse.Namespace) -> int:
    videos = _selection(plan, args.only, args.limit)
    persona = plan.get("character") or {}
    defaults = plan.get("defaults") or {}
    look = str(persona.get("look") or "").strip()
    out_dir: Path = args.out
    character_file = out_dir / "character.json"
    model = args.video_model or str(defaults.get("video_model") or "pixverse")

    saved_character = _read_saved_character(character_file)
    needs_character = bool(args.reference_url) or args.create_character
    # The persona look only drives the character sheet. Whenever the identity
    # comes from a Soul ID (reused or trained on the user's photos) the prompt
    # must not describe a different face.
    identity_look = (
        look if (args.create_character and not args.reference_url) else NEUTRAL_LOOK
    )

    print(f"Plan: {len(videos)} video(s) from {args.plan}")
    for video in videos:
        duration = int(video.get("duration") or defaults.get("duration") or 8)
        print(f"  - {video['slug']}: {video.get('hook') or ''} ({duration}s)")
    print("\nEstimated spend before video generation:")
    print(estimate(videos, needs_character))

    if args.dry_run:
        image_prompt, motion_prompt = build_prompts(videos[0], identity_look)
        print("\nDry run: no API calls, no credits spent.")
        print(f"\nSample still prompt ({videos[0]['slug']}):\n{image_prompt}")
        print(f"\nSample motion prompt:\n{motion_prompt}")
        return 0

    key_id, secret = credentials()
    if args.list_characters:
        return print_characters(key_id, secret)

    resolution = args.resolution or str(defaults.get("resolution") or "1080p")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.character:
        character_id, origin = resolve_character(
            explicit=args.character, saved="", key_id=key_id, secret=secret
        )
    elif args.reference_url or args.create_character:
        character_id, origin = "", "train"
    else:
        character_id, origin = resolve_character(
            explicit="", saved=saved_character, key_id=key_id, secret=secret
        )

    if origin == "train":
        image_urls = list(args.reference_url or [])
        source = "reference-urls"
        if not image_urls:
            print("\nCreating the AI Influencer character sheet...")
            sheet_url = create_character_sheet(persona, key_id, secret, args.poll_timeout)
            print(f"Character sheet: {sheet_url}")
            image_urls = [sheet_url]
            source = "ai-influencer-sheet"
        print("Training the Soul ID character reference (this takes a few minutes)...")
        character_id, portrait_url = train_character(
            persona, image_urls, key_id, secret, args.poll_timeout
        )
        _save_character(character_file, character_id, persona, portrait_url, source)
        print(f"Soul ID character ready: {character_id}")
    else:
        _save_character(character_file, character_id, persona, "", origin)
        print(f"\nUsing the existing Soul ID {character_id} (resolved by {origin})")

    records: list[dict[str, Any]] = []
    failures = 0
    for index, video in enumerate(videos, start=1):
        slug = str(video["slug"])
        duration = int(video.get("duration") or defaults.get("duration") or 8)
        image_prompt, motion_prompt = build_prompts(video, identity_look)
        record: dict[str, Any] = {
            "slug": slug,
            "title": video.get("title") or slug,
            "hook": video.get("hook"),
            "caption": video.get("caption"),
            "hashtags": video.get("hashtags") or [],
            "platforms": video.get("platforms") or [],
            "duration": duration,
            "image_prompt": image_prompt,
            "motion_prompt": motion_prompt,
            "status": "pending",
        }
        print(f"\n[{index}/{len(videos)}] {slug}")
        try:
            print("  generating the 9:16 still...")
            still_url, image_request_id = render_image(
                image_prompt, character_id, resolution, key_id, secret, args.poll_timeout
            )
            record["image_request_id"] = image_request_id
            still_path = download(still_url, out_dir / slug / "still.jpg")
            record["still"] = str(still_path.relative_to(out_dir))
            print(f"  still saved: {still_path}")

            print(f"  animating with {model} ({duration}s)...")
            video_url, video_request_id = render_video(
                still_url,
                motion_prompt,
                model=model,
                duration=duration,
                resolution=resolution,
                audio=args.audio,
                key_id=key_id,
                secret=secret,
                timeout=args.poll_timeout,
            )
            record["video_request_id"] = video_request_id
            video_path = download(video_url, out_dir / f"{slug}.mp4")
            record["file"] = str(video_path.relative_to(out_dir))
            record["status"] = "ready"
            print(f"  ready: {video_path}")
        except HiggsfieldError as exc:
            failures += 1
            record["status"] = "failed"
            record["error"] = str(exc)
            print(f"  FAILED: {exc}", file=sys.stderr)
        records.append(record)
        write_publish_plan(out_dir, records)

    write_publish_plan(out_dir, records)
    ready = [record for record in records if record["status"] == "ready"]
    print(
        f"\nDone: {len(ready)} video(s) in {out_dir}; "
        f"captions in {out_dir / 'captions.md'}"
    )
    if failures:
        print(f"{failures} video(s) failed; see the log above.", file=sys.stderr)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate AI-influencer videos for TikTok / Shorts with Higgsfield."
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", action="append", default=[], metavar="SLUG")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--character",
        default="",
        help="Existing Soul ID: its id (UUID) or its exact name",
    )
    parser.add_argument(
        "--list-characters",
        action="store_true",
        help="List the Soul IDs in this account and exit",
    )
    parser.add_argument(
        "--create-character",
        action="store_true",
        help="Create a new AI Influencer character instead of reusing an existing Soul ID",
    )
    parser.add_argument(
        "--reference-url",
        action="append",
        default=[],
        help="Public image URL used to train a new Soul ID (repeatable)",
    )
    parser.add_argument("--video-model", choices=sorted(VIDEO_ENDPOINTS), default=None)
    parser.add_argument(
        "--duration", type=int, default=None, help="Override every concept's clip duration"
    )
    parser.add_argument("--resolution", choices=("720p", "1080p"), default=None)
    parser.add_argument("--audio", action="store_true", help="Ask the video model for audio")
    parser.add_argument("--poll-timeout", type=float, default=900.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the plan and estimate cost without calling the API",
    )
    args = parser.parse_args(argv)

    try:
        plan = load_plan(args.plan)
        if args.duration:
            for video in plan["videos"]:
                video["duration"] = args.duration
        return run(plan, args)
    except HiggsfieldError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
