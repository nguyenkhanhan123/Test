"""Pydantic v2 schema connecting every stage of the AI film pipeline.

Defines the canonical contract for ``script.json`` (Step 1 -> Step 6),
per-shot word alignment tokens, and the final build metrics report.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ShotType(str, Enum):
    """Supported visual synthesis and camera motion modes for a shot."""

    VIDEO_AI = "video_ai"
    KEN_BURNS_ZOOM_IN = "ken_burns_zoom_in"
    KEN_BURNS_ZOOM_OUT = "ken_burns_zoom_out"
    KEN_BURNS_PAN_LEFT = "ken_burns_pan_left"
    KEN_BURNS_PAN_RIGHT = "ken_burns_pan_right"
    KEN_BURNS_TILT_UP = "ken_burns_tilt_up"

    @property
    def is_video_ai(self) -> bool:
        """Return True when this shot requires AI video generation."""
        return self == ShotType.VIDEO_AI


class EmotionTag(str, Enum):
    """Scene emotion tag used to select expressive voice references & SFX."""

    NEUTRAL = "neutral"
    CALM = "calm"
    TENSE = "tense"
    ANGRY_ACTION = "angry_action"
    WHISPER = "whisper"
    SAD = "sad"
    EXCITED = "excited"


class VoiceEmotionSample(BaseModel):
    """Reference audio + exact transcript for one emotion state of a speaker."""

    ref_audio: str = Field(..., description="Path to clean 5-10s WAV sample of user voice")
    ref_text: str = Field(..., min_length=1, description="Exact Vietnamese transcript of ref_audio")
    speed: float = Field(default=1.0, ge=0.7, le=1.35)


class VoiceSpec(BaseModel):
    """Voice configuration for a character or narrator."""

    engine: Literal["edge_tts", "vi_f5_tts", "qwen3_tts", "mms_tts_vie"] = "edge_tts"
    voice_id: str = Field(
        ...,
        description="Edge TTS voice short name or default reference audio path",
    )
    rate: str = Field(default="+0%", description="Speech rate offset for Edge TTS")
    pitch: str = Field(default="+0Hz", description="Pitch offset for Edge TTS")
    volume: str = Field(default="+0%", description="Volume offset for Edge TTS")
    emotion_bank: dict[EmotionTag, VoiceEmotionSample] = Field(
        default_factory=dict,
        description="Per-emotion reference samples for expressive Vi-F5-TTS / Qwen3-TTS cloning",
    )


class WordToken(BaseModel):
    """Single word timing token in local shot time (seconds)."""

    word: str
    start: float = Field(..., ge=0.0)
    end: float = Field(..., ge=0.0)

    @field_validator("end")
    @classmethod
    def _ensure_end_ge_start(cls, value: float, info) -> float:
        start = info.data.get("start", 0.0)
        return max(value, start)


class Shot(BaseModel):
    """Single atomic shot in the Audio-First film pipeline."""

    shot_id: str = Field(..., pattern=r"^shot_\d{3,}$")
    character: str
    character_lora: Optional[str] = None
    voice_ref: str
    voice_spec: Optional[VoiceSpec] = None
    shot_type: ShotType
    emotion: EmotionTag = Field(
        default=EmotionTag.NEUTRAL,
        description="Emotional delivery for voice cloning & scene pacing",
    )
    dialogue: str = Field(..., min_length=1)
    image_prompt: str = Field(..., min_length=10)
    motion_prompt: Optional[str] = None
    end_keyframe_path: Optional[str] = Field(
        default=None,
        description="Optional end keyframe for Wan 2.2 First-Last Frame (FLF2V) action shots",
    )
    driving_video_path: Optional[str] = Field(
        default=None,
        description="Optional driving stunt/motion video for Wan 2.2 Animate pose transfer",
    )
    sfx_cue: Optional[Literal["none", "wind_ominous", "meteor_impact", "compass_chime", "portal_hum"]] = "none"
    lead_in_sec: float = Field(default=0.30, ge=0.0, le=3.0)
    tail_pad_sec: float = Field(default=0.45, ge=0.0, le=5.0)
    audio_path: Optional[str] = None
    keyframe_path: Optional[str] = None
    video_clip_path: Optional[str] = None
    speech_duration_sec: Optional[float] = Field(default=None, gt=0.0)
    duration_sec: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Quantized shot clip duration in seconds (24fps grid)",
    )
    render_mode_used: Optional[str] = None


class EpisodeScript(BaseModel):
    """Top-level manifest shared across Steps 1 to 6."""

    episode_id: str
    title: str
    fps: int = Field(default=24, ge=12, le=60)
    width: int = Field(default=1280, ge=360)
    height: int = Field(default=720, ge=360)
    style_lora: Optional[str] = None
    style_prompt_prefix: str = (
        "cinematic dark anime film still, Makoto Shinkai and Ufotable lighting, "
        "detailed cel-shaded painterly background, dramatic atmospheric fog, 16:9 widescreen"
    )
    characters: dict[str, VoiceSpec] = Field(default_factory=dict)
    shots: list[Shot] = Field(..., min_length=1)

    @field_validator("shots")
    @classmethod
    def _ensure_unique_shot_ids(cls, shots: list[Shot]) -> list[Shot]:
        ids = [s.shot_id for s in shots]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate shot_id detected in EpisodeScript.shots")
        return shots

    def dynamic_ratio(self) -> tuple[int, int, float]:
        """Return (video_ai_count, ken_burns_count, video_ai_fraction)."""
        ai_count = sum(1 for s in self.shots if s.shot_type.is_video_ai)
        kb_count = len(self.shots) - ai_count
        return ai_count, kb_count, (ai_count / len(self.shots) if self.shots else 0.0)
