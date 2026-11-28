"""Step 6a: Global-offset .ass (karaoke-style) & .srt subtitle builder via pysubs2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pysubs2

from media_utils import count_video_frames
from schema import EpisodeScript, WordToken

CHARACTER_DISPLAY_NAMES = {
    "narrator": "",
    "minh_khang": "Minh Khang",
    "linh_chi": "Linh Chi",
}

CHARACTER_STYLES = {
    "narrator": "Narrator",
    "minh_khang": "MinhKhang",
    "linh_chi": "LinhChi",
}


def _configure_styles(subs: pysubs2.SSAFile, width: int, height: int) -> None:
    subs.info["PlayResX"] = str(width)
    subs.info["PlayResY"] = str(height)
    subs.info["ScaledBorderAndShadow"] = "yes"
    subs.info["WrapStyle"] = "1"

    base_font = "Segoe UI"
    base_size = max(26, int(round(height * 0.044)))  # ~32px at 720p

    narrator = pysubs2.SSAStyle(
        fontname=base_font,
        fontsize=base_size,
        primarycolor=pysubs2.Color(255, 225, 130, 0),    # Active karaoke highlight (warm gold)
        secondarycolor=pysubs2.Color(245, 245, 248, 0),  # Pre-spoken text (crisp off-white)
        outlinecolor=pysubs2.Color(10, 12, 18, 0),
        backcolor=pysubs2.Color(0, 0, 0, 115),
        bold=True,
        outline=2.2,
        shadow=1.2,
        alignment=pysubs2.Alignment.BOTTOM_CENTER,
        marginl=64,
        marginr=64,
        marginv=36,
    )

    minh_khang = narrator.copy()
    minh_khang.primarycolor = pysubs2.Color(145, 225, 255, 0)  # Active cool cyan-white
    minh_khang.secondarycolor = pysubs2.Color(255, 255, 255, 0)

    linh_chi = narrator.copy()
    linh_chi.primarycolor = pysubs2.Color(255, 190, 165, 0)    # Active soft coral-amber
    linh_chi.secondarycolor = pysubs2.Color(255, 250, 245, 0)

    subs.styles["Narrator"] = narrator
    subs.styles["MinhKhang"] = minh_khang
    subs.styles["LinhChi"] = linh_chi


def _restore_punctuation(tokens: list[WordToken], dialogue: str) -> list[WordToken]:
    """Re-attach punctuation from ``dialogue`` onto Edge TTS WordBoundary tokens."""
    raw_words = [w for w in dialogue.split() if w]
    if len(raw_words) != len(tokens):
        return tokens
    restored: list[WordToken] = []
    for tok, orig in zip(tokens, raw_words):
        clean_orig = orig.strip(".,!?;:…\"'()[]{}")
        if clean_orig.lower() == tok.word.lower():
            restored.append(WordToken(word=orig, start=tok.start, end=tok.end))
        else:
            restored.append(tok)
    return restored


def _split_clause_balanced(clause: list[WordToken], max_words: int = 8) -> list[list[WordToken]]:
    """Split one clause/sentence into balanced lines without 1-2 word orphan tails."""
    n = len(clause)
    if n <= max_words:
        return [clause]
    n_chunks = (n + max_words - 1) // max_words
    base, rem = divmod(n, n_chunks)
    out: list[list[WordToken]] = []
    pos = 0
    for i in range(n_chunks):
        size = base + (1 if i < rem else 0)
        out.append(clause[pos : pos + size])
        pos += size
    return out


def _chunk_tokens(
    tokens: list[WordToken],
    max_words: int = 9,
    pause_split_sec: float = 0.35,
) -> list[list[WordToken]]:
    """Group word tokens into natural, balanced phrases split on pauses and punctuation."""
    if not tokens:
        return []
    clauses: list[list[WordToken]] = []
    current: list[WordToken] = []
    for tok in tokens:
        if current:
            gap = tok.start - current[-1].end
            if gap >= pause_split_sec:
                clauses.append(current)
                current = []
        current.append(tok)
        ends_sentence = tok.word.endswith((".", "!", "?", "…"))
        ends_clause = tok.word.endswith((",", ";", ":"))
        if ends_sentence or (ends_clause and len(current) >= 3):
            clauses.append(current)
            current = []
    if current:
        clauses.append(current)

    chunks: list[list[WordToken]] = []
    for clause in clauses:
        chunks.extend(_split_clause_balanced(clause, max_words=max_words))
    return chunks


def _format_karaoke_chunk(chunk: list[WordToken], speaker_prefix: str) -> str:
    """Build ASS karaoke override tags (\\kf in centiseconds) for a phrase chunk."""
    word_parts: list[str] = []
    for idx, tok in enumerate(chunk):
        if idx < len(chunk) - 1:
            span_sec = max(0.06, chunk[idx + 1].start - tok.start)
        else:
            span_sec = max(0.08, tok.end - tok.start)
        cs = max(1, int(round(span_sec * 100)))
        word_parts.append(f"{{\\kf{cs}}}{tok.word}")
    karaoke_line = " ".join(word_parts)
    if speaker_prefix:
        return f"{{\\alpha&H35&\\fs-3}}{speaker_prefix}: {{\\r}}{karaoke_line}"
    return karaoke_line


def build_subtitles(
    base_dir: Path,
    script_path: Path,
    out_ass: Path,
    out_srt: Path,
    episode: EpisodeScript | None = None,
) -> tuple[Path, Path]:
    """Assemble global-offset .ass and .srt subtitles locked to actual clip lengths."""
    if episode is None:
        episode = EpisodeScript.model_validate_json(script_path.read_text(encoding="utf-8"))
    subs_ass = pysubs2.SSAFile()
    _configure_styles(subs_ass, episode.width, episode.height)
    subs_srt = pysubs2.SSAFile()

    current_offset_sec = 0.0
    sub_dir = base_dir / "assets" / "subtitles"

    for shot in episode.shots:
        # Measure the ACTUAL rendered clip frame count when available so cumulative offset never drifts
        rel_clip = shot.video_clip_path or f"assets/clips/{shot.shot_id}.mp4"
        clip_path = base_dir / rel_clip
        if clip_path.exists():
            actual_frames = count_video_frames(clip_path)
            shot_dur_sec = actual_frames / float(episode.fps)
        else:
            if shot.duration_sec is None:
                raise RuntimeError(f"Shot {shot.shot_id} missing duration_sec; run step3_tts.py first.")
            shot_dur_sec = shot.duration_sec

        words_path = sub_dir / f"{shot.shot_id}_words.json"
        raw_tokens = json.loads(words_path.read_text(encoding="utf-8"))
        tokens = _restore_punctuation(
            [WordToken.model_validate(item) for item in raw_tokens],
            shot.dialogue,
        )

        style_name = CHARACTER_STYLES.get(shot.character, "Narrator")
        speaker = CHARACTER_DISPLAY_NAMES.get(shot.character, "")

        chunks = _chunk_tokens(tokens)
        for idx, chunk in enumerate(chunks):
            local_start = chunk[0].start
            max_end = (
                chunks[idx + 1][0].start - 0.02
                if idx + 1 < len(chunks)
                else shot_dur_sec - 0.03
            )
            local_end = min(chunk[-1].end + 0.14, max_end)
            local_end = max(local_end, min(max_end, local_start + 0.18))
            global_start_ms = int(round((current_offset_sec + local_start) * 1000.0))
            global_end_ms = int(round((current_offset_sec + local_end) * 1000.0))

            ass_text = f"{{\\fad(80,80)}}" + _format_karaoke_chunk(chunk, speaker)
            subs_ass.events.append(
                pysubs2.SSAEvent(
                    start=global_start_ms,
                    end=global_end_ms,
                    text=ass_text,
                    style=style_name,
                )
            )

            plain_phrase = " ".join(tok.word for tok in chunk)
            srt_text = f"{speaker}: {plain_phrase}" if speaker else plain_phrase
            subs_srt.events.append(
                pysubs2.SSAEvent(
                    start=global_start_ms,
                    end=global_end_ms,
                    text=srt_text,
                )
            )

        current_offset_sec += shot_dur_sec

    out_ass.parent.mkdir(parents=True, exist_ok=True)
    subs_ass.save(str(out_ass))
    subs_srt.save(str(out_srt))
    print(
        f"Subtitles saved: {out_ass.name} ({len(subs_ass.events)} cues) & {out_srt.name} | "
        f"Total timeline={current_offset_sec:.3f}s"
    )
    return out_ass, out_srt


def main() -> None:
    parser = argparse.ArgumentParser(description="Build global-offset .ass and .srt subtitles")
    parser.add_argument(
        "--script",
        type=Path,
        default=Path(__file__).resolve().parent / "script.json",
    )
    args = parser.parse_args()
    base_dir = args.script.resolve().parent
    out_dir = base_dir / "output"
    build_subtitles(
        base_dir=base_dir,
        script_path=args.script.resolve(),
        out_ass=out_dir / "episode_01.ass",
        out_srt=out_dir / "episode_01.srt",
    )


if __name__ == "__main__":
    main()
