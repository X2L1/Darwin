"""Video Agent – analyses and proposes improvements to Darwin's video capabilities.

Responsibilities:
* Temporal coherence analysis
* Frame quality monitoring
* Script / narration prompt optimisation
* Video dataset curation
* Codec and pipeline efficiency improvements
"""

from __future__ import annotations

from typing import Any, Dict, List

from darwin.agents.base import BaseAgent, Proposal

_MIN_COHERENCE_SCORE = 0.6
_MIN_FRAME_QUALITY = 0.5
_TARGET_FPS_CONSISTENCY = 0.95


class VideoAgent(BaseAgent):
    """Agent that improves Darwin's video-generation pipeline."""

    @property
    def domain(self) -> str:
        return "video"

    def analyse(self, context: Dict[str, Any]) -> List[Proposal]:
        """Analyse video context and return improvement proposals.

        Context keys:
          * ``coherence_scores``   – list of float (0-1) per video clip
          * ``frame_quality_avg``  – average frame aesthetic score
          * ``fps_consistency``    – ratio of clips with stable FPS
          * ``pipeline_latency_s`` – average generation latency in seconds
          * ``script_scores``      – dict script_id -> readability/quality score
          * ``max_proposals``      – cap on returned proposals
        """
        max_proposals = int(context.get("max_proposals", 5))
        proposals: List[Proposal] = []

        proposals.extend(self._analyse_temporal_coherence(context))
        proposals.extend(self._analyse_frame_quality(context))
        proposals.extend(self._analyse_fps_consistency(context))
        proposals.extend(self._analyse_pipeline_latency(context))
        proposals.extend(self._analyse_scripts(context))

        proposals.sort(key=lambda p: p.expected_improvement, reverse=True)
        return proposals[:max_proposals]

    # ------------------------------------------------------------------

    def _analyse_temporal_coherence(self, context: Dict[str, Any]) -> List[Proposal]:
        scores: List[float] = context.get("coherence_scores", [])
        if not scores:
            return []
        low = [s for s in scores if s < _MIN_COHERENCE_SCORE]
        if not low:
            return []
        avg_low = sum(low) / len(low)
        return [
            self._make_proposal(
                title=f"Improve temporal coherence ({len(low)}/{len(scores)} clips below threshold)",
                description=(
                    f"{len(low)} clips have temporal coherence below {_MIN_COHERENCE_SCORE:.0%}. "
                    f"Average for poor clips: {avg_low:.2f}. "
                    "Apply optical-flow smoothing or increase conditioning frames "
                    "to reduce jitter between frames."
                ),
                risk_score=0.25,
                expected_improvement=0.2,
                metadata={"n_low": len(low), "avg_low": round(avg_low, 3)},
            )
        ]

    def _analyse_frame_quality(self, context: Dict[str, Any]) -> List[Proposal]:
        fq = context.get("frame_quality_avg", None)
        if fq is None or fq >= _MIN_FRAME_QUALITY:
            return []
        return [
            self._make_proposal(
                title=f"Increase frame quality (current avg: {fq:.2f})",
                description=(
                    f"Average per-frame aesthetic score is {fq:.2f}, below the {_MIN_FRAME_QUALITY:.0%} target. "
                    "Upscale resolution, apply sharpening post-processing, or fine-tune the "
                    "video model on higher-quality samples."
                ),
                risk_score=0.3,
                expected_improvement=0.18,
                metadata={"frame_quality_avg": fq},
            )
        ]

    def _analyse_fps_consistency(self, context: Dict[str, Any]) -> List[Proposal]:
        fps_c = context.get("fps_consistency", 1.0)
        if fps_c >= _TARGET_FPS_CONSISTENCY:
            return []
        return [
            self._make_proposal(
                title=f"Fix FPS inconsistency ({fps_c:.0%} clips stable)",
                description=(
                    f"Only {fps_c:.0%} of generated clips have stable frame rates. "
                    "Standardise the output pipeline to enforce a fixed FPS target "
                    "and interpolate missing frames with RIFE."
                ),
                risk_score=0.2,
                expected_improvement=0.1,
                metadata={"fps_consistency": fps_c},
            )
        ]

    def _analyse_pipeline_latency(self, context: Dict[str, Any]) -> List[Proposal]:
        latency = context.get("pipeline_latency_s", None)
        if latency is None or latency <= 30:
            return []
        return [
            self._make_proposal(
                title=f"Reduce video generation latency ({latency:.0f}s → target <30s)",
                description=(
                    f"Average pipeline latency is {latency:.0f} s. "
                    "Profile the bottleneck stage and apply: model quantisation, "
                    "batch inference, or async frame generation."
                ),
                risk_score=0.2,
                expected_improvement=0.12,
                metadata={"latency_s": latency},
            )
        ]

    def _analyse_scripts(self, context: Dict[str, Any]) -> List[Proposal]:
        script_scores: Dict[str, float] = context.get("script_scores", {})
        low = {sid: sc for sid, sc in script_scores.items() if sc < 0.5}
        if not low:
            return []
        return [
            self._make_proposal(
                title=f"Improve {len(low)} low-quality video script(s)",
                description=(
                    f"{len(low)} video scripts scored below 0.5 on readability/quality. "
                    "Re-prompt with structured narrative templates and run through the "
                    "Prompting Agent for optimisation."
                ),
                risk_score=0.1,
                expected_improvement=0.1,
                metadata={"low_scripts": list(low.keys())[:5]},
            )
        ]
