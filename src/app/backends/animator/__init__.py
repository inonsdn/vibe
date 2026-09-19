"""Character animation backends.

Separate from :mod:`app.backends` (the *renderer* backends) because the jobs are
genuinely different:

============  ==============================  ==============================
              RendererBackend                 CharacterAnimatorBackend
============  ==============================  ==============================
Input         a real source frame + a mask     a Hero Character + a pose
Output        garment pixels inside a mask     a whole person, whole frame
Constraint    must not alter the performer     defines what the performer is
Ground truth  the master video                 nothing — it invents the pixels
============  ==============================  ==============================

That last row is why a synthetic master needs explicit operator acceptance and
a garment render does not.
"""

from app.backends.animator.base import (
    AnimationChunkRequest,
    AnimationChunkResult,
    AnimatorCapabilities,
    AnimatorContext,
    CharacterAnimatorBackend,
    ContextMode,
)
from app.backends.animator.registry import (
    available_animators,
    create_animator,
    register_animator,
)

__all__ = [
    "AnimationChunkRequest",
    "AnimationChunkResult",
    "AnimatorCapabilities",
    "AnimatorContext",
    "CharacterAnimatorBackend",
    "ContextMode",
    "available_animators",
    "create_animator",
    "register_animator",
]
