"""Version constants.

These strings are written into every manifest so a render can be traced back to
the exact code and preprocessing conventions that produced it.
"""

from __future__ import annotations

APP_NAME = "garment-replacer"
APP_VERSION = "0.1.0"

#: Bump when frame extraction / mask conventions change in a way that
#: invalidates previously ingested templates.
PREPROCESSING_VERSION = "1"

#: Bump when the on-disk layout of a template or job directory changes.
LAYOUT_VERSION = "1"

#: Bump when the render manifest schema changes.
MANIFEST_SCHEMA_VERSION = "1"

__all__ = [
    "APP_NAME",
    "APP_VERSION",
    "LAYOUT_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "PREPROCESSING_VERSION",
]
