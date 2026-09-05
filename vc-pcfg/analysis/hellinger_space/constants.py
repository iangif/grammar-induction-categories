"""Shared constants for Hellinger-space analyses."""

from __future__ import annotations

PACKAGE_VERSION = "0.1.0"

REQUIRED_COLUMNS: tuple[str, ...] = (
    "category",
    "domain",
    "position",
    "feature",
    "value",
    "proportion",
)

POSITION_ORDER: tuple[str, ...] = ("TARGET", "L2", "L1", "R1", "R2")
POSITION_RANK = {position: index for index, position in enumerate(POSITION_ORDER)}

TARGET_DOMAIN = "lexical"
CONTEXT_DOMAIN = "contextual"

BLOCK_SUM_ATOL = 1e-8
BLOCK_SUM_RTOL = 1e-8
