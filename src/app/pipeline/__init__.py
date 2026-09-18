"""Pipeline services shared by the CLI and the API.

Every operator-facing action lives here exactly once; ``app/cli`` and
``app/api`` are thin adapters over these functions. That is what keeps the two
surfaces from drifting apart.
"""
