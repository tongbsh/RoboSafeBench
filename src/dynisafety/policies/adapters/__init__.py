"""Policy adapters — one module per model (or shared class + YAML variant).

Import individual adapter modules explicitly (or via YAML ``adapter:`` paths).
This package does **not** warn-and-skip on import failures.
"""

# Intentionally empty of eager model imports (S2 §1.8).
