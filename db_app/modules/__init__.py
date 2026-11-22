"""One Shiny module per dashboard tab.

Each file exposes `<tab>_ui(id, ...)` and `<tab>_server(id, ...)`. Shiny
namespaces every input/output id inside a module, so tabs can reuse ids like
"table" without clashing, and each tab is developed and tested in isolation.
"""
