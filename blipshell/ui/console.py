"""The single Rich Console shared by the CLI and its views.

Lives in its own module so `ui/views.py` and `ui/cli.py` can both import it
without importing each other.
"""

from rich.console import Console

console = Console()
