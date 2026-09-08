"""Prompt loading helpers for coordinator and worker LLM policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass
class PromptBundle:
    """Loaded prompt pair and file provenance."""

    key: str
    system_prompt: str
    user_template: str
    system_path: str | None = None
    user_path: str | None = None

    def render_user(self, variables: Mapping[str, Any]) -> str:
        """Render the user template using simple ``{{name}}`` substitution.

        Args:
            variables: Mapping of template variable names to string-like values.

        Returns:
            Rendered user prompt string.
        """

        rendered = self.user_template
        for key, value in variables.items():
            rendered = rendered.replace("{{" + key + "}}", str(value))
        return rendered


class PromptStore:
    """Load prompt templates from a filesystem directory."""

    def __init__(self, root: Path | str = Path("prompts")):
        """Store the root directory used for prompt lookup.

        Args:
            root: Filesystem root containing prompt subdirectories.
        """

        self.root = Path(root)

    def load(self, key: str, default_system: str, default_user: str) -> PromptBundle:
        """Load prompt files for ``key`` or fall back to provided defaults.

        Args:
            key: Prompt subdirectory name under ``root``.
            default_system: Fallback system prompt text.
            default_user: Fallback user prompt template text.

        Returns:
            Prompt bundle with loaded text and source-path metadata.
        """

        prompt_dir = self.root / key
        system_path = prompt_dir / "system.txt"
        user_path = prompt_dir / "user.txt"

        system_prompt = system_path.read_text(encoding="utf-8") if system_path.exists() else default_system
        user_template = user_path.read_text(encoding="utf-8") if user_path.exists() else default_user

        return PromptBundle(
            key=key,
            system_prompt=system_prompt,
            user_template=user_template,
            system_path=str(system_path) if system_path.exists() else None,
            user_path=str(user_path) if user_path.exists() else None,
        )
