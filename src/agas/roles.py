"""Worker roles for AGAS.

The paper defines exactly four worker roles, denoted ``{PR, SN, CA, IN}`` in
\\autoref{sec:method}:

* ``PR`` – Profiler. Submits a small number of safe interactions to probe the
  platform's reaction and discover new receptive users (see ``method_profiler``).
* ``SN`` – Sniper. Payload role reserved for high-trust / low-risk workers;
  delivers the actual rank-pushing action (see ``method_sniper``).
* ``CA`` – Camouflageur. Stealth role: rates benign or weakly related items to
  rebuild trust and dilute fake-user history (see ``method_camouflaguer``).
* ``IN`` – Inactive. No action this round; used to cool down or quarantine
  suspicious workers (see ``method_inactive``).

The short symbol names match the paper exactly so that ``code -> paper`` is a
trivial trace for a reviewer.
"""

from __future__ import annotations

from enum import Enum


class Role(str, Enum):
    """The four AGAS worker roles, denoted ``{PR, SN, CA, IN}`` in the paper."""

    PR = "PR"
    SN = "SN"
    CA = "CA"
    IN = "IN"

    @classmethod
    def all(cls) -> list["Role"]:
        """Return the canonical paper ordering ``[PR, SN, CA, IN]``."""

        return [cls.PR, cls.SN, cls.CA, cls.IN]


# Long-form display labels used by the LLM prompts.  These keep the prompts
# readable while the code-side comparisons stay on the short paper symbols.
ROLE_LONG_NAMES: dict[Role, str] = {
    Role.PR: "Profiler",
    Role.SN: "Sniper",
    Role.CA: "Camouflageur",
    Role.IN: "Inactive",
}


__all__ = ["Role", "ROLE_LONG_NAMES"]
