"""Who may read the board, and who may change it.

Its own module because both the local console server and the relay need it, and
the relay must not import `server.py` - that pulls numpy, pandas and scipy in
through the advice path, for a process that only ever hands back JSON somebody
else computed.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass


@dataclass
class Access:
    """Who may read the board, and who may change it.

    Loopback with no tokens is the default, and is the only mode the console had
    until the board needed to reach a second manager: `_is_same_origin` was the
    whole guard, and it works because nothing off the machine can reach the port.

    Sharing changes that premise - the board is now on a public hostname - so the
    guard becomes a token instead. Two of them, because the roles differ: a guest
    reads any seat, but only the owner may rewind the board. `/undo` is the sole
    recovery hatch for a bad frame, and a guest who can fire it mid-draft is a
    draft-night hazard, not a convenience.

    `seat` is a view selector, NOT a security boundary: a guest may read any
    seat's panel. That is fine between two managers sharing a tool, and it is
    stated here so nobody later mistakes it for isolation.
    """

    owner: str = ""
    guest: str = ""

    @property
    def shared(self) -> bool:
        return bool(self.owner or self.guest)

    @classmethod
    def generate(cls) -> Access:
        return cls(owner=secrets.token_urlsafe(16), guest=secrets.token_urlsafe(16))
