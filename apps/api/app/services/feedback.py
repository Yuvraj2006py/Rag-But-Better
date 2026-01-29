from __future__ import annotations

from app.core.models import Feedback


class FeedbackStore:
    def __init__(self) -> None:
        self._items: list[Feedback] = []

    def add(self, feedback: Feedback) -> None:
        self._items.append(feedback)

    def all(self) -> list[Feedback]:
        return list(self._items)
