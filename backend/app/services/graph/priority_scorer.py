import json
from pathlib import Path


class PriorityScorer:
    def __init__(self) -> None:
        metadata_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "skill_metadata.json"
        try:
            with metadata_path.open(encoding="utf-8") as f:
                self.meta = json.load(f)
        except Exception:
            self.meta = {}

    def get_difficulty(self, skill: str) -> int:
        data = self.meta.get(skill, {"difficulty": 2, "importance": 3})
        return int(data.get("difficulty", 2))

    def score(self, skill: str) -> float:
        data = self.meta.get(skill, {"difficulty": 2, "importance": 3})

        difficulty = int(data.get("difficulty", 2))
        importance = int(data.get("importance", 3))

        if difficulty == 0:
            difficulty = 1

        return importance / difficulty
