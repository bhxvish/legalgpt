"""Waiting out Groq free-tier limits in long benchmark runs."""

import re
import time


class QuotaWaiter:
    """On a daily-limit error, wait the time Groq names (its window is rolling) instead of stopping -
    up to `max_hours` of waiting in total. Returns False when the run should stop."""

    def __init__(self, patient: bool, max_hours: float = 12.0) -> None:
        self.patient, self.budget = patient, max_hours * 3600

    def wait(self, error: Exception) -> bool:
        if not self.patient:
            return False
        m = re.search(r"in about (\d+) (minute|second)", str(error))
        seconds = (int(m.group(1)) * (60 if m.group(2) == "minute" else 1) if m else 600) + 30
        seconds = min(seconds, 1800)
        if seconds > self.budget:
            return False
        self.budget -= seconds
        print(f"  daily limit: waiting {seconds // 60} min ({self.budget / 3600:.1f} h of waiting left)", flush=True)
        time.sleep(seconds)
        return True
