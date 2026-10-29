"""Future metrics: window precision/recall/F1, event recall, false alarms, state changes.
Ignore windows are excluded from labeled-window scores, not removed from online replay.
Unknown feeding-source intervals must not be counted as confirmed negative ground truth.
Seconds/hour metrics require verified timestamps/FPS.
"""

