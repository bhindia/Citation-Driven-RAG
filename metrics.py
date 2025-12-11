from typing import List

class FourMetrics:
    def __init__(self):
        self.total = 0
        self.hallucation_sum = 0.0
        self.grounded_sum = 0.0
        self.correct_count = 0

    def update(self, answer_text: str, selected_spans: List[str]):
        self.total += 1
        abstained = answer_text.startswith("I don't know")
        n_supported = len(selected_spans)

        halluc_frac = 0.0 if n_supported == 0 else 1.0 - (n_supported / max(1, n_supported))
        grounded_frac = 0.0 if n_supported == 0 else n_supported / max(1, n_supported)

        self.hallucation_sum += halluc_frac
        self.grounded_sum += grounded_frac

        if abstained or n_supported > 0:
            self.correct_count += 1

    def report(self):
        answered = max(1, self.total)
        return {
            "Total": self.total,
            "Accuracy": round(self.correct_count / self.total, 4),
            "Hallucination_Rate": round(self.hallucation_sum / answered, 4),
            "Groundedness_Score": round(self.grounded_sum / answered, 4)
        }
