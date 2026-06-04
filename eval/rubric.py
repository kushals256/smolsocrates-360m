"""
Scoring rubric for evaluating Socratic tutoring quality.

Scores model responses on 5 dimensions (0-3 each, max 15 total):
1. No Direct Answer — model avoids giving code solutions
2. Asks Questions — response contains guiding questions
3. Correct Direction — questions point toward the right fix/concept
4. Progressive Hints — starts broad, gets specific
5. Encouragement — positive, supportive tone
"""

import re
from dataclasses import dataclass, field


@dataclass
class RubricScore:
    """Score for a single evaluation dimension."""

    name: str
    score: int  # 0-3
    explanation: str

    def __post_init__(self):
        self.score = max(0, min(3, self.score))


@dataclass
class EvalResult:
    """Complete evaluation result for a single test case."""

    test_id: str
    prompt: str
    response: str
    model_name: str
    scores: list[RubricScore] = field(default_factory=list)

    @property
    def total_score(self) -> int:
        return sum(s.score for s in self.scores)

    @property
    def max_score(self) -> int:
        return len(self.scores) * 3

    @property
    def percentage(self) -> float:
        if self.max_score == 0:
            return 0.0
        return (self.total_score / self.max_score) * 100

    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "prompt": self.prompt,
            "response": self.response,
            "model_name": self.model_name,
            "scores": [
                {"name": s.name, "score": s.score, "explanation": s.explanation}
                for s in self.scores
            ],
            "total_score": self.total_score,
            "max_score": self.max_score,
            "percentage": round(self.percentage, 1),
        }


# --- Rule-Based Heuristic Scorers ---


def score_no_direct_answer(response: str) -> RubricScore:
    """
    Score 0-3: Does the response avoid giving direct code solutions?

    3 = No code blocks at all, purely guiding text
    2 = Has small inline code (`var`) but no solution blocks
    1 = Has a code block but it's partial/illustrative, not a solution
    0 = Gives a complete code solution
    """
    # Count fenced code blocks
    code_blocks = re.findall(r"```[\s\S]*?```", response)
    # Count inline code
    inline_code = re.findall(r"`[^`]+`", response)
    # Check for solution-like phrases
    solution_phrases = [
        "here's the solution",
        "here is the solution",
        "here's the code",
        "here is the code",
        "here's the fix",
        "here is the fix",
        "try this code",
        "use this code",
        "the answer is",
        "the solution is",
        "here's how to do it",
        "here is how to do it",
        "you can do it like this",
        "just do",
        "simply use",
        "just use",
    ]
    has_solution_phrase = any(phrase in response.lower() for phrase in solution_phrases)

    if len(code_blocks) >= 2 or (len(code_blocks) >= 1 and has_solution_phrase):
        return RubricScore(
            "No Direct Answer",
            0,
            f"Contains {len(code_blocks)} code block(s) with solution language",
        )
    elif len(code_blocks) == 1:
        # Check if it's a short illustrative snippet vs. full solution
        block_content = code_blocks[0]
        lines = block_content.strip().split("\n")
        if len(lines) <= 3:
            return RubricScore(
                "No Direct Answer", 2, "Has a small illustrative code snippet"
            )
        else:
            return RubricScore(
                "No Direct Answer", 1, f"Has a code block with {len(lines)} lines"
            )
    elif len(inline_code) > 5:
        return RubricScore(
            "No Direct Answer",
            2,
            f"No code blocks but {len(inline_code)} inline code references",
        )
    else:
        return RubricScore(
            "No Direct Answer", 3, "No code solutions provided — purely guiding text"
        )


def score_asks_questions(response: str) -> RubricScore:
    """
    Score 0-3: Does the response contain guiding questions?

    3 = 3+ well-formed questions
    2 = 1-2 questions
    1 = Has question marks but they're rhetorical
    0 = No questions at all
    """
    # Count question marks (simple but effective)
    questions = [
        sent.strip()
        for sent in re.split(r"[.!?\n]", response)
        if "?" in sent and len(sent.strip()) > 10
    ]
    # Also check for question-starting phrases
    question_starters = [
        "what",
        "why",
        "how",
        "can you",
        "could you",
        "have you",
        "do you",
        "did you",
        "would",
        "which",
        "where",
        "when",
    ]
    strong_questions = [
        q
        for q in questions
        if any(q.strip().lower().startswith(s) for s in question_starters)
    ]

    if len(strong_questions) >= 3:
        return RubricScore(
            "Asks Questions",
            3,
            f"Contains {len(strong_questions)} strong guiding questions",
        )
    elif len(strong_questions) >= 1:
        return RubricScore(
            "Asks Questions", 2, f"Contains {len(strong_questions)} guiding question(s)"
        )
    elif response.count("?") >= 1:
        return RubricScore(
            "Asks Questions",
            1,
            "Has question marks but questions may be weak/rhetorical",
        )
    else:
        return RubricScore("Asks Questions", 0, "No questions found in response")


def score_encouragement(response: str) -> RubricScore:
    """
    Score 0-3: Does the response have a positive, supportive tone?

    3 = Multiple encouraging phrases + warm tone
    2 = Some encouragement present
    1 = Neutral tone, no negative language
    0 = Negative, dismissive, or condescending
    """
    encouraging_phrases = [
        "great",
        "good",
        "nice",
        "well done",
        "excellent",
        "perfect",
        "you're on the right track",
        "that's a good",
        "i see you",
        "interesting",
        "let's",
        "let us",
        "we can",
        "you can",
        "keep going",
        "almost",
        "close",
        "think about",
        "consider",
        "try thinking",
        "good question",
        "good start",
        "you've got",
        "💡",
        "🎯",
        "✨",
        "👍",
        "that's right",
        "exactly",
    ]
    negative_phrases = [
        "wrong",
        "incorrect",
        "that's bad",
        "you should know",
        "obviously",
        "clearly you",
        "that's not how",
        "no,",
    ]

    pos_count = sum(1 for phrase in encouraging_phrases if phrase in response.lower())
    neg_count = sum(1 for phrase in negative_phrases if phrase in response.lower())

    if neg_count >= 2:
        return RubricScore(
            "Encouragement", 0, f"Contains {neg_count} negative/dismissive phrases"
        )
    elif pos_count >= 3:
        return RubricScore(
            "Encouragement", 3, f"Contains {pos_count} encouraging phrases — warm tone"
        )
    elif pos_count >= 1:
        return RubricScore(
            "Encouragement", 2, f"Contains {pos_count} encouraging phrase(s)"
        )
    elif neg_count == 0:
        return RubricScore(
            "Encouragement",
            1,
            "Neutral tone — no negative language but limited encouragement",
        )
    else:
        return RubricScore("Encouragement", 0, "Negative or dismissive tone detected")


def score_rule_based(response: str) -> list[RubricScore]:
    """Run all rule-based heuristic scorers."""
    return [
        score_no_direct_answer(response),
        score_asks_questions(response),
        score_encouragement(response),
    ]
