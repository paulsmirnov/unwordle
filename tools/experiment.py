import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

BASE_PATH = Path(os.path.dirname(__file__))
WORD_LENGTH = 5

LetterScores = dict[str, float]


class WordStats:
    overall: LetterScores
    """`overall[c]` is the probability that a letter `c` appears in a word."""

    def __init__(self, words: list[str]) -> None:
        counter = Counter()
        for word in words:
            counter.update(set(word))
        self.overall = normalize_counter(counter, len(words))


class PositionalWordStats(WordStats):
    positional: list[LetterScores]
    """`positional[i][c]` is the probability that a letter `c` appears in a word in the position `i`."""

    def __init__(self, words: list[str]) -> None:
        super().__init__(words)

        counters = defaultdict(Counter)
        for word in words:
            for index, letter in enumerate(word):
                counters[index][letter] += 1
        self.positional = [normalize_counter(counter, len(words)) for counter in counters.values()]


def normalize_counter(counter: Counter[str], norm: float) -> LetterScores:
    return {key: value / norm for key, value in counter.most_common()}


class GuessScorer:
    """A simple scorer based on probability to encounter a letter in an English word.

    Note: The score is greater if the guess gives more information after it is accepted or rejected
    (reduces the number of possible words).

    It ignores the position of the letters in the word.
    It also ignores duplicate letters as they don't bring any additional information.
    """

    def __init__(self, stats: WordStats) -> None:
        assert isinstance(stats, WordStats)
        self.stats = stats

    def score(self, guess: str) -> float:
        result = 0.0
        for letter in set(guess):
            f = self.stats.overall.get(letter, 0.0)

            # gray: the score if it is absent
            a = (1.0 - f) * f
            assert a >= 0

            # yellow or green: the score if it is present
            b = f * (1.0 - f)
            assert b >= 0

            # sum outcomes
            result += a + b
        return result / len(guess)

    def score_list(self, guesses: Iterable[str]) -> list[float]:
        return [self.score(guess) for guess in guesses]

    def score_dict(self, guesses: Iterable[str]) -> dict[str, float]:
        return {guess: self.score(guess) for guess in guesses}


class PositionalGuessScorer(GuessScorer):
    """A scorer based on probability to encounter a letter in specific position of a word.

    Note: The score is greater if the guess gives more information after it is accepted or rejected
    (reduces the number of possible words).

    It ignores dependencies between the letters in the word.
    """

    def __init__(self, stats: PositionalWordStats) -> None:
        assert isinstance(stats, PositionalWordStats)
        super().__init__(stats)

    def score(self, word: str) -> float:
        result = 0.0
        letters = defaultdict(list)
        for i, letter in enumerate(word):
            f = self.stats.overall.get(letter, 0.0)
            p = self.stats.positional[i].get(letter, 0.0)

            # gray: the score if it is absent
            a = (1.0 - f) * f
            assert a >= 0

            # yellow: the score if it is present in other positions
            b = (f - p) * (1.0 - f + p)
            assert b >= 0

            # green: the score if it is present in this position
            c = p * (1.0 - p)
            assert c >= 0

            letters[letter].append((a, b, c))

        # keep only the best score for each repeated letter
        for letter, outcomes in letters.items():
            a_scores, b_scores, c_scores = zip(*outcomes)
            result += max(a_scores) + max(b_scores) + max(c_scores)

        return result / len(word)


def main():
    words = read_words(BASE_PATH / "ospd.txt", WORD_LENGTH)
    scorer = PositionalGuessScorer(PositionalWordStats(words))
    scores = scorer.score_dict(words)

    for word in ["hello", "cruel", "abbey", "aargh", "spine"]:
        print(word, scores.get(word))

    print("---")

    sorted_scores = sorted((v, k) for k, v in scores.items())
    for rank, word in sorted_scores[:1] + sorted_scores[-1:]:
        print(word, rank)


def read_words(filepath: Path, length: int) -> list[str]:
    with filepath.open() as f:
        words = (line.strip() for line in f)
        words = [word for word in words if len(word) == length]
    return words


if __name__ == "__main__":
    main()
