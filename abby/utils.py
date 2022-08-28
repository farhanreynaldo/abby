"""Utils module."""


class Ratio:
    numerator: str
    denominator: str

    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator

    @property
    def name(self):
        return f"{self.numerator}/{self.denominator}"

    def __repr__(self):
        return f"{self.numerator}/{self.denominator}"
