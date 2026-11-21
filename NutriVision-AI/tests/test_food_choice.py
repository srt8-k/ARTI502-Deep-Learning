import unittest

from backend.food_choice import food_choice


class FoodChoiceTests(unittest.TestCase):
    def test_pizza_and_mandi_conflicts(self):
        for dino, other in [("pizza", "garlic bread"), ("paella", "lamb mandi with rice")]:
            with self.subTest(dino=dino):
                self.assertEqual(food_choice(dino, [other]), (None, [dino, other]))

    def test_agreement_ignores_case_spaces_and_underscores(self):
        self.assertEqual(food_choice("fried_rice", ["Fried Rice"]), ("Fried Rice", ["Fried Rice"]))

    def test_dino_fallback(self):
        self.assertEqual(food_choice("pizza", []), ("pizza", ["pizza"]))
