import type { FoodPrediction, PredictionResult } from '../types/api'

export function normalizeFoodName(value: string): string {
  return value.replaceAll('_', ' ').toLowerCase().trim().replace(/\s+/g, ' ')
}

// Conservative display-only rule, NOT a calibrated confidence threshold.
// Never changes model inference, backend predictions, saved weights, or metrics.
// We only skip a forced manual choice if BOTH independent Food-101 classifiers
// strongly agree on the same in-vocabulary class, and V2 detected food.
export function strongCnnDinoAgreement(result: PredictionResult): string | null {
  if (result.food_present !== true || result.food_presence_mode !== 'v2') return null
  const dino = result.food101_top3 ?? []
  const cnn = result.cnn_food101_top3 ?? []
  if (dino.length < 2 || cnn.length < 2) return null

  const hasStrongLead = (top: FoodPrediction, second: FoodPrediction) =>
    Number.isFinite(top.softmax_score_percent) &&
    Number.isFinite(second.softmax_score_percent) &&
    top.softmax_score_percent >= 80 &&
    top.softmax_score_percent - second.softmax_score_percent >= 20

  if (!hasStrongLead(dino[0], dino[1]) || !hasStrongLead(cnn[0], cnn[1])) return null
  if (!normalizeFoodName(dino[0].name) || normalizeFoodName(dino[0].name) !== normalizeFoodName(cnn[0].name)) return null
  return normalizeFoodName(dino[0].name)
}

// Fallback DISPLAY label for V2 FOOD when the CNNs are uncertain.
// SigLIP2 supplies an unverified *candidate*, never a confirmed classification.
// This selection does not change the regression prediction or Food-101 output.
export function provisionalSiglipSuggestion(result: PredictionResult): string | null {
  if (result.food_present !== true || result.food_presence_mode !== 'v2') return null
  if (result.suggested_food || strongCnnDinoAgreement(result)) return null
  const top = result.open_vocab_suggestions?.[0]
  return typeof top === 'string' && top.trim() ? top.trim() : null
}

export function displaySuggestion(result: PredictionResult): string | null {
  if (result.food_present === false) return null
  // Preserve explicit backend output, then strong CNN agreement, then a
  // clearly identified provisional open-vocabulary *display* suggestion.
  return result.suggested_food ?? strongCnnDinoAgreement(result) ?? provisionalSiglipSuggestion(result)
}
