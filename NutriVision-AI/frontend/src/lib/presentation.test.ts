import { describe, expect, it } from 'vitest'
import { displaySuggestion, provisionalSiglipSuggestion, strongCnnDinoAgreement } from './presentation'
import type { PredictionResult } from '../types/api'

const base: PredictionResult = {
  prediction_id: 'mock',
  suggested_food: null,
  food_candidates: ['pizza', 'garlic bread'],
  food101_top3: [
    { name: 'pizza', softmax_score_percent: 84.7 },
    { name: 'garlic_bread', softmax_score_percent: 0.78 },
  ],
  cnn_food101_top3: [
    { name: 'pizza', softmax_score_percent: 86.12 },
    { name: 'garlic_bread', softmax_score_percent: 2.38 },
  ],
  open_vocab_suggestions: ['garlic bread'],
  possible_ingredients: [],
  estimated_calories: 399.7,
  estimated_portion_mass_g: 191.9,
  calorie_unit: 'kcal',
  requires_confirmation: true,
  user_can_correct_food: true,
  message: 'Unverified estimate',
  device: 'cpu',
  inference_seconds: 27.95,
  food_present: true,
  food_presence_mode: 'v2',
}

describe('conservative UI auto-suggestion rule', () => {
  it('uses matching strong CNN and DINOv2 scores for pizza', () => {
    expect(strongCnnDinoAgreement(base)).toBe('pizza')
    expect(displaySuggestion(base)).toBe('pizza')
  })
  it('never suggests food when V2 says no food', () => {
    expect(displaySuggestion({ ...base, food_present: false, suggested_food: 'pizza' })).toBeNull()
  })
  it('does not use CNN agreement when V2 is off', () => {
    expect(displaySuggestion({ ...base, food_presence_mode: 'off', food_present: null })).toBeNull()
  })
  it('does not accept low-scoring paella for mandi', () => {
    expect(strongCnnDinoAgreement({
      ...base,
      food101_top3: [
        { name: 'paella', softmax_score_percent: 32.03 },
        { name: 'baby_back_ribs', softmax_score_percent: 24.47 },
      ],
      cnn_food101_top3: [
        { name: 'paella', softmax_score_percent: 43.88 },
        { name: 'prime_rib', softmax_score_percent: 18.17 },
      ],
    })).toBeNull()
  })
  it('does not auto-suggest if model predictions disagree or margin is weak', () => {
    expect(strongCnnDinoAgreement({ ...base, cnn_food101_top3: [
      { name: 'garlic_bread', softmax_score_percent: 91 },
      { name: 'pizza', softmax_score_percent: 1 },
    ] })).toBeNull()
    expect(strongCnnDinoAgreement({ ...base, food101_top3: [
      { name: 'pizza', softmax_score_percent: 85 },
      { name: 'garlic_bread', softmax_score_percent: 80 },
    ] })).toBeNull()
  })
  it('honors the backend name where one is supplied', () => {
    expect(displaySuggestion({ ...base, suggested_food: 'margherita pizza' })).toBe('margherita pizza')
  })
})


describe('V2 uncertain dish presentation', () => {
  const mandi: PredictionResult = {
    ...base,
    food_candidates: ['paella', 'lamb mandi with rice'],
    food101_top3: [
      { name: 'paella', softmax_score_percent: 32.03 },
      { name: 'baby_back_ribs', softmax_score_percent: 24.47 },
    ],
    cnn_food101_top3: [
      { name: 'paella', softmax_score_percent: 43.88 },
      { name: 'prime_rib', softmax_score_percent: 18.17 },
    ],
    open_vocab_suggestions: ['lamb mandi with rice', 'Saudi Arabian lamb kabsa with rice'],
  }

  it('shows uncertain SigLIP2 first choice, never low-scoring paella, for V2 food', () => {
    expect(strongCnnDinoAgreement(mandi)).toBeNull()
    expect(provisionalSiglipSuggestion(mandi)).toBe('lamb mandi with rice')
    expect(displaySuggestion(mandi)).toBe('lamb mandi with rice')
  })
  it('does not override high-agreement CNN pizza', () => {
    expect(provisionalSiglipSuggestion(base)).toBeNull()
    expect(displaySuggestion(base)).toBe('pizza')
  })
  it('never uses SigLIP2 fallback when no food or when the detector is off', () => {
    expect(displaySuggestion({ ...mandi, food_present: false })).toBeNull()
    expect(provisionalSiglipSuggestion({ ...mandi, food_presence_mode: 'off' })).toBeNull()
    expect(displaySuggestion({ ...mandi, food_presence_mode: 'off', food_present: null })).toBeNull()
  })
  it('leaves dish unknown when the open-vocabulary model is disabled', () => {
    expect(displaySuggestion({ ...mandi, open_vocab_suggestions: [] })).toBeNull()
  })
})
