export interface FoodPrediction {
  name: string
  softmax_score_percent: number
}

export interface PredictionResult {
  prediction_id: string
  suggested_food: string | null
  food_candidates?: string[]
  food101_top3: FoodPrediction[]
  cnn_food101_top3?: FoodPrediction[]
  open_vocab_suggestions: string[]
  possible_ingredients: string[]
  estimated_calories: number | null
  estimated_portion_mass_g?: number | null
  calorie_unit: string
  requires_confirmation: boolean
  user_can_correct_food: boolean
  message: string
  device: string
  inference_seconds: number
  // Optional to keep this UI compatible with the legacy backend (mode off).
  food_present?: boolean | null
  food_presence_mode?: 'off' | 'v2' | string
  food_presence_score_percent?: number | null
  food_presence_threshold?: number | null
}

export interface FeedbackRequest {
  prediction_id: string
  corrected_food: string
  confirmed_ingredients: string[]
}

export interface FeedbackResponse {
  saved: boolean
  prediction_id: string
  corrected_food: string
}
