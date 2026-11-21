/// <reference types="vite/client" />
import type { FeedbackRequest, FeedbackResponse, PredictionResult } from '../types/api'

export const MAX_IMAGE_BYTES = 10 * 1024 * 1024

export class ApiError extends Error {
  constructor(message: string, public readonly status?: number) {
    super(message)
    this.name = 'ApiError'
  }
}

export function validateImage(file: File): string | null {
  if (!['image/jpeg', 'image/png'].includes(file.type)) return 'Choose a JPEG or PNG image.'
  if (file.size === 0) return 'The image is empty. Choose another file.'
  if (file.size > MAX_IMAGE_BYTES) return 'The image must be 10 MB or smaller.'
  return null
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail: string | undefined
    try {
      const data: unknown = await response.json()
      if (typeof data === 'object' && data !== null && 'detail' in data) {
        const value = data.detail
        if (typeof value === 'string') detail = value
      }
    } catch { /* Non-JSON server errors use the status message below. */ }

    const messages: Record<number, string> = {
      400: 'This image could not be read. Choose a different JPEG or PNG.',
      404: 'This prediction is no longer available for feedback.',
      413: 'The image is too large. Choose one under 10 MB and 20 million pixels.',
      422: 'The request was incomplete. Check the image or correction and try again.',
      503: 'The AI models are still loading. Please try again shortly.',
    }
    throw new ApiError(messages[response.status] ?? detail ?? 'Something went wrong. Please try again.', response.status)
  }
  return response.json() as Promise<T>
}

async function request<T>(url: string, init: RequestInit): Promise<T> {
  const configured = Number(import.meta.env.VITE_REQUEST_TIMEOUT_MS)
  const timeout = Number.isSafeInteger(configured) && configured > 0 && configured <= 2147483647
    ? configured : 180_000
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeout)
  try {
    return await parseResponse<T>(await fetch(url, { ...init, signal: controller.signal }))
  } catch (error) {
    if (controller.signal.aborted) {
      throw new ApiError(url === '/predict'
        ? 'Analysis timed out. The server may still be processing your photo. Wait before choosing Retry analysis.'
        : 'Feedback timed out. Saving was not confirmed. You can retry with the same correction.')
    }
    if (error instanceof ApiError) throw error
    if (error instanceof SyntaxError) throw new ApiError('The server returned an unreadable response. Please try again.')
    throw new ApiError('Unable to reach the server. Check that the API is running and try again.')
  } finally {
    clearTimeout(timer)
  }
}

export function predictFood(image: File, analyzeIngredients: boolean): Promise<PredictionResult> {
  const validationError = validateImage(image)
  if (validationError) return Promise.reject(new ApiError(validationError))
  const form = new FormData()
  form.append('image', image)
  form.append('analyze_ingredients', String(analyzeIngredients))
  return request<PredictionResult>('/predict', { method: 'POST', body: form })
}

export async function submitFeedback(feedback: FeedbackRequest): Promise<FeedbackResponse> {
  const response = await request<FeedbackResponse>('/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(feedback),
  })
  if (!response || response.saved !== true || response.prediction_id !== feedback.prediction_id) {
    throw new ApiError('The server did not confirm this correction. You can retry saving.')
  }
  return response
}
