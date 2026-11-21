import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, predictFood, submitFeedback, validateImage } from './api'

afterEach(() => {
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
  vi.useRealTimers()
})

describe('image validation', () => {
  it('accepts JPEG and PNG and rejects invalid or oversized images', () => {
    expect(validateImage(new File(['a'], 'meal.jpg', { type: 'image/jpeg' }))).toBeNull()
    expect(validateImage(new File(['a'], 'meal.png', { type: 'image/png' }))).toBeNull()
    expect(validateImage(new File(['a'], 'meal.webp', { type: 'image/webp' }))).toMatch(/JPEG or PNG/)
    expect(validateImage(new File([], 'empty.png', { type: 'image/png' }))).toMatch(/empty/)
    const large = new File(['a'], 'large.png', { type: 'image/png' })
    Object.defineProperty(large, 'size', { value: 10 * 1024 * 1024 + 1 })
    expect(validateImage(large)).toMatch(/10 MB/)
  })
})

describe('API contract', () => {
  it.each([
    [new File(['x'], 'x.gif', { type: 'image/gif' }), /JPEG or PNG/],
    [new File([], 'x.png', { type: 'image/png' }), /empty/],
    [new File([new Uint8Array(10 * 1024 * 1024 + 1)], 'x.jpg', { type: 'image/jpeg' }), /10 MB/],
  ] as const)('blocks invalid uploads before sending', async (file, message) => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    await expect(predictFood(file, true)).rejects.toThrow(message)
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('aborts a slow request at the configured deadline without automatic retry', async () => {
    vi.useFakeTimers()
    vi.stubEnv('VITE_REQUEST_TIMEOUT_MS', '100')
    const fetchMock = vi.fn((_url: string, init: RequestInit) => new Promise((_resolve, reject) => {
      init.signal!.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')))
    }))
    vi.stubGlobal('fetch', fetchMock)
    const check = expect(predictFood(new File(['x'], 'x.png', { type: 'image/png' }), false))
      .rejects.toThrow(/Analysis timed out/)
    await vi.advanceTimersByTimeAsync(100)
    await check
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('rejects feedback acknowledged for a different prediction', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true, json: async () => ({ saved: true, prediction_id: 'other' }),
    }))
    await expect(submitFeedback({ prediction_id: 'mine', corrected_food: 'Pizza', confirmed_ingredients: [] }))
      .rejects.toThrow(/did not confirm/)
  })
  it('sends multipart predict fields without setting a content type', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ prediction_id: 'id-1' }) })
    vi.stubGlobal('fetch', fetchMock)
    const image = new File(['image'], 'meal.png', { type: 'image/png' })
    await predictFood(image, false)
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(url).toBe('/predict')
    expect(init.method).toBe('POST')
    expect(init.headers).toBeUndefined()
    expect(init.body).toBeInstanceOf(FormData)
    expect((init.body as FormData).get('image')).toBe(image)
    expect((init.body as FormData).get('analyze_ingredients')).toBe('false')
  })

  it('sends the actual feedback JSON schema', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ saved: true, prediction_id: 'id-1' }) })
    vi.stubGlobal('fetch', fetchMock)
    await submitFeedback({ prediction_id: 'id-1', corrected_food: 'Kabsa', confirmed_ingredients: ['rice'] })
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(url).toBe('/feedback')
    expect(init.headers).toEqual({ 'Content-Type': 'application/json' })
    expect(JSON.parse(init.body as string)).toEqual({ prediction_id: 'id-1', corrected_food: 'Kabsa', confirmed_ingredients: ['rice'] })
  })

  it.each([400, 413, 422, 503])('maps HTTP %i to a useful error', async (status) => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status, json: async () => ({ detail: 'server detail' }) }))
    await expect(predictFood(new File(['x'], 'meal.png', { type: 'image/png' }), true)).rejects.toMatchObject({ status })
  })

  it('reports network errors', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
    await expect(predictFood(new File(['x'], 'meal.png', { type: 'image/png' }), true)).rejects.toBeInstanceOf(ApiError)
  })
})
