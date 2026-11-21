import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { ResultPanel } from './components/ResultPanel'

const result = {
  prediction_id: 'prediction-1', suggested_food: 'chicken_kabsa',
  food101_top3: [{ name: 'fried_rice', softmax_score_percent: 52.3 }],
  open_vocab_suggestions: ['chicken_kabsa', 'chicken mandi'], possible_ingredients: ['chicken', 'rice'],
  estimated_calories: 612.4, calorie_unit: 'kcal', requires_confirmation: true,
  user_can_correct_food: true, message: 'Confirm the dish.', device: 'cpu', inference_seconds: 2.1,
}

afterEach(() => { cleanup(); vi.unstubAllGlobals() })

describe('scanner states', () => {
  it.each([['pizza', 'garlic bread'], ['paella', 'lamb mandi']])('requires an explicit choice for %s', async (first, second) => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ saved: true, prediction_id: result.prediction_id }) })
    vi.stubGlobal('fetch', fetchMock)
    render(<ResultPanel result={{ ...result, suggested_food: null, food_candidates: [first, second] }} />)
    await userEvent.click(screen.getByText('Correct This Result'))
    expect(screen.getByRole('button', { name: first })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Awaiting your confirmation' })).toBeInTheDocument()
    expect(screen.queryByText('kcal')).not.toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: second }))
    expect(screen.getByRole('heading', { name: second })).toBeInTheDocument()
    expect(screen.getByText('612')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Save feedback' }))
    expect(await screen.findByText('Correction saved')).toBeInTheDocument()
    expect(JSON.parse(fetchMock.mock.calls[0][1].body).corrected_food).toBe(second)
  })

  it('dismisses a non-food analysis locally and removes every food detail', async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    render(<ResultPanel result={result} />)
    await userEvent.click(screen.getByText('Correct This Result'))
    await userEvent.click(screen.getByRole('button', { name: 'Hide this result (I see no food)' }))
    expect(screen.getByRole('heading', { name: 'No food in this image' })).toBeInTheDocument()
    for (const detail of ['chicken kabsa', 'rice', 'kcal', '612', 'fried rice', 'Save feedback']) {
      expect(screen.queryByText(detail)).not.toBeInTheDocument()
    }
    expect(fetchMock).not.toHaveBeenCalled()
  })
  it('shows an empty state and rejects unsupported files', async () => {
    render(<App />)
    expect(screen.getByText('Your insights will appear here')).toBeInTheDocument()
    await userEvent.upload(screen.getByLabelText('Choose a food image'), new File(['x'], 'food.webp', { type: 'image/webp' }), { applyAccept: false })
    expect(screen.getByRole('alert')).toHaveTextContent('JPEG or PNG')
    expect(screen.getByRole('button', { name: /Analyze photo/ })).toBeDisabled()
  })

  it('shows loading, estimated results, and saves a correction', async () => {
    let resolvePredict!: (value: unknown) => void
    const fetchMock = vi.fn()
      .mockImplementationOnce(() => new Promise((resolve) => { resolvePredict = resolve }))
      .mockResolvedValueOnce({ ok: false, status: 503, json: async () => ({}) })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ saved: true, prediction_id: 'prediction-1', corrected_food: 'Chicken Kabsa' }) })
    vi.stubGlobal('fetch', fetchMock)
    render(<App />)
    await userEvent.upload(screen.getByLabelText('Choose a food image'), new File(['image'], 'meal.png', { type: 'image/png' }))
    const analyze = screen.getByRole('button', { name: /Analyze photo/ })
    await userEvent.click(analyze)
    expect(screen.getByRole('button', { name: /Analyzing your meal/ })).toBeDisabled()
    resolvePredict({ ok: true, json: async () => result })
    expect(await screen.findByRole('heading', { name: 'chicken kabsa' })).toBeInTheDocument()
    expect(screen.getByText('Estimated calories for the entire image')).toBeInTheDocument()
    expect(screen.queryByText('Advanced AI Analysis')).not.toBeInTheDocument()
    await userEvent.click(screen.getByText('Correct This Result'))
    expect(screen.queryByText(/fried rice|52\.3%|chicken mandi/i)).not.toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('Food name'), { target: { value: 'Chicken Kabsa' } })
    expect(screen.getByText(/edits are local/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'chicken' }))
    await userEvent.click(screen.getByRole('button', { name: 'Save feedback' }))
    expect(await screen.findByRole('alert')).toBeInTheDocument()
    expect(screen.queryByText('Correction saved')).not.toBeInTheDocument()
    expect(screen.getByLabelText('Food name')).toHaveValue('Chicken Kabsa')
    await userEvent.click(screen.getByRole('button', { name: 'Retry saving feedback' }))
    await waitFor(() => expect(screen.getByText('Correction saved')).toBeInTheDocument())
    const [, feedbackInit] = fetchMock.mock.calls[2] as [string, RequestInit]
    expect(JSON.parse(feedbackInit.body as string)).toEqual({ prediction_id: 'prediction-1', corrected_food: 'Chicken Kabsa', confirmed_ingredients: ['chicken'] })
  })

  it('retains the photo for manual retry after disconnect', async () => {
    const fetchMock = vi.fn().mockRejectedValueOnce(new TypeError('Failed to fetch'))
      .mockResolvedValueOnce({ ok: true, json: async () => result })
    vi.stubGlobal('fetch', fetchMock)
    render(<App />)
    await userEvent.upload(screen.getByLabelText('Choose a food image'), new File(['x'], 'food.png', { type: 'image/png' }))
    await userEvent.click(screen.getByRole('button', { name: /Analyze photo/ }))
    expect(await screen.findByRole('alert')).toHaveTextContent(/Unable to reach/)
    expect(fetchMock).toHaveBeenCalledTimes(1)
    await userEvent.click(screen.getByRole('button', { name: 'Retry analysis' }))
    expect(await screen.findByText('Needs confirmation')).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })
})

describe('V2 food detection integration', () => {
  it('shows V2 no-food result without any food, calories, portion, or correction UI', () => {
    render(<ResultPanel result={{
      ...result,
      food_present: false,
      food_presence_mode: 'v2',
      food_presence_score_percent: 0.01,
      suggested_food: null,
      food_candidates: [],
      food101_top3: [],
      cnn_food101_top3: [],
      estimated_calories: null,
      estimated_portion_mass_g: null,
      possible_ingredients: [],
      requires_confirmation: false,
      user_can_correct_food: false,
    }} />)
    expect(screen.getByRole('heading', { name: 'No food detected' })).toBeInTheDocument()
    expect(screen.getByText(/No dish, calories, or portion mass were estimated/)).toBeInTheDocument()
    expect(screen.queryByText('kcal')).not.toBeInTheDocument()
    expect(screen.queryByText('g')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Save feedback' })).not.toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: /Food-101/ })).not.toBeInTheDocument()
  })

  it('auto-displays pizza when V2, CNN and DINOv2 strongly agree, even if SigLIP2 disagrees', () => {
    render(<ResultPanel result={{
      ...result,
      suggested_food: null,
      food_candidates: ['pizza', 'garlic bread'],
      food_present: true,
      food_presence_mode: 'v2',
      food_presence_score_percent: 99.96,
      food101_top3: [
        { name: 'pizza', softmax_score_percent: 84.7 },
        { name: 'garlic_bread', softmax_score_percent: 0.78 },
      ],
      cnn_food101_top3: [
        { name: 'pizza', softmax_score_percent: 86.12 },
        { name: 'garlic_bread', softmax_score_percent: 2.38 },
      ],
      open_vocab_suggestions: ['garlic bread', 'pizza'],
      estimated_calories: 399.7,
      estimated_portion_mass_g: 191.9,
      requires_confirmation: true,
    }} />)
    expect(screen.getByRole('heading', { name: 'pizza' })).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Awaiting your confirmation' })).not.toBeInTheDocument()
    expect(screen.getByText('400')).toBeInTheDocument()
    expect(screen.getByText('192')).toBeInTheDocument()
    expect(screen.getByText('AI-estimated portion mass · whole image')).toBeInTheDocument()
    expect(screen.getByRole('note')).toHaveTextContent('Results are estimates. Please confirm your meal and portion.')
    expect(screen.queryByText('garlic bread')).not.toBeInTheDocument()
    expect(screen.getByLabelText('Food name')).toHaveValue('pizza')
  })

  it('does not incorrectly auto-label an ambiguous Saudi meal as paella', async () => {
    render(<ResultPanel result={{
      ...result,
      suggested_food: null,
      food_candidates: ['paella', 'lamb mandi with rice'],
      food_present: true,
      food_presence_mode: 'v2',
      food101_top3: [
        { name: 'paella', softmax_score_percent: 32.03 },
        { name: 'baby_back_ribs', softmax_score_percent: 24.47 },
      ],
      cnn_food101_top3: [
        { name: 'paella', softmax_score_percent: 43.88 },
        { name: 'prime_rib', softmax_score_percent: 18.17 },
      ],
      open_vocab_suggestions: ['lamb mandi with rice'],
      estimated_calories: 530.2,
      estimated_portion_mass_g: 504,
      requires_confirmation: true,
    }} />)
    expect(screen.getByRole('heading', { name: 'lamb mandi with rice' })).toBeInTheDocument()
    expect(screen.getByText('530')).toBeInTheDocument()
    expect(screen.getByText('504')).toBeInTheDocument()
    expect(screen.getByText('Possible dish · Please confirm')).toBeInTheDocument()
    expect(screen.queryByText(/SigLIP2|ConvNeXt|DINOv2/i)).not.toBeInTheDocument()
    expect(screen.getByRole('note')).toHaveTextContent('Results are estimates. Please confirm your meal and portion.')
    expect(screen.queryByRole('heading', { name: 'paella' })).not.toBeInTheDocument()
    expect(screen.getByLabelText('Food name')).toHaveValue('lamb mandi with rice')
  })

  it('shows independent estimates with Unknown dish when SigLIP2 is disabled', () => {
    render(<ResultPanel result={{
      ...result,
      suggested_food: null,
      food_present: true,
      food_presence_mode: 'v2',
      food_candidates: [],
      open_vocab_suggestions: [],
      estimated_portion_mass_g: 504,
    }} />)
    expect(screen.getByRole('heading', { name: 'Unknown dish' })).toBeInTheDocument()
    expect(screen.getByText('612')).toBeInTheDocument()
    expect(screen.getByText('504')).toBeInTheDocument()
    expect(screen.getByRole('note')).toHaveTextContent('Results are estimates. Please confirm your meal and portion.')
  })

  it('displays automatic donut estimates without a portion input', () => {
    render(<ResultPanel result={{ ...result, suggested_food: 'donuts', estimated_portion_mass_g: 254, estimated_calories: 408 }} />)
    expect(screen.getByRole('heading', { name: 'donuts' })).toBeInTheDocument()
    expect(screen.getByText('254')).toBeInTheDocument()
    expect(screen.getByText('408')).toBeInTheDocument()
    expect(screen.queryByRole('spinbutton', { hidden: true })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /adjust portion/i })).not.toBeInTheDocument()
  })
})
