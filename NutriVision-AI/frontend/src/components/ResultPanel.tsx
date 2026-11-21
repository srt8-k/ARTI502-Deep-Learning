import { AlertCircle, Check, CheckCircle2, ChevronDown, Clock3, Sparkles } from 'lucide-react'
import { useEffect, useState } from 'react'
import { submitFeedback } from '../lib/api'
import { displaySuggestion, provisionalSiglipSuggestion, strongCnnDinoAgreement } from '../lib/presentation'
import type { PredictionResult } from '../types/api'

function displayName(name: string) { return name.replaceAll('_', ' ') }

export function ResultPanel({ result }: { result: PredictionResult }) {
  const proposedName = displaySuggestion(result)
  const autoFromAgreement = result.suggested_food == null && strongCnnDinoAgreement(result) != null
  const provisionalFromSiglip = provisionalSiglipSuggestion(result) != null
  const isV2Food = result.food_present === true && result.food_presence_mode === 'v2'
  const [foodName, setFoodName] = useState(proposedName ?? '')
  const [confirmed, setConfirmed] = useState<string | null>(null)
  const [noFood, setNoFood] = useState(false)
  const [selectedIngredients, setSelectedIngredients] = useState<string[]>([])
  const [saved, setSaved] = useState(false)
  const [saving, setSaving] = useState(false)
  const [feedbackError, setFeedbackError] = useState<string | null>(null)

  useEffect(() => {
    setFoodName(proposedName ?? '')
    setConfirmed(null)
    setNoFood(false)
    setSelectedIngredients([])
    setSaved(false)
    setSaving(false)
    setFeedbackError(null)
  }, [result, proposedName])

  const isDirty = foodName.trim() !== (proposedName ?? '') || selectedIngredients.length > 0

  async function saveFeedback() {
    if (saving || !foodName.trim() || !result.user_can_correct_food || result.food_present === false || noFood) return
    setSaving(true)
    setFeedbackError(null)
    try {
      await submitFeedback({ prediction_id: result.prediction_id, corrected_food: foodName.trim(), confirmed_ingredients: selectedIngredients })
      setSaved(true)
      setConfirmed(foodName.trim())
    } catch (error) {
      setFeedbackError(error instanceof Error ? error.message : 'Unable to save your feedback.')
    } finally { setSaving(false) }
  }

  if (result.food_present === false) return (
    <section className="result-panel" role="status" aria-labelledby="no-food-heading">
      <span className="eyebrow"><CheckCircle2 size={16} /> Scan complete</span>
      <h2 id="no-food-heading" className="mt-6 text-3xl font-bold text-stone-900">No food detected</h2>
      <p className="mt-4 text-sm leading-6 text-stone-600">No dish, calories, or portion mass were estimated.</p>
      <p className="mt-3 text-sm text-stone-600">If the image contains food, try a clearer photo.</p>
      <p className="mt-3 text-xs text-stone-500">Analysis time: {result.inference_seconds.toFixed(1)}s</p>
    </section>
  )

  if (noFood) return <section className="result-panel" role="status">
    <h2 className="text-2xl font-bold">No food in this image</h2>
    <p className="mt-3 text-sm text-stone-600">You hid this result in your browser. No feedback was sent to the AI.</p>
  </section>

  const currentName = confirmed ?? proposedName
  const calories = typeof result.estimated_calories === 'number' && Number.isFinite(result.estimated_calories)
    ? result.estimated_calories : null
  const mass = typeof result.estimated_portion_mass_g === 'number' && Number.isFinite(result.estimated_portion_mass_g)
    ? result.estimated_portion_mass_g : null
  const needsManualChoice = proposedName == null && confirmed == null && !isV2Food

  return (
    <>
    <section className="result-panel" aria-labelledby="result-heading">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <span className="eyebrow"><Sparkles size={15} /> {!confirmed && !autoFromAgreement && (result.requires_confirmation || provisionalFromSiglip || !currentName) ? 'Needs confirmation' : isV2Food ? 'Food detected' : 'Scan complete'}</span>
        <span className="flex items-center gap-1.5 text-xs font-medium text-stone-500"><Clock3 size={14} /> {result.inference_seconds.toFixed(1)}s analysis</span>
      </div>
      {proposedName == null && <fieldset className="mt-4" disabled={saving}>
        <legend className="text-sm text-stone-700">No reliable dish name was available. {isV2Food ? 'The estimates below refer to the image; you can add a food name.' : 'Select a suggestion or enter a correction below.'}</legend>
        <div className="mt-2 flex flex-wrap gap-2">
          {(result.food_candidates ?? []).map(name => <button type="button" key={name} className="ingredient-button"
            aria-pressed={confirmed === name} onClick={() => {
              setConfirmed(name); setFoodName(name); setSaved(false); setFeedbackError(null)
            }}>{displayName(name)}</button>)}
        </div>
      </fieldset>}

      <div className="result-overview">
        <div className="min-w-0">
          <p className="text-sm font-medium text-stone-500">{confirmed ? 'User-selected food' : provisionalFromSiglip ? 'Possible dish · Please confirm' : 'AI food suggestion · unverified'}</p>
          <h2 id="result-heading" className="mt-1 text-3xl font-bold capitalize tracking-tight text-stone-900">
            {currentName ? displayName(currentName) : isV2Food ? 'Unknown dish' : 'Awaiting your confirmation'}
          </h2>
        </div>
        {(currentName || isV2Food) && <div className="nutrition-grid">
          {calories !== null && <div className="calorie-card">
            <span className="text-4xl font-extrabold tabular-nums">{Math.round(calories)}</span>
            <span className="ml-1 font-semibold">{result.calorie_unit}</span>
            <p className="text-xs font-medium">Estimated calories for the entire image</p>
          </div>}
          {mass !== null && <div className="calorie-card">
            <span className="text-4xl font-extrabold tabular-nums">{Math.round(mass)}</span>
            <span className="ml-1 font-semibold">g</span>
            <p className="text-xs font-medium">AI-estimated portion mass · whole image</p>
          </div>}
        </div>}
      </div>

      {needsManualChoice && <p className="mt-4 text-sm text-stone-600">Select a food name to see the calorie and portion estimates.</p>}
      <div className="notice mt-5" role="note">
        <AlertCircle size={18} className="shrink-0" />
        <span>Results are estimates. Please confirm your meal and portion.</span>
      </div>

    </section>
    <div className="analysis-details">
      <details>
        <summary>Correct This Result <ChevronDown size={18} /></summary>
        <div className="detail-content">
          <div className="mt-7 border-t border-orange-100 pt-6">
            <h3 className="section-title">Possible ingredients · unverified suggestions</h3>
            <p className="mt-1 text-sm text-stone-500">Select only ingredients you can confirm from the meal.</p>
            <fieldset className="mt-4" disabled={saving}><legend className="sr-only">Confirm suggested ingredients</legend>
              {result.possible_ingredients.length ? <div className="flex flex-wrap gap-2">{result.possible_ingredients.map(ingredient => {
                const selected = selectedIngredients.includes(ingredient)
                return <button key={ingredient} type="button" aria-pressed={selected} className={`ingredient-button ${selected ? 'ingredient-selected' : ''}`}
                  onClick={() => { setSelectedIngredients(selected ? selectedIngredients.filter(item => item !== ingredient) : [...selectedIngredients, ingredient]); setSaved(false) }}>
                  {selected && <Check size={14} />} {displayName(ingredient)}
                </button>
              })}</div> : <p className="text-sm text-stone-500">Ingredient suggestions are unavailable for this scan.</p>}
            </fieldset>
          </div>

          <div className="mt-7 border-t border-orange-100 pt-6">
            <h3 className="section-title">Help improve the record</h3>
            <p className="mt-1 text-sm text-stone-500">Saving feedback records your correction; it does not recalculate calories or retrain the AI.</p>
            <label htmlFor="corrected-food" className="mt-5 block text-sm font-semibold text-stone-800">Food name</label>
            <input id="corrected-food" className="text-input mt-2" value={foodName} maxLength={150}
              disabled={saving || !result.user_can_correct_food} onChange={event => { setFoodName(event.target.value); setSaved(false) }} />
            <p className="mt-2 text-xs text-stone-500">{saved ? 'Saved to feedback.' : isDirty ? 'Your edits are local until you save feedback.' : 'Review the food name before saving.'}</p>
            {feedbackError && <p className="mt-4 text-sm text-red-700" role="alert">{feedbackError}</p>}
            <div className="mt-5 flex flex-wrap items-center gap-3">
              <button type="button" className="primary-button" disabled={saving || !foodName.trim() || saved || !result.user_can_correct_food}
                onClick={saveFeedback}>{saving ? 'Saving…' : saved ? 'Feedback saved' : feedbackError ? 'Retry saving feedback' : 'Save feedback'}</button>
              {saved && <span className="flex items-center gap-1.5 text-sm font-medium text-emerald-700" role="status"><CheckCircle2 size={17} /> Correction saved</span>}
            </div>
          </div>
          <button type="button" className="ingredient-button mt-4" disabled={saving} onClick={() => setNoFood(true)}>
            Hide this result (I see no food)
          </button>
        </div>
      </details>
    </div>
    </>
  )
}
