import { ArrowRight, Camera, CheckCircle2, ChefHat, Heart, Leaf, Menu, ScanLine, ShieldCheck, Sparkles, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { ResultPanel } from './components/ResultPanel'
import { UploadZone } from './components/UploadZone'
import { predictFood, validateImage } from './lib/api'
import type { PredictionResult } from './types/api'

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [fileError, setFileError] = useState<string | null>(null)
  const [apiError, setApiError] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [analyzeIngredients, setAnalyzeIngredients] = useState(true)
  const [analyzing, setAnalyzing] = useState(false)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const inFlight = useRef(false)
  const scanId = useRef(0)

  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl) }, [previewUrl])

  useEffect(() => {
    if (window.location.hash === '#scanner') {
      document.getElementById('scanner')?.scrollIntoView?.({ behavior: 'instant' })
    }
  }, [])

  function chooseFile(selected: File) {
    const error = validateImage(selected)
    setFileError(error)
    setApiError(null)
    setResult(null)
    scanId.current += 1
    setFile(null)
    setPreviewUrl(null)
    if (error) return
    setFile(selected)
    setPreviewUrl(URL.createObjectURL(selected))
  }

  function clearFile() {
    setFile(null)
    setPreviewUrl(null)
    setFileError(null)
    setApiError(null)
    setResult(null)
    scanId.current += 1
  }

  async function analyze() {
    if (!file || inFlight.current) return
    inFlight.current = true
    const currentScan = scanId.current
    setAnalyzing(true)
    setApiError(null)
    setResult(null)
    try {
      const prediction = await predictFood(file, analyzeIngredients)
      if (currentScan === scanId.current) setResult(prediction)
    } catch (error) {
      if (currentScan === scanId.current) setApiError(error instanceof Error ? error.message : 'Analysis failed. Please try again.')
    } finally {
      inFlight.current = false
      setAnalyzing(false)
    }
  }

  return (
    <div className="app-shell">
      <header className="site-header">
        <div className="container flex items-center justify-between gap-4">
          <a href="#top" className="brand" aria-label="NutriVision AI home"><span className="brand-icon"><ScanLine size={23} strokeWidth={2.4} /></span><span>nutri<span className="text-orange-600">vision</span><span className="brand-ai">AI</span></span></a>
          <nav className="desktop-nav" aria-label="Main navigation"><a href="#how-it-works">How it works</a><a href="#scanner">Food scanner</a><a href="#about">About</a></nav>
          <a className="header-cta" href="#scanner">Start scanning <ArrowRight size={16} /></a>
          <button className="mobile-menu-button" type="button" aria-label={mobileMenuOpen ? 'Close menu' : 'Open menu'} aria-expanded={mobileMenuOpen} aria-controls="mobile-navigation" onClick={() => setMobileMenuOpen(!mobileMenuOpen)}>{mobileMenuOpen ? <X size={23} /> : <Menu size={23} />}</button>
        </div>
        {mobileMenuOpen && <nav id="mobile-navigation" className="mobile-nav" aria-label="Mobile navigation"><a href="#how-it-works" onClick={() => setMobileMenuOpen(false)}>How it works</a><a href="#scanner" onClick={() => setMobileMenuOpen(false)}>Food scanner</a><a href="#about" onClick={() => setMobileMenuOpen(false)}>About</a></nav>}
      </header>

      <main id="top">
        <section className="hero container" aria-labelledby="hero-heading">
          <div className="hero-copy">
            <span className="hero-badge"><Sparkles size={14} /> A clearer view of your plate</span>
            <h1 id="hero-heading">See your food <span>in a whole new light.</span></h1>
            <p className="hero-description">Snap a meal photo and explore AI food suggestions and an estimated calorie count. Simple insights, with you in control.</p>
            <div className="hero-actions"><a className="primary-button hero-button" href="#scanner">Scan your meal <ArrowRight size={18} /></a><a className="text-link" href="#how-it-works">How it works <ArrowRight size={17} /></a></div>
            <p className="hero-footnote"><ShieldCheck size={17} /> Your photo is sent for analysis; the API stores prediction details, not the image.</p>
          </div>
          <div className="hero-art" aria-hidden="true">
            <div className="hero-orbit orbit-one" /><div className="hero-orbit orbit-two" />
            <div className="floating-card floating-card-top"><span className="floating-icon"><Sparkles size={17} /></span><span><strong>AI food scan</strong><small>Explore your meal</small></span></div>
            <div className="plate-shadow" /><div className="plate"><div className="plate-inner"><div className="food-leaf leaf-one" /><div className="food-leaf leaf-two" /><div className="food-leaf leaf-three" /><div className="food-leaf leaf-four" /><div className="food-tomato tomato-one" /><div className="food-tomato tomato-two" /><div className="food-tomato tomato-three" /><div className="food-protein" /><div className="food-rice" /><div className="food-sprinkle sprinkle-one" /><div className="food-sprinkle sprinkle-two" /></div></div>
            <div className="floating-card floating-card-bottom"><span className="floating-icon heart-icon"><Heart size={18} /></span><span><strong>Mindful choices</strong><small>Estimates, not absolutes</small></span></div>
          </div>
        </section>

        <section id="how-it-works" className="steps-section" aria-labelledby="steps-heading"><div className="container"><div className="section-heading"><span className="eyebrow">Easy as 1, 2, 3</span><h2 id="steps-heading">From photo to insight</h2><p>A quick way to explore what might be on your plate.</p></div>
          <div className="steps-grid"><article className="step-card"><span className="step-icon peach"><Camera size={24} /></span><span className="step-number">01</span><h3>Upload a photo</h3><p>Choose a clear JPEG or PNG of your meal, up to 10 MB.</p></article><article className="step-card"><span className="step-icon orange"><ScanLine size={24} /></span><span className="step-number">02</span><h3>Let AI explore</h3><p>Food recognition and a calorie model analyze the whole image.</p></article><article className="step-card"><span className="step-icon green"><CheckCircle2 size={24} /></span><span className="step-number">03</span><h3>Review & correct</h3><p>Check the suggestions and save a correction when needed.</p></article></div>
        </div></section>

        <section id="scanner" className="scanner-section container" aria-labelledby="scanner-heading"><div className="scanner-intro"><span className="eyebrow"><Sparkles size={15} /> Your food scanner</span><h2 id="scanner-heading">What’s on your plate?</h2><p>Upload a meal photo to get an estimated calorie count and food suggestions.</p></div>
          <div className="scanner-grid"><div className="scanner-card"><div className="scanner-card-heading"><span className="scanner-card-icon"><Camera size={22} /></span><div><h3>Upload your meal</h3><p>A clear, well-lit photo works best.</p></div></div>
              <UploadZone file={file} previewUrl={previewUrl} error={fileError} busy={analyzing} onFile={chooseFile} onClear={clearFile} />
              <label className="ingredient-toggle"><input type="checkbox" checked={analyzeIngredients} disabled={analyzing} onChange={(event) => setAnalyzeIngredients(event.target.checked)} /><span><strong>Suggest possible ingredients</strong><small>Optional AI suggestions when available</small></span></label>
              <button type="button" className="primary-button analyze-button" disabled={!file || analyzing} onClick={analyze}>{analyzing ? <><span className="spinner" aria-hidden="true" /> Analyzing your meal…</> : <><ScanLine size={19} /> {apiError ? 'Retry analysis' : 'Analyze photo'} <ArrowRight size={18} /></>}</button>
              {analyzing && <p className="mt-3 text-center text-sm text-stone-500" role="status">AI analysis may take a little while. Please keep this page open.</p>}
              {apiError && <div className="error-box mt-4" role="alert">{apiError}</div>}
            </div>
            {result ? <ResultPanel result={result} /> : <div className="empty-card"><div className="empty-illustration"><ChefHat size={53} strokeWidth={1.4} /><span><Sparkles size={20} /></span></div><h3>{analyzing ? 'Finding food insights…' : 'Your insights will appear here'}</h3><p>{analyzing ? 'We’re analyzing the photo and preparing your estimates.' : 'Upload a meal photo and select Analyze photo to see AI suggestions and estimated calories.'}</p><div className="empty-divider" /><div className="empty-points"><span><Leaf size={16} /> Food suggestions</span><span><Sparkles size={16} /> Calorie estimate</span><span><CheckCircle2 size={16} /> Your corrections</span></div></div>}</div>
        </section>

        <section id="about" className="about-section"><div className="container about-content"><span className="about-icon"><Heart size={26} /></span><div><h2>A thoughtful way to explore food</h2><p>NutriVision AI is a research prototype. Results can be wrong, especially for mixed or unfamiliar dishes. Confirm the dish and portion yourself; calorie estimates are for the photographed plate and are not medical advice.</p></div></div></section>
      </main>
      <footer className="site-footer"><div className="container footer-content"><span className="brand text-lg"><span className="brand-icon"><ScanLine size={18} /></span><span>nutri<span className="text-orange-600">vision</span><span className="brand-ai">AI</span></span></span><span>Explore your plate with curiosity.</span><span>AI estimates · Your judgment matters</span></div></footer>
    </div>
  )
}
