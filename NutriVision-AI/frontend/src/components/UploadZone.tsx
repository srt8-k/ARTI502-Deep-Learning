import { ImagePlus, UploadCloud, X } from 'lucide-react'
import { useRef, useState } from 'react'

interface UploadZoneProps {
  file: File | null
  previewUrl: string | null
  error: string | null
  busy: boolean
  onFile: (file: File) => void
  onClear: () => void
}

export function UploadZone({ file, previewUrl, error, busy, onFile, onClear }: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  return (
    <div>
      <input ref={inputRef} hidden disabled={busy} type="file" id="food-image" accept="image/jpeg,image/png" aria-label="Choose a food image"
        onChange={(event) => { const selected = event.target.files?.[0]; if (selected) onFile(selected); event.target.value = '' }} />
      {file && previewUrl ? (
        <div className="preview-wrap">
          <img src={previewUrl} alt="Selected meal preview" className="preview-image" />
          <div className="preview-footer">
            <div className="min-w-0"><p className="break-all font-semibold text-stone-900">{file.name}</p><p className="text-sm text-stone-500">{file.size < 1024 ? `${file.size} B` : file.size < 1024 * 1024 ? `${Math.ceil(file.size / 1024)} KB` : `${(file.size / 1024 / 1024).toFixed(1)} MB`}</p></div>
            <button type="button" className="change-image" disabled={busy} onClick={() => inputRef.current?.click()}>Change image</button>
            <button type="button" className="icon-button" onClick={onClear} disabled={busy} aria-label="Remove image"><X size={18} /></button>
          </div>
        </div>
      ) : (
        <div className={`drop-zone ${dragging ? 'drop-zone-active' : ''}`}
          onDragEnter={(event) => { event.preventDefault(); setDragging(true) }} onDragOver={(event) => event.preventDefault()}
          onDragLeave={(event) => { event.preventDefault(); setDragging(false) }}
          onDrop={(event) => { event.preventDefault(); setDragging(false); const selected = event.dataTransfer.files[0]; if (selected && !busy) onFile(selected) }}>
          <div className="upload-icon"><ImagePlus size={30} strokeWidth={1.7} /></div>
          <h3 className="mt-5 text-xl font-bold text-stone-900">Drop your food photo here</h3>
          <p className="mt-2 text-sm text-stone-500">Or choose an image from your device</p>
          <button type="button" className="choose-button" onClick={() => inputRef.current?.click()} disabled={busy}><UploadCloud size={18} /> Choose image</button>
          <p className="mt-5 text-xs text-stone-400">JPEG or PNG · Up to 10 MB</p>
        </div>
      )}
      {error && <p className="mt-3 text-sm font-medium text-red-700" role="alert">{error}</p>}
    </div>
  )
}
