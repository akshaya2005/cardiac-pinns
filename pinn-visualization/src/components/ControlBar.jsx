import { useEffect, useRef, useState } from 'react'
import { MODEL_OPTIONS, COLOR_SCHEME_OPTIONS } from '../constants'

// Custom listbox instead of a native <select>: macOS renders the native
// dropdown popup itself, so CSS on <option> is ignored there regardless
// of browser. This gives us full control over colors/font.
function Dropdown({ label, placeholder, value, options, onChange }) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef(null)

  useEffect(() => {
    function handleClickOutside(e) {
      if (rootRef.current && !rootRef.current.contains(e.target)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const selected = options.find((opt) => opt.value === value)

  return (
    <div className="dropdown" ref={rootRef}>
      <button
        type="button"
        className="dropdown-trigger"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={label}
        onClick={() => setOpen((o) => !o)}
      >
        <span>{selected ? selected.label : placeholder}</span>
        <span className="dropdown-caret" aria-hidden="true">
          ▾
        </span>
      </button>
      {open && (
        <ul className="dropdown-list" role="listbox">
          {options.map((opt) => (
            <li
              key={opt.value}
              role="option"
              aria-selected={opt.value === value}
              className="dropdown-option"
              onClick={() => {
                onChange(opt.value)
                setOpen(false)
              }}
            >
              {opt.label}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export default function ControlBar({
  model,
  onModelChange,
  colorScheme,
  onColorSchemeChange,
  onReload,
  canReload,
  loading,
}) {
  return (
    <div className="control-bar">
      <Dropdown
        label="Select model"
        placeholder="Select model"
        value={model}
        onChange={onModelChange}
        options={MODEL_OPTIONS}
      />
      <Dropdown
        label="Select color scheme"
        placeholder="Select color scheme"
        value={colorScheme}
        onChange={onColorSchemeChange}
        options={COLOR_SCHEME_OPTIONS}
      />
      <span className="initial-condition-label">Initial condition: Planar</span>
      <button
        type="button"
        className="reload-button"
        onClick={onReload}
        disabled={!canReload || loading}
      >
        {loading ? 'Running…' : 'Reload'}
      </button>
    </div>
  )
}
