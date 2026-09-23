import { useEffect, useRef, useState } from 'react'
import {
  authEnabled,
  currentUser,
  onAuthChange,
  saveWithEmail,
  saveWithGoogle,
  signInWithEmail,
  signInWithGoogle,
  signOut,
} from '../lib/supabase'

function userLabel(user) {
  if (!user || user.is_anonymous) return null
  return user.email || user.user_metadata?.email || 'Signed in'
}

/**
 * Header control for the Supabase account.
 *
 * Anonymous visitors can attach an email or Google to their current user (the
 * project is kept), or sign in to an account they saved before. Signed-in
 * users can sign out. Hidden when auth is not configured.
 */
export default function AccountMenu() {
  const [user, setUser] = useState(null)
  const [open, setOpen] = useState(false)
  const [mode, setMode] = useState('save') // 'save' | 'signin'
  const [email, setEmail] = useState('')
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState(null) // { kind: 'ok' | 'error', text }
  const rootRef = useRef(null)

  useEffect(() => {
    if (!authEnabled) return undefined
    currentUser().then(setUser).catch(() => {})
    return onAuthChange(setUser)
  }, [])

  useEffect(() => {
    if (!open) return undefined
    const onDown = (e) => {
      if (rootRef.current && !rootRef.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [open])

  if (!authEnabled) return null

  const label = userLabel(user)

  async function run(action, success) {
    setBusy(true)
    setNotice(null)
    try {
      await action()
      if (success) setNotice({ kind: 'ok', text: success })
    } catch (err) {
      setNotice({ kind: 'error', text: err?.message || 'Something went wrong.' })
    } finally {
      setBusy(false)
    }
  }

  function submitEmail(e) {
    e.preventDefault()
    const address = email.trim()
    if (!address) return
    if (mode === 'save') {
      run(() => saveWithEmail(address), `Check ${address} and open the link to confirm. Your project stays as it is.`)
    } else {
      run(() => signInWithEmail(address), `Check ${address} for a sign-in link.`)
    }
  }

  async function handleSignOut() {
    await run(signOut)
    window.location.reload()
  }

  return (
    <div ref={rootRef} className="account-menu">
      <button
        type="button"
        className={`account-trigger${label ? ' signed-in' : ''}`}
        onClick={() => { setOpen((v) => !v); setNotice(null) }}
        title={label ? `Signed in as ${label}` : 'Your project is kept on this device. Save it to an account to keep it anywhere.'}
      >
        <span aria-hidden>{label ? '●' : '☁'}</span>
        <span className="account-trigger-text">{label || 'Save your project'}</span>
      </button>

      {open && (
        <div className="account-popover" role="dialog" aria-label="Account">
          {label ? (
            <>
              <p className="account-title">Signed in</p>
              <p className="account-body">{label}. Your project is saved to this account until you start a new one.</p>
              <button type="button" className="account-btn" disabled={busy} onClick={handleSignOut}>Sign out</button>
            </>
          ) : (
            <>
              <div className="account-tabs">
                <button type="button" className={mode === 'save' ? 'active' : ''} onClick={() => { setMode('save'); setNotice(null) }}>
                  Save this project
                </button>
                <button type="button" className={mode === 'signin' ? 'active' : ''} onClick={() => { setMode('signin'); setNotice(null) }}>
                  I have an account
                </button>
              </div>
              <p className="account-body">
                {mode === 'save'
                  ? 'Your project is saved anonymously on this device. Add an email or Google account to open it anywhere.'
                  : 'Sign in to open a project you saved before. The project on this device will not carry over.'}
              </p>
              <form onSubmit={submitEmail} className="account-form">
                <input
                  type="email"
                  required
                  placeholder="you@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  disabled={busy}
                />
                <button type="submit" className="account-btn primary" disabled={busy}>
                  {mode === 'save' ? 'Send link' : 'Email me a link'}
                </button>
              </form>
              <button
                type="button"
                className="account-btn"
                disabled={busy}
                onClick={() => run(mode === 'save' ? saveWithGoogle : signInWithGoogle)}
              >
                Continue with Google
              </button>
            </>
          )}
          {notice && <p className={`account-notice ${notice.kind}`}>{notice.text}</p>}
        </div>
      )}
    </div>
  )
}
