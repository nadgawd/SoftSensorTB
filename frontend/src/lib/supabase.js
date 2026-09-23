/**
 * Supabase sign-in.
 *
 * Every visitor gets an anonymous user on first load, so their project is
 * saved server-side without a sign-up form. "Save your project" later attaches
 * an email or Google identity to that same user id, which keeps the project.
 *
 * Without VITE_SUPABASE_URL / VITE_SUPABASE_ANON_KEY (local development with
 * run.sh) auth is off and requests go out without a token.
 */
import { createClient } from '@supabase/supabase-js'

const url = import.meta.env.VITE_SUPABASE_URL
const anonKey = import.meta.env.VITE_SUPABASE_ANON_KEY

export const supabase = url && anonKey
  ? createClient(url, anonKey, {
    auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true, flowType: 'pkce' },
  })
  : null

export const authEnabled = Boolean(supabase)

let pendingSignIn = null

/** The current session, signing in anonymously the first time. */
export async function ensureSession() {
  if (!supabase) return null
  const { data } = await supabase.auth.getSession()
  if (data.session) return data.session
  pendingSignIn ??= supabase.auth.signInAnonymously().then(({ data: signIn, error }) => {
    pendingSignIn = null
    if (error) throw error
    return signIn.session
  })
  return pendingSignIn
}

export async function authHeaders() {
  const session = await ensureSession()
  return session?.access_token ? { Authorization: `Bearer ${session.access_token}` } : {}
}

export async function currentUser() {
  const session = await ensureSession()
  return session?.user ?? null
}

function returnUrl() {
  return window.location.origin + window.location.pathname
}

/** Attach an email to the anonymous user; the link in the email confirms it. */
export async function saveWithEmail(email) {
  const { error } = await supabase.auth.updateUser({ email }, { emailRedirectTo: returnUrl() })
  if (error) throw error
}

/** Attach a Google identity to the anonymous user (needs manual linking enabled). */
export async function saveWithGoogle() {
  const { error } = await supabase.auth.linkIdentity({ provider: 'google', options: { redirectTo: returnUrl() } })
  if (error) throw error
}

/** Sign in to an account saved earlier, e.g. on another device. */
export async function signInWithEmail(email) {
  const { error } = await supabase.auth.signInWithOtp({
    email,
    options: { shouldCreateUser: false, emailRedirectTo: returnUrl() },
  })
  if (error) throw error
}

export async function signInWithGoogle() {
  const { error } = await supabase.auth.signInWithOAuth({ provider: 'google', options: { redirectTo: returnUrl() } })
  if (error) throw error
}

export async function signOut() {
  await supabase.auth.signOut()
}

export function onAuthChange(callback) {
  if (!supabase) return () => {}
  const { data } = supabase.auth.onAuthStateChange((_event, session) => callback(session?.user ?? null))
  return () => data.subscription.unsubscribe()
}
