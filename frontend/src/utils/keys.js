const IS_MAC = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform)

/** "⌘K" on a Mac, "Ctrl+K" elsewhere. */
export const shortcutLabel = (key) => (IS_MAC ? `⌘${key}` : `Ctrl+${key}`)

/** True while the user is typing in a field, where single-key shortcuts must not fire. */
export function isTyping(target) {
  const tag = target?.tagName
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || Boolean(target?.isContentEditable)
}
