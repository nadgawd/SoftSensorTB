import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'

const REMARK_PLUGINS = [remarkGfm, remarkMath]
const REHYPE_PLUGINS = [[rehypeKatex, { throwOnError: false, strict: 'ignore' }]]

// Code spans and fences are left alone so LaTeX-looking code is not rendered.
const CODE = /(```[\s\S]*?(?:```|$)|`[^`\n]*`)/

// remark-math only understands $…$ and $$…$$, but Qwen also writes \(…\) and \[…\].
// A $$…$$ alone on one line is meant as a display equation, yet remark-math
// renders it inline unless the delimiters sit on their own lines.
function normalizeMath(text) {
  return text
    .split(CODE)
    .map((part, i) => (i % 2 === 1 ? part : part
      .replace(/\\\[([\s\S]+?)\\\]/g, (_, tex) => `\n$$\n${tex.trim()}\n$$\n`)
      .replace(/\\\(([\s\S]+?)\\\)/g, (_, tex) => `$${tex.trim()}$`)
      .replace(/^[ \t]*\$\$([^\n$]+)\$\$[ \t]*$/gm, (_, tex) => `$$\n${tex.trim()}\n$$`)))
    .join('')
}

export default function ChatMarkdown({ children }) {
  return (
    <ReactMarkdown remarkPlugins={REMARK_PLUGINS} rehypePlugins={REHYPE_PLUGINS}>
      {normalizeMath(children ?? '')}
    </ReactMarkdown>
  )
}
