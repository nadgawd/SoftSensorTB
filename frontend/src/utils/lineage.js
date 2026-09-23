/**
 * Lay out a dataset's version tree as a git-style graph.
 *
 * Rows are chronological (row 0 = oldest). A version stays in its parent's
 * lane when it is that parent's first child; every later child forks into a
 * new lane. Lanes are reused once the branch occupying them has ended, so the
 * graph only gets as wide as the number of branches alive at the same time.
 *
 * @param {Array<{id: string, parent_version_id?: string|null, created_at?: string}>} versions
 * @returns {{
 *   nodes: Array<object & {row: number, lane: number, parentId: string|null, isFork: boolean}>,
 *   edges: Array<{from: string, to: string}>,
 *   laneCount: number,
 * }}
 */
export function layoutLineage(versions) {
  const ordered = [...(versions || [])].sort((a, b) => toTime(a) - toTime(b))
  const rowOf = new Map(ordered.map((v, i) => [v.id, i]))

  const children = new Map()
  for (const v of ordered) {
    const p = v.parent_version_id
    if (p && rowOf.has(p)) {
      if (!children.has(p)) children.set(p, [])
      children.get(p).push(v.id)
    }
  }

  // A branch is a chain of first-children. It occupies its lane from the
  // parent's row (where the connector leaves) through its last node.
  const branchOf = new Map()
  const branches = []
  const nodes = ordered.map((v, row) => {
    const parentId = v.parent_version_id && rowOf.has(v.parent_version_id) ? v.parent_version_id : null
    const continuesParent = parentId !== null && children.get(parentId)[0] === v.id
    let branch
    if (continuesParent) {
      branch = branchOf.get(parentId)
      branch.end = row
    } else {
      branch = { start: parentId !== null ? rowOf.get(parentId) : row, end: row, lane: -1 }
      branches.push(branch)
    }
    branchOf.set(v.id, branch)
    return { ...v, row, parentId, isFork: parentId !== null && !continuesParent }
  })

  const laneEnds = []
  for (const b of branches) {
    let lane = laneEnds.findIndex((end) => end < b.start)
    if (lane === -1) lane = laneEnds.length
    laneEnds[lane] = b.end
    b.lane = lane
  }

  for (const n of nodes) n.lane = branchOf.get(n.id).lane

  const edges = nodes.filter((n) => n.parentId).map((n) => ({ from: n.parentId, to: n.id }))
  return { nodes, edges, laneCount: Math.max(1, laneEnds.length) }
}

/**
 * Lay out the version tree with the root on the left and each generation one
 * column to the right. Leaves take consecutive rows in depth-first order
 * (siblings oldest first) and every parent is centred on its children.
 *
 * @returns {{
 *   nodes: Array<object & {depth: number, slot: number, parentId: string|null}>,
 *   edges: Array<{from: string, to: string}>,
 *   depthCount: number,
 *   slotCount: number,
 * }}
 */
export function layoutTree(versions) {
  const ordered = [...(versions || [])].sort((a, b) => toTime(a) - toTime(b))
  const known = new Set(ordered.map((v) => v.id))
  const parentOf = new Map(
    ordered.map((v) => [v.id, v.parent_version_id && known.has(v.parent_version_id) ? v.parent_version_id : null]),
  )

  const children = new Map()
  for (const v of ordered) {
    const p = parentOf.get(v.id)
    if (p) {
      if (!children.has(p)) children.set(p, [])
      children.get(p).push(v.id)
    }
  }

  const pos = new Map()
  let nextSlot = 0
  // Iterative post-order so a long chain of versions cannot overflow the stack.
  for (const root of ordered.filter((v) => !parentOf.get(v.id))) {
    const stack = [{ id: root.id, depth: 0, visited: false }]
    while (stack.length) {
      const frame = stack[stack.length - 1]
      const kids = children.get(frame.id) || []
      if (!frame.visited && kids.length) {
        frame.visited = true
        for (let i = kids.length - 1; i >= 0; i--) stack.push({ id: kids[i], depth: frame.depth + 1, visited: false })
        continue
      }
      stack.pop()
      const slot = kids.length
        ? (pos.get(kids[0]).slot + pos.get(kids[kids.length - 1]).slot) / 2
        : nextSlot++
      pos.set(frame.id, { depth: frame.depth, slot })
    }
  }

  const nodes = ordered.map((v) => ({ ...v, parentId: parentOf.get(v.id), ...pos.get(v.id) }))
  const edges = nodes.filter((n) => n.parentId).map((n) => ({ from: n.parentId, to: n.id }))
  const depthCount = 1 + Math.max(0, ...nodes.map((n) => n.depth))
  return { nodes, edges, depthCount, slotCount: Math.max(1, nextSlot) }
}

/** Ids of ``id`` and every version it was derived from. */
export function ancestorIds(nodes, id) {
  const parentOf = new Map(nodes.map((n) => [n.id, n.parentId]))
  const out = new Set()
  for (let cur = id; cur && !out.has(cur); cur = parentOf.get(cur)) out.add(cur)
  return out
}

function toTime(v) {
  if (!v?.created_at) return 0
  const iso = v.created_at.endsWith('Z') || /[+-]\d\d:\d\d$/.test(v.created_at) ? v.created_at : `${v.created_at}Z`
  return new Date(iso).getTime() || 0
}
