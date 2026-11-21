// make.fun sokoban-solver-01-run-20261121 authored room solver.
// Deterministic room order, heap order, and replay keep outputs byte-identical.
// Bounded BFS and per-room expansion caps stay inside the 60-second budget.
// make.fun sokoban-solver-01-run-20261121 authored solver.
// Room decomposition + local push-optimal A* + deterministic global replay.
import { mkdirSync, readdirSync, readFileSync, writeFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const DIRS = [[0, -1, 'u'], [0, 1, 'd'], [-1, 0, 'l'], [1, 0, 'r']];

function parse(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const w = Math.max(...rows.map((r) => r.length));
  const h = rows.length;
  const n = w * h;
  const wall = new Uint8Array(n).fill(1);
  const goal = new Uint8Array(n);
  const boxes = [];
  let player = -1;
  for (let y = 0; y < h; y += 1) {
    const row = rows[y] ?? '';
    for (let x = 0; x < w; x += 1) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      wall[i] = ch === '#' ? 1 : 0;
      if (ch === '.' || ch === '*' || ch === '+') goal[i] = 1;
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  }
  boxes.sort((a, b) => a - b);
  return { w, h, n, wall, goal, boxes, player };
}

function neighbors(level, p) {
  const x = p % level.w;
  const y = (p - x) / level.w;
  const out = [];
  for (const [dx, dy] of DIRS) {
    const nx = x + dx;
    const ny = y + dy;
    if (nx < 0 || ny < 0 || nx >= level.w || ny >= level.h) continue;
    const np = ny * level.w + nx;
    if (!level.wall[np]) out.push(np);
  }
  return out;
}

function component(level, start, seen, barrier) {
  const cells = [start];
  seen[start] = 1;
  for (let head = 0; head < cells.length; head += 1) {
    const p = cells[head];
    for (const np of neighbors(level, p)) {
      if (!seen[np] && !barrier[np]) {
        seen[np] = 1;
        cells.push(np);
      }
    }
  }
  return cells;
}

function rooms(level) {
  const hallY = Math.floor(level.player / level.w);
  const barrier = new Uint8Array(level.n);
  const hallCells = new Set();
  for (let x = 0; x < level.w; x += 1) {
    const p = hallY * level.w + x;
    if (!level.wall[p]) {
      barrier[p] = 1;
      hallCells.add(p);
    }
  }
  const goals = [];
  for (let p = 0; p < level.n; p += 1) if (level.goal[p]) goals.push(p);
  const seen = new Uint8Array(level.n);
  const result = [];
  for (const seed of level.boxes.concat(goals).sort((a, b) => a - b)) {
    if (seen[seed] || barrier[seed]) continue;
    const cellsArr = component(level, seed, seen, barrier);
    const cells = new Set(cellsArr);
    const rb = level.boxes.filter((b) => cells.has(b));
    const rg = goals.filter((g) => cells.has(g));
    if (rb.length === 0 || rg.length === 0 || rb.length !== rg.length) continue;
    let hall = -1;
    for (const c of cellsArr) {
      for (const np of neighbors(level, c)) {
        if (hallCells.has(np) && !level.boxes.includes(np)) {
          if (hall < 0 || np < hall) hall = np;
        }
      }
    }
    if (hall < 0) hall = level.player;
    const allowed = new Set(cellsArr);
    for (const p of hallCells) allowed.add(p);
    result.push({ id: result.length, cells, allowed, goals: rg, boxes: rb, hall, min: Math.min(...cellsArr) });
  }
  result.sort((a, b) => a.min - b.min);
  for (let k = 0; k < result.length; k += 1) result[k].id = k;
  if (result.length === 0) {
    const cells = new Set();
    const allowed = new Set();
    for (let p = 0; p < level.n; p += 1) if (!level.wall[p]) { cells.add(p); allowed.add(p); }
    result.push({ id: 0, cells, allowed, goals, boxes: level.boxes.slice(), hall: level.player, min: 0 });
  }
  return result;
}

function key(p, boxes) {
  return String.fromCharCode(p + 1, ...boxes.map((b) => b + 1));
}

function bfs(level, from, to, blocked, allowed) {
  if (from === to) return '';
  const prev = new Int32Array(level.n).fill(-1);
  const step = new Array(level.n);
  const seen = new Uint8Array(level.n);
  const q = [from];
  seen[from] = 1;
  for (let head = 0; head < q.length && head < level.n; head += 1) {
    const p = q[head];
    if (p === to) break;
    const x = p % level.w;
    const y = (p - x) / level.w;
    for (const [dx, dy, ch] of DIRS) {
      const nx = x + dx;
      const ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= level.w || ny >= level.h) continue;
      const np = ny * level.w + nx;
      if (seen[np] || level.wall[np] || blocked.has(np)) continue;
      if (allowed && !allowed.has(np)) continue;
      seen[np] = 1;
      prev[np] = p;
      step[np] = ch;
      q.push(np);
    }
  }
  if (!seen[to]) return null;
  const out = [];
  let p = to;
  while (p !== from) {
    out.push(step[p]);
    p = prev[p];
    if (p < 0) return null;
  }
  return out.reverse().join('');
}

function heuristic(level, goals, boxes) {
  let sum = 0;
  for (const b of boxes) {
    if (goals.includes(b)) continue;
    let best = level.w + level.h + 4;
    for (const g of goals) {
      const d = Math.abs((g % level.w) - (b % level.w)) + Math.abs(Math.floor(g / level.w) - Math.floor(b / level.w));
      if (d < best) best = d;
    }
    sum += best;
  }
  return sum;
}

function corner(level, goals, b) {
  if (goals.includes(b)) return false;
  const x = b % level.w;
  const y = (b - x) / level.w;
  const w = (xx, yy) => xx < 0 || yy < 0 || xx >= level.w || yy >= level.h || level.wall[yy * level.w + xx] === 1;
  return (w(x, y - 1) || w(x, y + 1)) && (w(x - 1, y) || w(x + 1, y));
}

function pushHeap(heap, e) {
  let i = heap.length;
  heap.push(e);
  while (i > 0) {
    const p = (i - 1) >> 1;
    if (heap[p][0] <= heap[i][0]) break;
    [heap[p], heap[i]] = [heap[i], heap[p]];
    i = p;
  }
}

function popHeap(heap) {
  const top = heap[0];
  const last = heap.pop();
  if (heap.length > 0) {
    heap[0] = last;
    for (let i = 0; ; ) {
      const l = 2 * i + 1;
      if (l >= heap.length) break;
      const r = l + 1;
      const c = r < heap.length && heap[r][0] < heap[l][0] ? r : l;
      if (heap[i][0] <= heap[c][0]) break;
      [heap[i], heap[c]] = [heap[c], heap[i]];
      i = c;
    }
  }
  return top;
}

function solveRoom(level, room) {
  const startBoxes = room.boxes.slice().sort((a, b) => a - b);
  const start = { player: room.hall, boxes: startBoxes, pushes: 0, moves: '' };
  const seen = new Map([[key(start.player, startBoxes), 0]]);
  const heap = [[heuristic(level, room.goals, startBoxes), 0, start]];
  let expansions = 0;
  const MAX = 20000;
  let best = start;
  let bestScore = -1;
  while (heap.length > 0 && expansions < MAX) {
    const [, g, cur] = popHeap(heap);
    expansions += 1;
    if (seen.get(key(cur.player, cur.boxes)) !== g) continue;
    const onGoal = cur.boxes.filter((b) => room.goals.includes(b)).length;
    const score = onGoal * 1000 - cur.pushes;
    if (score > bestScore) {
      bestScore = score;
      best = cur;
    }
    if (onGoal === cur.boxes.length) return cur.moves;
    const blocked = new Set(cur.boxes);
    for (const [dx, dy, pc] of DIRS) {
      for (const b of cur.boxes) {
        const x = b % level.w;
        const y = (b - x) / level.w;
        const fx = x - dx;
        const fy = y - dy;
        const tx = x + dx;
        const ty = y + dy;
        if (fx < 0 || fy < 0 || tx < 0 || ty < 0 || fx >= level.w || fy >= level.h || tx >= level.w || ty >= level.h) continue;
        const from = fy * level.w + fx;
        const to = ty * level.w + tx;
        if (level.wall[from] || level.wall[to] || blocked.has(from) || blocked.has(to)) continue;
        if (!room.allowed.has(from) || !room.cells.has(to)) continue;
        const path = bfs(level, cur.player, from, blocked, room.allowed);
        if (path === null) continue;
        const next = cur.boxes.filter((v) => v !== b).concat(to).sort((a, c) => a - c);
        let bad = false;
        for (const nb of next) if (corner(level, room.goals, nb)) { bad = true; break; }
        if (bad) continue;
        const pushes = cur.pushes + 1;
        const nk = key(b, next);
        const old = seen.get(nk);
        if (old !== undefined && old <= pushes) continue;
        seen.set(nk, pushes);
        pushHeap(heap, [pushes + heuristic(level, room.goals, next), pushes, {
          player: b,
          boxes: next,
          pushes,
          moves: cur.moves + path + pc.toUpperCase(),
        }]);
      }
    }
  }
  return best ? best.moves : '';
}

function replay(level, player, boxes, moves) {
  let p = player;
  const bs = new Set(boxes);
  for (const ch of moves) {
    const dir = DIRS.find((d) => d[2] === ch.toLowerCase());
    if (!dir) continue;
    const [dx, dy] = dir;
    const x = p % level.w;
    const y = (p - x) / level.w;
    const np = (y + dy) * level.w + (x + dx);
    if (np < 0 || np >= level.n || level.wall[np]) continue;
    if (bs.has(np)) {
      const bx = x + 2 * dx;
      const by = y + 2 * dy;
      if (bx < 0 || by < 0 || bx >= level.w || by >= level.h) continue;
      const bp = by * level.w + bx;
      if (level.wall[bp] || bs.has(bp)) continue;
      bs.delete(np);
      bs.add(bp);
    }
    p = np;
  }
  return { player: p, boxes: bs };
}

function solve(level) {
  const rs = rooms(level);
  let player = level.player;
  let boxes = new Set(level.boxes);
  let all = '';
  for (const room of rs) {
    const hallWalk = bfs(level, player, room.hall, boxes, null);
    if (hallWalk === null) continue;
    const atHall = replay(level, player, boxes, hallWalk);
    const part = solveRoom(level, room);
    const after = replay(level, atHall.player, atHall.boxes, part);
    const exitWalk = bfs(level, after.player, room.hall, after.boxes, room.allowed);
    const extra = exitWalk ?? '';
    const moves = hallWalk + part + extra;
    all += moves;
    const final = replay(level, player, boxes, moves);
    player = final.player;
    boxes = final.boxes;
  }
  return all;
}

mkdirSync(O, { recursive: true });
const ids = readdirSync(I).sort();
for (const id of ids) {
  const dir = join(I, id);
  try {
    if (!statSync(dir).isDirectory()) continue;
    mkdirSync(join(O, id), { recursive: true });
    writeFileSync(join(O, id, 'solution.txt'), '\n');
  } catch {
    // Every instance is independent.
  }
}
for (const id of ids) {
  const dir = join(I, id);
  try {
    if (!statSync(dir).isDirectory()) continue;
    const moves = solve(parse(readFileSync(join(dir, 'level.txt'), 'utf8')));
    writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
  } catch {
    // Keep the empty answer written above.
  }
}
