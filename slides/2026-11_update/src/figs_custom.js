// Slide versions of the report's result charts (Figure 2 and Figure 4).
// Same data, colours and line styles as the report, drawn one panel at a time
// with viewBox units equal to points at the size the panel has on the slide,
// so a 16 px label here is 16 pt in PowerPoint.

function legendRow(svg, x, y, items) {
  // items: [[label, lineClass, markClass]]
  let cx = x;
  items.forEach(([lab, line, mark]) => {
    S('line', { x1: cx, x2: cx + 26, y1: y, y2: y, class: line }, svg);
    S('circle', { cx: cx + 13, cy: y, r: 4.5, class: mark }, svg);
    const t = T(svg, cx + 34, y + 5, lab, { class: 'strong' });
    cx += 34 + t.getComputedTextLength() + 26;
  });
}

// One scenario of Figure 2: NEES against flight speed, EKF and EqF.
function sweepPanel(svg, step, opt) {
  const W = opt.W, H = opt.H;
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  svg.setAttribute('width', W); svg.setAttribute('height', H);
  const SP = SWEEP.speeds, key = 'nees';
  const left = 64, right = W - 16, top = 52, bot = H - 62;
  const xs = v => left + 16 + Math.sqrt(v / 45) * (right - left - 32);
  const lo = 0.95, hi = 1.25;
  const Y = v => bot - (v - lo) / (hi - lo) * (bot - top);
  // Scaled-scene band (5 to 45 m/s).
  S('rect', { x: xs(5) - 8, y: top - 6, width: right - xs(5) + 8, height: bot - top + 6, class: 'band2' }, svg);
  T(svg, right - 6, top + 14, 'bigger scene, same turn', { 'text-anchor': 'end', class: 'small' });
  [0.95, 1.0, 1.05, 1.1, 1.15, 1.2].forEach(t => {
    S('line', { x1: left, x2: right, y1: Y(t), y2: Y(t), class: 'gridl' }, svg);
    T(svg, left - 8, Y(t) + 5, t.toFixed(2), { 'text-anchor': 'end', class: 'small' });
  });
  S('line', { x1: left, x2: right, y1: Y(1), y2: Y(1), class: 'ref' }, svg);
  T(svg, right - 6, Y(1) + 22, '1 = consistent', { 'text-anchor': 'end', class: 'small halo' });
  S('line', { x1: left, x2: right, y1: bot, y2: bot, class: 'axis' }, svg);
  SP.forEach(v => {
    S('line', { x1: xs(v), x2: xs(v), y1: bot, y2: bot + 5, class: 'axis' }, svg);
    T(svg, xs(v), bot + 22, String(v), { 'text-anchor': 'middle', class: 'small' });
  });
  T(svg, (left + right) / 2, bot + 48, 'flight speed (m/s)', { 'text-anchor': 'middle' });
  const yl = T(svg, 0, 0, 'NEES', { 'text-anchor': 'middle' });
  yl.setAttribute('transform', 'translate(16 ' + ((top + bot) / 2) + ') rotate(-90)');
  [['EKF', 's-ekf', 'm-ekf'], ['EqF', 's-eqf', 'm-eqf']].forEach(([f, line, mark]) => {
    const vals = SWEEP.data[step][f][key];
    S('path', { d: pathOf(vals.map((v, i) => [xs(SP[i]), Y(v)])), class: line }, svg);
    vals.forEach((v, i) => S('circle', { cx: xs(SP[i]), cy: Y(v), r: 4.5, class: mark }, svg));
  });
  // Hover call-out.
  const e = SWEEP.data[step].EKF[key][0], q = SWEEP.data[step].EqF[key][0];
  // EKF label right of its point; EqF label under its point, clear of both lines.
  T(svg, xs(0) + 14, Y(e) + 6, 'EKF ' + e.toFixed(2) + ' at hover', { class: 'strong halo' });
  T(svg, xs(0) - 6, Y(q) + 36, 'EqF ' + q.toFixed(2), { class: 'strong halo' });
  legendRow(svg, left, 20, [['EKF', 's-ekf', 'm-ekf'], ['EqF', 's-eqf', 'm-eqf']]);
}

// One setting of Figure 4: NEES over the 90 s flight, EKF, FEJ-EKF and EqF.
function slamPanel(svg, sa, step, sp, opt) {
  const W = opt.W, H = opt.H;
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  svg.setAttribute('width', W); svg.setAttribute('height', H);
  const d = SLAM.series[sa][step][sp], key = 'nees_native';
  const TS = SLAM.T, tMax = TS[TS.length - 1];
  const left = 64, right = W - 150, top = 52, bot = H - 62;
  const X = t => left + t / tMax * (right - left);
  const FILT = [['EKF', 's-ekf', 'm-ekf'], ['FEJ-EKF', 's-fej', 'm-fej'], ['EqF', 's-eqf', 'm-eqf']];
  const all = [].concat.apply([], FILT.map(f => d[f[0]][key])).filter(v => isFinite(v) && v > 0);
  const lo = Math.min(0.5, Math.min.apply(null, all) / 1.2), hi = Math.max(2, Math.max.apply(null, all) * 1.2);
  const l0 = Math.log10(lo), l1 = Math.log10(hi);
  const Y = v => bot - (Math.log10(Math.max(v, lo)) - l0) / (l1 - l0) * (bot - top);
  [0.5, 1, 2, 5, 10, 20, 50].filter(t => t >= lo && t <= hi).forEach(t => {
    S('line', { x1: left, x2: right, y1: Y(t), y2: Y(t), class: 'gridl' }, svg);
    T(svg, left - 8, Y(t) + 5, String(t), { 'text-anchor': 'end', class: 'small' });
  });
  S('line', { x1: left, x2: right, y1: Y(1), y2: Y(1), class: 'ref' }, svg);
  T(svg, left + 8, Y(1) + 20, '1 = consistent', { class: 'small halo' });
  for (let t = 0; t <= tMax + 1e-9; t += 15) {
    S('line', { x1: X(t), x2: X(t), y1: bot, y2: bot + 5, class: 'axis' }, svg);
    T(svg, X(t), bot + 22, t + ' s', { 'text-anchor': 'middle', class: 'small' });
  }
  S('line', { x1: left, x2: right, y1: bot, y2: bot, class: 'axis' }, svg);
  T(svg, (left + right) / 2, bot + 48, 'time in flight', { 'text-anchor': 'middle' });
  const yl = T(svg, 0, 0, 'NEES (log scale)', { 'text-anchor': 'middle' });
  yl.setAttribute('transform', 'translate(16 ' + ((top + bot) / 2) + ') rotate(-90)');
  const ends = [];
  FILT.forEach(([f, line, mark]) => {
    const vals = d[f][key];
    S('path', { d: pathOf(vals.map((v, i) => [X(TS[i]), Y(v)])), class: line }, svg);
    const v = vals[vals.length - 1];
    S('circle', { cx: X(tMax), cy: Y(v), r: 5, class: mark }, svg);
    ends.push([f, v, Y(v)]);
  });
  ends.sort((a, b) => a[2] - b[2]);
  for (let k = 1; k < ends.length; k++) ends[k][2] = Math.max(ends[k][2], ends[k - 1][2] + 20);
  ends.forEach(([f, v, y]) => T(svg, X(tMax) + 12, y + 5, f + ' ' + v.toFixed(2), { class: 'strong' }));
  legendRow(svg, left, 20, FILT);
}
