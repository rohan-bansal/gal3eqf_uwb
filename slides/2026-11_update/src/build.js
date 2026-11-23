// Build gal3uwb_update.pptx in the current directory (build/), from build/img.
// Layout: LAYOUT_WIDE (13.333 x 7.5 in), Arial, plain white. Short text, simple words,
// a picture slide before each scenario, all the math in the appendix.
const fs = require('fs');
const path = require('path');
const pptxgen = require('pptxgenjs');
const { applyTheme } = require(path.join(process.env.HOME,
  '.claude/skills/synced/a1e1ea47-6147-4b0c-8258-d89a38b50227_3880d501-2c80-4180-8ffa-2ee964d7d248/pptx/scripts/apply_theme.js'));

const IMG = path.resolve('img');
const MAN = JSON.parse(fs.readFileSync(path.join(IMG, 'manifest.json'), 'utf8'));
const REPORT = 'https://claude.ai/artifact/99n9r1YAuPoRj5mXPhsE8R';

const THEME = {
  name: 'Gal3 UWB', headFontFace: 'Arial', bodyFontFace: 'Arial',
  colors: {
    dk1: '1D2229', lt1: 'FFFFFF', dk2: '1F4E79', lt2: 'F3F5F8',
    accent1: '2A78D6',  // EqF blue (report)
    accent2: 'B8481A',  // EKF orange, text shade (report)
    accent3: '0F7A54',  // FEJ green, text shade (report)
    accent4: 'E6F0FB',  // EqF soft fill
    accent5: '8F6100',  // tag clock amber (report)
    accent6: '6B7280',  // muted text
    hlink: '2A78D6', folHlink: '5B4A9E',
  },
};

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';
pres.theme = { headFontFace: 'Arial', bodyFontFace: 'Arial' };
pres.title = 'Galilean EqF for UWB-IMU localization: update';
pres.author = 'Rohan Bansal';
const C = pres.SchemeColor;
const INK = C.text1, NAVY = C.text2, PANEL = C.background2, EQF = C.accent1, EKF = C.accent2, FEJ = C.accent3,
  EQF_SOFT = C.accent4, CLOCK = C.accent5, MUTED = C.accent6, WHITE = C.background1;
const X0 = 0.6, CW = 12.133, XR = X0 + CW;

// ---------- layouts ----------
pres.defineSlideMaster({
  title: 'TITLE', background: { color: WHITE },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 0.8, y: 1.9, w: 11.7, h: 1.8, fontSize: 38, bold: true, color: NAVY, valign: 'bottom', align: 'left', margin: 0 }, text: '' } },
    { placeholder: { options: { name: 'sub', type: 'body', x: 0.8, y: 4.0, w: 11.7, h: 0.5, fontSize: 22, color: INK, valign: 'top', margin: 0 }, text: '' } },
    { placeholder: { options: { name: 'author', type: 'body', x: 0.8, y: 4.55, w: 11.7, h: 0.45, fontSize: 20, color: MUTED, valign: 'top', margin: 0 }, text: '' } },
  ],
});
pres.defineSlideMaster({
  title: 'CONTENT', background: { color: WHITE },
  objects: [
    { placeholder: { options: { name: 'kicker', type: 'body', x: X0, y: 0.3, w: CW, h: 0.32, fontSize: 14, bold: true, color: EQF, valign: 'top', margin: 0 }, text: '' } },
    { placeholder: { options: { name: 'title', type: 'title', x: X0, y: 0.66, w: CW, h: 0.96, fontSize: 28, bold: true, color: NAVY, valign: 'top', align: 'left', margin: 0 }, text: '' } },
  ],
  slideNumber: { x: 12.23, y: 7.0, w: 0.5, h: 0.3, fontSize: 12, color: MUTED, align: 'right' },
});

// ---------- helpers ----------
// Markup: **bold**, _{sub}, ^{sup}.
function rich(s, base) {
  const out = [];
  let bold = false, last = 0, m;
  const re = /(\*\*|_\{[^}]*\}|\^\{[^}]*\})/g;
  const push = (t, extra) => { if (t) out.push({ text: t, options: Object.assign({}, base || {}, bold ? { bold: true } : {}, extra || {}) }); };
  while ((m = re.exec(s))) {
    push(s.slice(last, m.index));
    const tok = m[0];
    if (tok === '**') bold = !bold;
    else if (tok[0] === '_') push(tok.slice(2, -1), { subscript: true });
    else push(tok.slice(2, -1), { superscript: true });
    last = re.lastIndex;
  }
  push(s.slice(last));
  return out;
}
function paras(items, o) {
  o = o || {};
  const runs = [];
  items.forEach((it, k) => {
    const item = typeof it === 'string' ? { t: it } : it;
    const base = {};
    if (item.color) base.color = item.color;
    if (item.size) base.fontSize = item.size;
    const r = rich(item.t, base);
    if (!r.length) r.push({ text: ' ', options: {} });
    const bullet = item.bullet !== undefined ? item.bullet : o.bullet;
    r[0].options.bullet = bullet ? { indent: o.indent || 20 } : false;
    r[0].options.paraSpaceAfter = item.gap !== undefined ? item.gap : (o.gap !== undefined ? o.gap : 8);
    if (k < items.length - 1) r[r.length - 1].options.breakLine = true;
    runs.push(...r);
  });
  return runs;
}
function text(slide, items, o) {
  const runs = typeof items === 'string' ? rich(items) : paras(items, o);
  slide.addText(runs, Object.assign({ isTextBox: true, margin: 0, valign: 'top', color: INK, fontSize: 20 }, o, { bullet: undefined, gap: undefined, indent: undefined }));
}
function head(slide, s, x, y, w, color, size) {
  slide.addText(rich(s), { isTextBox: true, x, y, w, h: 0.4, margin: 0, fontSize: size || 20, bold: true, color: color || NAVY, valign: 'top' });
}
function eq(slide, id, x, y, scale) {
  const m = MAN['eq_' + id];
  if (!m) throw new Error('missing equation ' + id);
  const s = scale || 1, w = m.w / 96 * s, h = m.h / 96 * s;
  slide.addImage({ path: path.join(IMG, 'eq_' + id + '.png'), x, y, w, h, altText: 'Equation ' + id });
  return { w, h };
}
function fig(slide, id, x, y, w, alt) {
  const m = MAN[id];
  const h = w * m.h / m.w;
  slide.addImage({ path: path.join(IMG, id + '.png'), x, y, w, h, altText: alt });
  return { w, h };
}
function cite(slide, s) {
  slide.addText(rich(s), { isTextBox: true, x: X0, y: 7.0, w: 11.4, h: 0.3, margin: 0, fontSize: 12, color: MUTED, valign: 'top' });
}
function box(slide, x, y, w, h, fill, line) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.06, fill: { color: fill || PANEL }, line: line ? { color: line, width: 1.25 } : { color: fill || PANEL, width: 0 } });
}
function line(slide, x1, y1, x2, y2, o) {
  o = o || {};
  const flipH = x2 < x1, flipV = y2 < y1;
  slide.addShape(pres.shapes.LINE, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), flipH, flipV,
    line: Object.assign({ color: o.color || INK, width: o.width || 1.5 }, o.dash ? { dashType: o.dash } : {},
      o.end ? { endArrowType: 'triangle' } : {}, o.begin ? { beginArrowType: 'triangle' } : {}),
  });
}
function content(kicker, title, section) {
  const s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: section });
  s.addText(kicker, { placeholder: 'kicker' });
  s.addText(title, { placeholder: 'title' });
  return s;
}
// A scenario picture slide: the report's Figure 1 panel, large, with one caption line.
function pictureSlide(kicker, title, id, caption, alt, notes) {
  const s = content(kicker, title, 'Scenarios');
  const w = 9.4, x = (13.333 - w) / 2;
  const d = fig(s, id, x, 1.75, w, alt);
  text(s, caption, { x: x, y: 1.8 + d.h, w: w, h: 0.4, fontSize: 18, color: MUTED });
  s.addNotes(notes);
  return s;
}
// Results slide: chart left, short takeaways right.
function resultSlide(kicker, title, figId, alt, points, src, notes) {
  const s = content(kicker, title, 'Scenarios');
  fig(s, figId, X0, 1.8, 7.3, alt);
  head(s, 'Takeaways', 8.3, 1.9, 4.43);
  text(s, points, { x: 8.3, y: 2.45, w: 4.43, h: 4.4, bullet: true, gap: 14 });
  cite(s, src);
  s.addNotes(notes);
  return s;
}

// ===================================================================
// Title
pres.addSection({ title: 'Title' });
{
  const s = pres.addSlide({ masterName: 'TITLE', sectionTitle: 'Title' });
  s.addText('A Galilean equivariant filter for UWB–IMU drone localization', { placeholder: 'title' });
  s.addText('Update meeting · November 2026', { placeholder: 'sub' });
  s.addText('Rohan Bansal', { placeholder: 'author' });
  s.addText([{ text: 'Full report: ', options: { color: MUTED } }, { text: REPORT.replace('https://', ''), options: { hyperlink: { url: REPORT }, color: EQF } }],
    { isTextBox: true, x: 0.8, y: 6.55, w: 11.7, h: 0.35, margin: 0, fontSize: 14 });
  s.addNotes('Update on the Galilean filter for drone localization with UWB radios and an IMU. First a refresher on the problem, then the four test scenarios one by one, then an overall summary.');
}

// ===================================================================
pres.addSection({ title: 'Background' });
// B1. Setup picture
{
  const s = content('Background', 'The drone has an IMU and a radio tag; fixed anchors listen', 'Background');
  const d = fig(s, 'pic_room1', X0 + 0.2, 1.8, 4.75, 'Top view: a tag in the middle, eight anchors around the room, radio lines from the tag to each anchor');
  text(s, [
    '**IMU:** measures acceleration and turning rate, 200 times a second',
    '**UWB tag:** sends a radio packet 20 times a second',
    '**Anchors:** 8 radios at fixed spots; each one records when a packet arrives',
    '**Goal:** track where the drone is',
  ], { x: 6.0, y: 2.3, w: 6.7, h: 4.2, gap: 18 });
  text(s, 'Top view (report Figure 1)', { x: X0 + 0.2, y: 1.85 + d.h, w: 4.75, h: 0.35, fontSize: 14, color: MUTED });
  s.addNotes('The drone has an IMU, which measures acceleration and turning rate, and a UWB tag, which is a small radio. The tag sends a packet 20 times a second. Eight anchors at fixed, known spots record when each packet arrives. From these arrival times and the IMU we want to track the drone.');
}
// B2. Timing diagram
{
  const s = content('Background', 'Each arrival time gives the distance to an anchor, plus the tag\'s clock error', 'Background');
  const yT = 2.6, yA = 4.15, xs = 3.0, xe = XR - 0.1, xT = 3.7, xD = 5.7, x1 = 9.9;
  text(s, '**Tag\'s clock**', { x: X0, y: yT - 0.2, w: 2.3, h: 0.4, fontSize: 18, color: CLOCK });
  text(s, '**Anchors\' clock**', { x: X0, y: yA - 0.2, w: 2.3, h: 0.4, fontSize: 18 });
  line(s, xs, yT, xe, yT, { color: CLOCK, width: 1.5, end: true });
  line(s, xs, yA, xe, yA, { color: INK, width: 1.5, end: true });
  text(s, 'time', { x: xe - 0.6, y: yT + 0.1, w: 0.6, h: 0.3, fontSize: 14, color: MUTED, align: 'right' });
  // send event
  line(s, xT, yT - 0.18, xT, yT + 0.18, { color: CLOCK, width: 3 });
  text(s, 'tag sends a packet and writes its own time **T** in it', { x: xT - 0.3, y: yT - 0.62, w: 6.5, h: 0.4, fontSize: 18 });
  line(s, xT, yT, xD, yA, { color: MUTED, width: 1.25, dash: 'dash' });
  text(s, 'same moment', { x: (xT + xD) / 2 + 0.18, y: (yT + yA) / 2 - 0.25, w: 2.0, h: 0.35, fontSize: 15, color: MUTED });
  // reference and arrival
  line(s, xT, yT + 0.2, xT, yA + 1.2, { color: MUTED, width: 0.75, dash: 'sysDot' });
  line(s, xD, yA - 0.12, xD, yA + 0.12, { color: MUTED, width: 1.5 });
  line(s, x1, yA - 0.18, x1, yA + 0.18, { color: INK, width: 3 });
  line(s, x1, yA + 0.2, x1, yA + 1.2, { color: MUTED, width: 0.75, dash: 'sysDot' });
  text(s, 'anchor 1 receives it at time **t_{1}**', { x: x1 - 2.2, y: yA - 0.62, w: 4.4, h: 0.4, fontSize: 18, align: 'center' });
  // the two parts of the measurement
  s.addShape(pres.shapes.RECTANGLE, { x: xT, y: yA + 0.3, w: xD - xT, h: 0.2, fill: { color: CLOCK }, line: { color: CLOCK, width: 0 } });
  s.addShape(pres.shapes.RECTANGLE, { x: xD, y: yA + 0.3, w: x1 - xD, h: 0.2, fill: { color: EQF }, line: { color: EQF, width: 0 } });
  text(s, 'clock error **δ**', { x: xT, y: yA + 0.55, w: xD - xT, h: 0.35, fontSize: 16, color: CLOCK, align: 'center' });
  text(s, 'travel time = distance ÷ speed of light', { x: xD, y: yA + 0.55, w: x1 - xD, h: 0.35, fontSize: 16, color: EQF, align: 'center' });
  line(s, xT, yA + 1.15, x1, yA + 1.15, { color: INK, width: 1.25, end: true, begin: true });
  text(s, 'what we measure: **t_{1} − T**', { x: xT, y: yA + 1.25, w: x1 - xT, h: 0.4, fontSize: 18, align: 'center' });
  box(s, X0, 6.2, CW, 0.65);
  text(s, '**t_{1} − T  =  distance to anchor 1 ÷ c  +  δ**', { x: X0 + 0.25, y: 6.32, w: CW - 0.5, h: 0.4, fontSize: 20 });
  s.addNotes('The tag writes its own send time T into each packet. The anchor records when the packet arrives, t_1, on the anchors\' clock. The two clocks disagree by an unknown amount, delta. So the difference t_1 minus T is the travel time, which is distance over the speed of light, plus delta. One packet gives one such number for every anchor that hears it.');
}
// B3. Two ways
{
  const s = content('Background', 'We keep the clock error and estimate it, instead of cancelling it', 'Background');
  const bw = 5.95, bh = 3.35, y = 1.85, x2 = X0 + bw + 0.233;
  box(s, X0, y, bw, bh, PANEL);
  head(s, 'Usual way (TDoA)', X0 + 0.3, y + 0.2, bw - 0.6, MUTED, 22);
  text(s, 'Subtract the times of two anchors:', { x: X0 + 0.3, y: y + 0.75, w: bw - 0.6, h: 0.4 });
  eq(s, 'tdoa', X0 + 0.3, y + 1.3);
  text(s, 'δ cancels. The clock is never estimated.', { x: X0 + 0.3, y: y + 2.55, w: bw - 0.6, h: 0.5 });
  box(s, x2, y, bw, bh, EQF_SOFT, EQF);
  head(s, 'Our way (raw times)', x2 + 0.3, y + 0.2, bw - 0.6, EQF, 22);
  text(s, 'Keep the time of each anchor:', { x: x2 + 0.3, y: y + 0.75, w: bw - 0.6, h: 0.4 });
  eq(s, 'raw', x2 + 0.3, y + 1.3);
  text(s, 'δ stays. The filter estimates it, along with the position.', { x: x2 + 0.3, y: y + 2.55, w: bw - 0.6, h: 0.7 });
  text(s, '**Why:** with the clock inside the filter, motion and clock can be handled together. That is what the Galilean filter is built for (next slide).',
    { x: X0, y: 5.55, w: CW, h: 0.9 });
  s.addNotes('There are two ways to deal with delta. The usual way, time difference of arrival, subtracts the times at two anchors, so delta cancels and the filter never sees the clock. We keep every raw time instead. Then delta stays in the measurement and the filter estimates it together with position. Keeping the clock in the filter is what lets us use the Galilean group, which treats motion and time together.');
}
// B4. State and the group
{
  const s = content('Background', 'The filter tracks motion and clock together, and stores them as one matrix', 'Background');
  head(s, 'What the filter estimates', X0, 1.85, 5.8);
  text(s, [
    'orientation **R**, velocity **v**, position **p**',
    'tag clock error **δ** and its drift **b_{ρ}**',
    'small IMU sensor errors (biases)',
  ], { x: X0, y: 2.45, w: 5.8, h: 2.6, bullet: true, gap: 14 });
  head(s, 'Stored as one matrix', 6.9, 1.85, 5.8);
  const f = eq(s, 'Fmat', 6.9, 2.4);
  text(s, 'R, v, p and δ fit in one 5 × 5 matrix: the Galilean group Gal(3).', { x: 6.9, y: 2.55 + f.h, w: 5.83, h: 0.8, fontSize: 18 });
  box(s, X0, 5.35, CW, 1.3, PANEL);
  text(s, [
    '**EqF** (Delama et al. 2026): measures its error on this group.',
    '**EKF** (standard filter): subtracts each number on its own.',
  ], { x: X0 + 0.25, y: 5.5, w: CW - 0.5, h: 1.05, gap: 8 });
  cite(s, 'Delama et al., Galilean State Estimation for INS with Unknown Time Delay, RSS 2026 (arXiv:2605.13266)');
  s.addNotes('The filter estimates orientation, velocity, position, the tag clock error and its drift, and the small biases of the IMU. Orientation, velocity, position and the clock error fit into one 5 by 5 matrix, an element of the Galilean group. The equivariant filter of Delama et al. defines its error on this group. The standard EKF subtracts each quantity separately. That is the only difference between the two filters.');
}
// B5. Questions
{
  const s = content('Background', 'We ask three questions', 'Background');
  box(s, X0, 1.8, CW, 2.45, PANEL);
  text(s, [
    '**Q1.** Does linking clock and motion in one group help with UWB?',
    '**Q2.** Is the EqF better than the EKF at knowing its own error?',
    '**Q3.** When anchor positions are only roughly known, does the EqF avoid becoming too confident?',
  ], { x: X0 + 0.3, y: 2.0, w: CW - 0.6, h: 2.1, fontSize: 22, gap: 14 });
  head(s, 'How we score a filter: NEES', X0, 4.6, CW);
  text(s, [
    '**1:** the filter\'s uncertainty matches its real error (we call this **consistent**)',
    '**above 1:** the filter is too confident',
  ], { x: X0, y: 5.15, w: CW, h: 1.0, bullet: true, gap: 8 });
  text(s, 'Tests: simulated flights from 0 to 45 m/s, and real IMU data from the MILUV dataset.', { x: X0, y: 6.3, w: CW, h: 0.4, fontSize: 18, color: MUTED });
  cite(s, 'MILUV: Shalaby et al., 2025 (arXiv:2504.14376)');
  s.addNotes('Three questions. Q1: the Galilean group links the clock to motion; does that matter for UWB? Q2: is the EqF better at knowing how wrong it is? Q3: when the anchor positions are only roughly known, EKF-based mapping is known to become too confident; does the EqF avoid that? We score this with NEES: 1 means the filter\'s uncertainty matches its real error, above 1 means it is too confident.');
}

// ===================================================================
pres.addSection({ title: 'Scenarios' });
// S1
pictureSlide('Scenario S1', 'S1: the anchors share one clock, and the tag clock is off by a fixed amount', 'pic_sc1',
  'Left: top view. Right: clock errors over 30 s. Only one unknown clock number, δ.',
  'S1: anchors share one clock; the tag clock has a constant offset',
  'The simplest case. All anchors share one clock. The tag clock is off by a fixed amount, so there is only one clock number to estimate.');
{
  const s = content('Scenario S1', 'The filter predicts with the IMU and corrects with each packet', 'Scenarios');
  const y = 1.95, bw = 4.9, bh = 1.65, x2 = XR - bw - 0.4;
  box(s, X0 + 0.4, y, bw, bh, PANEL);
  text(s, [{ t: '**Predict**', size: 22, gap: 6 }, 'IMU, 200 times a second: move the estimate forward'], { x: X0 + 0.65, y: y + 0.2, w: bw - 0.5, h: 1.3 });
  box(s, x2, y, bw, bh, EQF_SOFT);
  text(s, [{ t: '**Correct**', size: 22, gap: 6 }, 'each UWB packet: compare the measured and predicted times'], { x: x2 + 0.25, y: y + 0.2, w: bw - 0.5, h: 1.3 });
  line(s, X0 + 0.4 + bw + 0.1, y + 0.55, x2 - 0.1, y + 0.55, { color: INK, width: 2, end: true });
  line(s, x2 - 0.1, y + 1.1, X0 + 0.4 + bw + 0.1, y + 1.1, { color: INK, width: 2, end: true });
  box(s, X0, 4.15, CW, 1.85, WHITE, EQF);
  text(s, [
    '**Q1 is tested here.** The EqF\'s correct step has extra terms that link the clock to motion.',
    'For UWB they are tiny: the clock is known to 0.1 ns, and in 0.1 ns a drone at 45 m/s moves 4.5 nm.',
  ], { x: X0 + 0.3, y: 4.35, w: CW - 0.6, h: 1.5, gap: 12 });
  text(s, 'EqF: 20 error states. EKF: 17. Equations in Appendix A.', { x: X0, y: 6.3, w: CW, h: 0.4, fontSize: 18, color: MUTED });
  s.addNotes('Both filters work the same way. The IMU predicts the motion forward 200 times a second; each UWB packet corrects the estimate. The EqF\'s correction contains terms that link the clock to motion; that is the Galilean part, and Q1 asks whether it matters. For UWB the clock is known to about a tenth of a nanosecond, and in that time even a 45 m/s drone moves only 4.5 nanometres, so we expect no effect.');
}
resultSlide('Scenario S1 · Results', 'S1: same accuracy as the EKF; better uncertainty only when hovering', 'sweep_s1',
  'NEES against flight speed in S1 for EKF and EqF', [
    'Position error: about 3 cm for both',
    'Hovering: EqF 1.06, EKF 1.16',
    'From 5 m/s on: the same',
    '**Q1:** clock error is the same at 0 and 45 m/s, so the clock–motion link does not matter',
  ], 'Simulation, 50 flights of 30 s per speed (report Figure 2). Grey: scene made bigger so the turn stays the same.',
  'NEES against speed. When hovering the EqF is closer to 1, 1.06 against 1.16. From 5 m/s on the two are the same. Position error is about 3 cm for both. And the clock error does not change between hover and 45 m/s, so the answer to Q1 is no.');
// S2
pictureSlide('Scenario S2', 'S2: the tag clock now drifts', 'pic_sc2',
  'The tag\'s crystal runs a little fast or slow (up to about 20 millionths) and wanders.',
  'S2: anchors share one clock; the tag clock drifts',
  'Same as S1, but the tag clock now drifts, like a real crystal: a little fast or slow, and wandering.');
resultSlide('Scenario S2 · Results', 'S2: only the clock model changes, and the results match S1', 'sweep_s2',
  'NEES against flight speed in S2 for EKF and EqF', [
    '**Filter change:** the clock error and drift may now wander',
    'Hovering: EqF 1.08, EKF 1.19',
    'From 5 m/s on: the same',
    'Clock error: 3 cm (of light travel) for both',
  ], 'Simulation, 50 flights of 30 s per speed (report Figure 2).',
  'The only change to the filter is extra noise in the clock model. The picture is the same as S1: the EqF is slightly better when hovering, and the same from 5 m/s.');
// S3
pictureSlide('Scenario S3', 'S3: each anchor has its own clock, and the anchors send packets to each other', 'pic_sc3',
  'Dashed lines: packets between anchors. Right: each anchor clock drifts differently.',
  'S3: every anchor has its own drifting clock and the anchors exchange packets',
  'Now the anchors are not synchronized: each has its own clock. To fix that, the anchors also send packets to each other.');
{
  const s = content('Scenario S3', 'S3: the filter adds an offset and a drift for each anchor clock', 'Scenarios');
  const d = fig(s, 'pic_room3', X0 + 0.2, 1.8, 4.4, 'Top view: anchors exchange packets');
  text(s, 'Dashed: packets between anchors', { x: X0 + 0.2, y: 1.85 + d.h, w: 4.4, h: 0.35, fontSize: 14, color: MUTED });
  const xr = 5.6, wr = XR - xr;
  const small = (t, y) => text(s, t, { x: xr, y, w: wr, h: 0.6, fontSize: 16 });
  // Clock offset and drift
  head(s, 'Each anchor clock has an offset and a drift', xr, 1.8, wr);
  eq(s, 's3dyn', xr + 0.3, 2.25);
  small('Δ_{i}: offset from anchor 0\'s clock. κ_{i}: its drift. η: small random wander. Anchor 0 sets the time. 7 anchors × 2 = **14 more states**.', 2.75);
  // Anchor packets
  head(s, 'Anchor-to-anchor packets give clock differences', xr, 3.5, wr);
  eq(s, 'anc', xr + 0.3, 3.95, 20 / 22);
  small('Anchor j sends, anchor m receives. Their distance is known, so the time gives Δ_{m} − Δ_{j}.', 4.4);
  // Group
  head(s, 'The group: anchor clocks are plain numbers', xr, 5.15, wr);
  eq(s, 's3grp', xr + 0.3, 5.6);
  small('G: the S1 group. Gal(3) has one clock slot, used by the tag. Anchor-clock errors are simple differences, as in the EKF.', 6.05);
  s.addNotes('Each anchor except anchor 0, the reference, gets a clock offset and a drift: 14 more numbers for 8 anchors. The offset grows with the drift, and both wander a little. Anchors also send packets to each other. Since the distance between two anchors is known, the arrival time tells us the difference between their clocks. For the group: Gal(3) has one clock slot, which the tag uses. The anchor clocks are added next to it as plain numbers, so their error is a simple difference, the same as in the EKF.');
}
resultSlide('Scenario S3 · Results', 'S3: all clocks are estimated, and the EqF and EKF still match', 'sweep_s3',
  'NEES against flight speed in S3 for EKF and EqF', [
    'Hovering: EqF 1.02, EKF 1.07',
    'Position error: 3 to 4 cm for both',
    '**Real IMU (MILUV), S1 to S3:** both filters give the same result on every flight',
  ], 'Simulation, 50 flights of 30 s per speed (report Figure 2). MILUV: 10 flights (report Figure 3).',
  'With the anchor packets, all clocks are estimated and again the two filters match. On real IMU data from MILUV, S1 to S3, the two filters give the same result on every flight. So with known anchor positions the EqF brings little.');
// S4
pictureSlide('Scenario S4', 'S4: the anchor positions are only known to about 30 cm', 'pic_sc4',
  'Dashed: the whole scene turned. Every reading stays the same.',
  'S4: anchor positions known only to about 30 cm; turning or shifting the whole scene gives the same readings',
  'Now the anchor positions are only known roughly, to about 30 cm, so the filter must estimate them too. Turning the whole scene, drone and anchors together, or shifting it, does not change any reading. So the data can never tell the filter about that.');
{
  const s = content('Scenario S4', 'S4: the EqF stays unsure about turning the whole scene; the EKF does not', 'Scenarios');
  text(s, 'The data cannot tell if the whole scene is turned or shifted. A good filter must stay unsure about it.', { x: X0, y: 1.85, w: CW, h: 0.8, fontSize: 22 });
  const bw = 3.85, gap = (CW - 3 * bw) / 2, y = 2.95, bh = 3.4;
  const cols = [
    ['EKF', EKF, PANEL, null, 'Its math changes with every new estimate, so it slowly becomes sure about the turn.', '→ too confident'],
    ['FEJ-EKF (common fix)', FEJ, PANEL, null, 'Keeps the math at the first estimates. But the first anchor guesses are 30 cm off.', '→ wrong when anchors exchange packets'],
    ['EqF', EQF, EQF_SOFT, EQF, 'Anchors are part of the group, so the turn looks the same for every estimate.', '→ stays unsure'],
  ];
  cols.forEach(([h, col, fill, edge, body, out], k) => {
    const x = X0 + k * (bw + gap);
    box(s, x, y, bw, bh, fill, edge);
    head(s, h, x + 0.25, y + 0.2, bw - 0.5, col, 22);
    text(s, body, { x: x + 0.25, y: y + 0.85, w: bw - 0.5, h: 1.7 });
    text(s, '**' + out + '**', { x: x + 0.25, y: y + 2.55, w: bw - 0.5, h: 0.75, color: col });
  });
  cite(s, 'FEJ-EKF: Huang, Mourikis and Roumeliotis, ICRA 2008. Equations in Appendix B.');
  s.addNotes('The data cannot tell whether the whole scene is turned or shifted, so a good filter has to stay unsure about that. The EKF linearizes at its current estimate, which changes every step, and it slowly becomes sure about the turn: too confident. The common fix, FEJ-EKF, keeps the linearization at the first estimates, but the first anchor guesses are 30 cm off. The EqF puts the anchors into the group, so turning the whole scene looks the same for every estimate and the filter stays unsure, without freezing anything.');
}
resultSlide('Scenario S4 · Results', 'S4: with anchor packets, only the EqF keeps its uncertainty right', 'slam_s3_2',
  'NEES over a 90 s flight with roughly known anchors, S3 clocks, 2 m/s: EKF, FEJ-EKF and EqF', [
    '**With anchor packets** (chart): EqF 1.1, FEJ-EKF 7.6 after 90 s',
    '**Fixed tag clock, 5 m/s:** EqF 1.05, FEJ 1.95, EKF 5.75',
    '**Hovering:** all three are too confident',
    'Position error: same for all, 12 to 15 cm',
  ], 'Simulation, 30 flights of 90 s; anchors 30 cm off; S3 clocks at 2 m/s in the chart (report Figure 4).',
  'NEES over a 90 second flight, with the anchor packets of S3, at 2 m/s. FEJ-EKF climbs to 7.6: it uses the anchor packets with its frozen, 30 cm-off anchor guesses. The EqF stays near 1. With a fixed tag clock at 5 m/s, the EqF is again the only one near 1. When hovering, all three are too confident: there the drone can turn on its own, which is not a whole-scene turn, so the group does not help. Accuracy is the same for all three.');

// ===================================================================
pres.addSection({ title: 'Overall' });
{
  const s = content('Overall', 'Overall: the EqF helps only when the anchor positions must be estimated', 'Overall');
  const H = (t) => ({ text: t, options: { bold: true, color: NAVY, fill: { color: PANEL } } });
  const c = (t, o) => ({ text: rich(t), options: o || {} });
  const win = { fill: { color: EQF_SOFT }, bold: true };
  const rows = [
    [H('Scenario'), H('Accuracy'), H('Best at knowing its error'), H('Real IMU (MILUV)')],
    [c('**S1** fixed tag clock'), c('same'), c('EqF, only when hovering'), c('same')],
    [c('**S2** drifting tag clock'), c('same'), c('EqF, only when hovering'), c('same')],
    [c('**S3** anchor clocks'), c('same'), c('EqF, only when hovering'), c('same')],
    [c('**S4** + fixed tag clock'), c('same'), c('EqF (at 5 m/s)', win), c('FEJ failed in 2 of 80 runs')],
    [c('**S4** + drifting tag clock'), c('same'), c('FEJ-EKF'), c('FEJ-EKF')],
    [c('**S4** + anchor clocks'), c('same'), c('EqF', win), c('FEJ-EKF is worst')],
  ];
  s.addTable(rows, {
    x: X0, y: 1.85, w: CW, colW: [3.4, 1.9, 3.6, 3.233], fontSize: 18, color: INK, valign: 'middle',
    margin: [0.06, 0.12, 0.06, 0.12], border: { type: 'solid', pt: 0.75, color: 'C9CFD7' }, rowH: 0.55,
  });
  text(s, 'With anchors off by 30 cm, no filter is right when hovering. With 10 cm, all three are fine.', { x: X0, y: 6.0, w: CW, h: 0.7, fontSize: 18 });
  s.addNotes('One row per scenario. Accuracy is the same everywhere. With known anchors, S1 to S3, the EqF is only slightly better when hovering. It matters with roughly known anchors: with a fixed tag clock and with anchor packets it is the only filter near 1. FEJ-EKF wins with a drifting tag clock, also on real IMU data. And with only a 10 cm anchor error, all three filters are fine.');
}
{
  const s = content('Conclusions', 'The EqF\'s gain comes from estimating anchors, not from the clock link', 'Overall');
  text(s, [
    '**Q1:** the clock–motion link is too small to matter for UWB.',
    '**Q2:** with known anchors, the EqF and EKF do the same; the EqF is slightly better when hovering.',
    '**Q3:** with roughly known anchors and anchor packets, only the EqF keeps its uncertainty right. Hovering is still unsolved.',
  ], { x: X0, y: 1.85, w: CW, h: 3.3, fontSize: 22, gap: 18 });
  box(s, X0, 5.25, CW, 1.45, PANEL);
  text(s, [{ t: '**Next**', gap: 6 },
    'Fix hovering (first-estimate math for the drone\'s own heading inside the EqF) · compare with an invariant EKF · real timestamps from Crazyflie'],
    { x: X0 + 0.25, y: 5.4, w: CW - 0.5, h: 1.2, fontSize: 18 });
  s.addText([{ text: 'Report: ', options: { color: MUTED } }, { text: REPORT.replace('https://', ''), options: { hyperlink: { url: REPORT }, color: EQF } }, { text: '   ·   feedback welcome', options: { color: MUTED } }],
    { isTextBox: true, x: X0, y: 7.0, w: 11.4, h: 0.3, margin: 0, fontSize: 12 });
  s.addNotes('Q1: no, the clock link is negligible for UWB. Q2: with known anchors the two filters are the same in practice. Q3: yes, with roughly known anchors and anchor packets the EqF is the only filter whose uncertainty stays right, but hovering is not solved. Limits: the fast scenes are scaled up, and the MILUV timestamps are simulated. Next steps are listed; feedback welcome.');
}

// ===================================================================
pres.addSection({ title: 'References and appendix' });
{
  const s = content('Sources', 'References', 'References and appendix');
  text(s, [
    'Delama et al. (2026). Galilean State Estimation for INS with Unknown Time Delay. Robotics: Science and Systems (RSS). arXiv:2605.13266.',
    'Huang, G. P., Mourikis, A. I., and Roumeliotis, S. I. (2008). Analysis and improvement of the consistency of extended Kalman filter based SLAM. IEEE ICRA, 473–479. doi:10.1109/ROBOT.2008.4543252.',
    'Li, M., and Mourikis, A. I. (2013). High-precision, consistent EKF-based visual-inertial odometry. International Journal of Robotics Research, 32(6), 690–711.',
    'Shalaby, M. A., et al. (2025). MILUV: A Multi-UAV Indoor Localization dataset with UWB and Vision. arXiv:2504.14376.',
    'Full report for this work: ' + REPORT,
  ], { x: X0, y: 1.8, w: CW, h: 4.8, fontSize: 16, gap: 12 });
  s.addNotes('References.');
}
// Appendix A: filter equations
{
  const s = content('Appendix A · Filter equations', 'The EqF on Gal(3): state, prediction and correction', 'References and appendix');
  head(s, 'Measurement and state', X0, 1.8, CW, NAVY, 18);
  const m = eq(s, 'meas', X0, 2.2, 0.8);
  const f = eq(s, 'Fmat', X0, 2.35 + m.h, 0.75);
  text(s, [
    'v_{F}, p_{F}: velocity and position in a free-falling frame',
    'v = v_{F} + gδ,  p = p_{F} + ½gδ^{2}',
    'error: X = exp(ε) X̂, ε has 20 numbers',
  ], { x: X0 + f.w + 0.4, y: 2.45 + m.h, w: CW - f.w - 0.4, h: 1.3, fontSize: 16, gap: 4 });
  head(s, 'Predict (IMU) and correct (packet)', X0, 4.15, CW, NAVY, 18);
  const p = eq(s, 'prop', X0, 4.55, 0.85);
  const h = eq(s, 'H', X0, 4.65 + p.h, 0.85);
  text(s, 'Blue: the clock–motion link (Q1). u_{i}: direction from anchor i to the tag.', { x: X0 + h.w + 0.3, y: 4.75 + p.h, w: CW - h.w - 0.3, h: 0.9, fontSize: 15 });
  s.addNotes('Backup with the filter equations from the report, sections 2 and 3.');
}
// Appendix A: symmetry group per scenario
{
  const s = content('Appendix A · Symmetry group', 'The symmetry group grows with each scenario', 'References and appendix');
  const H = (t) => ({ text: t, options: { bold: true, color: NAVY, fill: { color: PANEL } } });
  const c = (t) => ({ text: rich(t), options: {} });
  const colW = [1.6, 4.0, 2.9, 3.633], y0 = 1.85, hH = 0.5, rH = 0.85;
  s.addTable([
    [H('Scenario'), H('Group'), H('Error size'), H('New in this scenario')],
    [c('**S1, S2**'), c(' '), c('20'), c('Motion and tag clock (10), biases (10)')],
    [c('**S3**'), c(' '), c('34'), c('Anchor clock offsets and drifts (14)')],
    [c('**S4**'), c(' '), c('44 (58 with S3 clocks)'), c('Anchor positions, 8 × 3 = 24, as extra matrix columns')],
  ], {
    x: X0, y: y0, w: CW, colW, rowH: [hH, rH, rH, rH], fontSize: 18, color: INK, valign: 'middle',
    margin: [0.06, 0.12, 0.06, 0.12], border: { type: 'solid', pt: 0.75, color: 'C9CFD7' },
  });
  ['g_s12', 'g_s3', 'g_s4'].forEach((id, k) => {
    const m = MAN['eq_' + id], h = m.h / 96;
    eq(s, id, X0 + colW[0] + 0.15, y0 + hH + k * rH + (rH - h) / 2);
  });
  // Key, 2 x 2
  const ky = y0 + hH + 3 * rH + 0.35;
  head(s, 'How to read it', X0, ky, CW, NAVY, 18);
  const key = (id, t, x, y, dx, w) => {
    const h = MAN['eq_' + id].h / 96;
    eq(s, id, x, y + (0.4 - h) / 2);
    text(s, t, { x: x + dx, y: y + 0.03, w, h: 0.4, fontSize: 16 });
  };
  key('k_gal', 'rotation, velocity, position, clock (10)', X0, ky + 0.5, 1.2, 4.6);
  key('k_alg', 'biases, in the same slots as the IMU input (10)', X0, ky + 1.0, 1.2, 4.6);
  key('k_semi', 'linked: the right part moves with the frame', 7.0, ky + 0.5, 0.5, XR - 7.5);
  key('k_times', 'side by side: the error is a plain difference', 7.0, ky + 1.0, 0.5, XR - 7.5);
  s.addNotes('The symmetry group for each scenario. In S1 and S2 it is Gal(3), which holds rotation, velocity, position and the tag clock offset, linked with gal(3), which holds the biases in the same four slots as the IMU input: 20 numbers. In S3 the 14 anchor clock numbers are added side by side, so their error is a plain difference, as in the EKF. In S4 the 24 anchor position numbers become extra columns of the group matrix, so they turn and shift with the frame; with S3 clocks the anchor clocks are added side by side again, 58 numbers in total.');
}
// Appendix B: S4 math
{
  const s = content('Appendix B · S4 details', 'In the EqF the anchors are columns of the group matrix, so the scene turn never moves', 'References and appendix');
  const x = eq(s, 'Xanc', X0, 1.85);
  const xr = X0 + x.w + 0.4, wr = XR - xr;
  text(s, 'Turning or shifting the scene is the same direction for every estimate:', { x: xr, y: 1.9, w: wr, h: 0.7, fontSize: 16 });
  const a = eq(s, 'nyaw_eqf', xr, 2.65);
  eq(s, 'ntau', xr, 2.75 + a.h);
  text(s, 'In the EKF the turn direction depends on the estimate:', { x: X0, y: 4.1, w: CW, h: 0.35, fontSize: 16 });
  eq(s, 'nyaw_ekf', X0, 4.5, 1.15);
  text(s, 'A linearized filter stays unsure along N if', { x: X0, y: 5.35, w: 4.6, h: 0.35, fontSize: 16 });
  eq(s, 'obs', X0 + 4.6, 5.3);
  text(s, 'The EKF builds H at the predicted estimate but predicts from the corrected one, so the second condition fails at every update.',
    { x: X0, y: 5.9, w: CW, h: 0.8, fontSize: 16 });
  cite(s, 'Huang, Mourikis and Roumeliotis, ICRA 2008; Li and Mourikis, IJRR 2013');
  s.addNotes('Backup for questions on S4.');
}
// Appendix C: faster flights
{
  const s = content('Appendix C · Faster flights (not in the report)', 'Above 5 m/s the EqF\'s S1 advantage goes away; with anchor packets it stays closest to 1', 'References and appendix');
  const H = (t) => ({ text: t, options: { bold: true, color: NAVY, fill: { color: PANEL } } });
  const c = (t, o) => ({ text: rich(t), options: o || {} });
  const best = { fill: { color: EQF_SOFT } };
  const rows = [
    [H('Setting'), H('S1 clocks'), H('S2 clocks'), H('S3 clocks')],
    [c('Room, 5 m/s'), c('5.75 / 1.95 / **1.05**', best), c('7.09 / **1.72** / 2.68'), c('1.21 / 3.72 / **1.04**', best)],
    [c('Room, 7.5 m/s'), c('7.51 / **2.16** / 15.35'), c('10.95 / **2.10** / 11.80'), c('1.34 / 2.85 / **1.26**', best)],
    [c('Room, 10 m/s'), c('3.98 / 4.90 / 8.95'), c('6.86 / 5.20 / 10.29'), c('3.61 / 4.97 / 3.51')],
    [c('Bigger scene, 10 m/s'), c('0.98 / 1.01 / 0.95'), c('0.99 / 1.01 / 0.95'), c('1.08 / 1.49 / 1.03')],
    [c('Bigger scene, 20 m/s'), c('0.97 / 0.97 / 0.94'), c('1.01 / 1.00 / 0.97'), c('1.23 / 1.26 / 1.21')],
  ];
  text(s, 'NEES after 90 s: **EKF / FEJ-EKF / EqF**. Anchors 30 cm off. Blue: EqF closest to 1.', { x: X0, y: 1.8, w: CW, h: 0.4, fontSize: 16 });
  s.addTable(rows, {
    x: X0, y: 2.25, w: CW, colW: [3.0, 3.044, 3.044, 3.045], fontSize: 16, color: INK, valign: 'middle',
    margin: [0.05, 0.1, 0.05, 0.1], border: { type: 'solid', pt: 0.75, color: 'C9CFD7' }, rowH: 0.45,
  });
  text(s, [
    '**Room:** at 7.5 m/s the EqF loses its S1 lead (not explained yet). At 10 m/s even known anchors give 2.9 to 4.4.',
    '**Bigger scene:** all filters are fine, because 30 cm is small next to the scene.',
    '**Anchor error grown with the scene (1.2 to 4.8 m):** every filter fails.',
  ], { x: X0, y: 5.15, w: CW, h: 1.7, fontSize: 16, bullet: true, gap: 6 });
  s.addNotes('Backup, not in the report. Three ways to fly faster. In the same room the turns get very sharp, and the EqF loses its S1 lead at 7.5 m/s; I have not found why yet. In a bigger scene everything is fine because 30 cm is small there. If the anchor error grows with the scene, every filter fails. The anchor-packet case stays the EqF\'s best result.');
}

(async () => {
  const out = 'gal3uwb_update.pptx';
  await pres.writeFile({ fileName: out });
  // pptxgenjs writes an <a:pPr> before every run; only the first belongs to the paragraph.
  // Later ones are invalid and make LibreOffice drop the bullet, so remove them.
  const JSZip = require(require.resolve('jszip', { paths: [require.resolve('pptxgenjs')] }));
  const zip = await JSZip.loadAsync(fs.readFileSync(out));
  let removed = 0;
  for (const name of Object.keys(zip.files).filter(n => /^ppt\/(slides|slideLayouts|slideMasters)\/[^/]+\.xml$/.test(n))) {
    const xml = await zip.file(name).async('string');
    const fixed = xml.replace(/(<\/a:r>|<a:br\/>)<a:pPr\b(?:[^>]*?\/>|[^>]*?>[\s\S]*?<\/a:pPr>)/g, (m, head) => { removed++; return head; });
    zip.file(name, fixed);
  }
  fs.writeFileSync(out, await zip.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE' }));
  console.log('removed', removed, 'extra a:pPr');
  await applyTheme(out, THEME);
  console.log('wrote', out);
})().catch(e => { console.error(e); process.exit(1); });
