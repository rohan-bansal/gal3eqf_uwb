"""Write build/figs.html and build/eqs.html for screenshotting.

figs.html reuses the report: its CSS (light theme), its scenario-diagram code
(Figure 1) verbatim, and its data (SWEEP, SLAM). figs_custom.js draws one
panel of Figures 2 and 4 at slide size.

eqs.html renders each slide equation with MathJax at the point size it has on
the slide (font-size in px = pt * 96 / 72), so the build script places the
image at its CSS size / 96 inches.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = Path(sys.argv[1]).resolve()
MATHJAX = (HERE / "../../.tools/node_modules/mathjax-full/es5/tex-svg-full.js").resolve()
INK = "#1d2229"
EQF = "#2a78d6"
EKF = "#b8481a"

# Slide geometry of each figure, in points (see build.js).
FIG_RESULT = (526, 342)   # results slides: 7.3 x 4.75 in
FIG_S2 = (475, 266)       # S2 slide: 6.6 x 3.7 in

# id, LaTeX, point size on the slide
EQS = [
    ("tdoa", r"t_1 - t_2 = \frac{d_1 - d_2}{c}", 26),
    ("raw", r"t_i - T = \frac{d_i}{c} + \delta", 26),
    ("meas", r"y_{k,i} \;=\; c\,(t_{k,i} - T_k) \;=\; \lVert p + R\,\ell - a_i \rVert + c\,(\delta + \Delta_i) + n_{k,i}", 26),
    ("state", r"x = (R,\ v,\ p,\ \delta,\ b_\omega,\ b_a,\ b_\rho)", 24),
    ("Fmat", r"F = \begin{bmatrix} R & v_F & p_F \\ 0 & 1 & \delta \\ 0 & 0 & 1 \end{bmatrix} \in \mathrm{Gal}(3)", 24),
    ("vp", r"v = v_F + g\,\delta, \qquad p = p_F + \tfrac12\, g\,\delta^2", 20),
    ("galact", r"(x,\ t) \;\mapsto\; (R\,x + v\,t + p,\ \ t + s)", 20),
    ("err_eqf", r"X = \exp(\varepsilon)\,\hat X, \qquad \varepsilon \in \mathbb R^{20}", 20),
    ("grp", r"\mathbf G = \mathrm{Gal}(3) \ltimes \mathfrak{gal}(3)", 20),
    ("err_ekf", r"e = x \ominus \hat x, \qquad e \in \mathbb R^{17}", 20),
    ("nees", r"\overline{\mathrm{NEES}} = \frac{1}{\dim e}\; e^\top P^{-1} e", 26),
    ("dyn", r"\dot F = -g_N^\wedge F + F\,(w - b)^\wedge, \qquad w = (\omega_m,\ a_m,\ 0,\ 1)", 20),
    ("prop", r"\hat F \leftarrow \Gamma\,\hat F\,\exp\!\big((w - \hat b)\,\Delta T\big), \qquad P \leftarrow A\,P\,A^\top + B\,Q\,B^\top", 20),
    ("H", r"H_i = \big[\; \underbrace{-u_i^\top (\hat p_F + \hat R\,\ell)^\wedge}_{\text{rotation}} \quad \underbrace{{\color{" + EQF + r"}\hat\delta}\, u_i^\top}_{\text{velocity}} \quad \underbrace{u_i^\top}_{\text{position}} \quad \underbrace{c + {\color{" + EQF + r"}\hat\delta}\, u_i^\top g}_{\text{clock}} \;\big|\; \underbrace{0}_{\text{biases}} \;\big]", 20),
    ("s2", r"\dot\delta = -b_\rho + \eta_\delta, \qquad \dot b_\rho = \eta_\rho", 24),
    ("anc", r"y_{jm} = \lVert a_m - a_j \rVert + c\,(\Delta_m - \Delta_j) + n", 22),
    ("s3dyn", r"\dot\Delta_i = \kappa_i + \eta_\Delta, \qquad \dot\kappa_i = \eta_\kappa, \qquad \Delta_0 \equiv 0", 20),
    ("s3grp", r"\mathbf G \times \mathbb R^{2(N-1)} = \mathbf G \times \mathbb R^{14}", 20),
    ("prior", r"\hat a_i^{(0)} \sim \mathcal N(a_i,\ \sigma_a^2 I), \qquad \sigma_a = 30 \text{ or } 10 \text{ cm}", 20),
    ("scene", r"(R,\ v,\ p,\ a_i) \;\mapsto\; (R_z R,\ R_z v,\ R_z p + \tau,\ R_z a_i + \tau)", 19),
    ("nyaw_eqf", r"N_{\mathrm{yaw}} = (e_z,\ 0,\ \dots,\ 0)", 20),
    ("nyaw_ekf", r"N_{\mathrm{yaw}}(\hat x) = \big(\hat R^\top e_z,\ e_z \times \hat v,\ e_z \times \hat p,\ 0,\ e_z \times \hat a_i\big)", 16),
    # Appendix A
    ("Xanc", r"X = \begin{bmatrix} R & v_F & p_F & a_0 & \cdots & a_{N-1} \\ 0 & 1 & \delta & 0 & \cdots & 0 \\ 0 & 0 & 1 & 0 & \cdots & 0 \\ 0 & 0 & 0 & & I_N & \end{bmatrix}", 20),
    ("Hanc", r"H_i:\quad -u_i^\top(\hat p_F + \hat R\,\ell - \hat a_i)^\wedge \ \text{(rotation)}, \qquad -u_i^\top \ \text{(anchor } i\text{)}", 16),
    ("ntau", r"N_\tau = (0,\ 0,\ \tau,\ 0,\ \dots,\ 0,\ \tau,\ \dots,\ \tau)", 20),
    ("obs", r"H_k N_k = 0, \qquad \Phi_k N_k = N_{k+1}", 20),
    # Appendix A: symmetry group per scenario
    ("g_s12", r"\mathrm{Gal}(3) \ltimes \mathfrak{gal}(3)", 20),
    ("g_s3", r"\big(\mathrm{Gal}(3) \ltimes \mathfrak{gal}(3)\big) \times \mathbb R^{14}", 20),
    ("g_s4", r"\big(\mathrm{Gal}(3) \ltimes \mathbb R^{24}\big) \ltimes \mathfrak{gal}(3)", 20),
    ("k_gal", r"\mathrm{Gal}(3)", 20),
    ("k_alg", r"\mathfrak{gal}(3)", 20),
    ("k_semi", r"\ltimes", 20),
    ("k_times", r"\times", 20),
]


def figs_html():
    rep = (HERE / "report.html").read_text()
    css = re.search(r"<style>\n/\* Layout.*?</style>", rep, re.S).group(0)
    lines = rep.split("\n")

    def block(start, end):  # 1-based inclusive line numbers in report.html
        return "\n".join(lines[start - 1:end])

    # Helpers (NS ... fmtLen) and the Figure 1 code, verbatim from the report.
    i_helpers = next(i for i, l in enumerate(lines) if "const NS = 'http://www.w3.org/2000/svg';" in l) + 1
    i_tip = next(i for i, l in enumerate(lines) if "// Tooltip for any element with data-tip." in l)
    i_scen = next(i for i, l in enumerate(lines) if l.strip().startswith("function scenariosFigure()")) + 1
    i_fig2 = next(i for i, l in enumerate(lines) if "// ---------- Figure 2" in l)
    helpers = block(i_helpers, i_tip)
    scen = block(i_scen, i_fig2)
    sweep = next(l for l in lines if l.strip().startswith("const SWEEP = "))
    slam = next(l for l in lines if l.strip().startswith("const SLAM = "))
    custom = (HERE / "figs_custom.js").read_text()
    override = """
<style>
  :root { --panel: #ffffff; --paper: #ffffff; }
  body { background: #fff; margin: 0; padding: 20px; }
  svg.viz { width: auto; }
  svg.viz text { font-family: Arial, "Liberation Sans", Helvetica, sans-serif; font-size: 17px; fill: #4a505b; }
  svg.viz .small { font-size: 16px; fill: #5b626d; }
  svg.viz .strong { fill: #16191e; font-weight: 600; }
  .band2 { fill: #f1f3f6; }
  .shot { display: inline-block; background: #fff; margin: 0 0 20px; }
  /* Figure 1 panels: larger type for the slide */
  .scen svg.viz text { font-size: 15px; }
  .scen svg.viz .small { font-size: 14.5px; }
  .pic svg.viz text { font-size: 12px; }
  .pic svg.viz .small { font-size: 11px; }
</style>"""
    W, H = FIG_RESULT
    w2, h2 = FIG_S2
    body = f"""
<div style="position:absolute; left:-5000px">
<svg class="viz" id="sc1"></svg><svg class="viz" id="sc2"></svg><svg class="viz" id="sc3"></svg><svg class="viz" id="sc4"></svg>
</div>
<div class="pic shot" data-shot="pic_sc1"><svg class="viz" id="p1" viewBox="0 0 420 225" width="630" height="337.5"></svg></div>
<div class="pic shot" data-shot="pic_sc2"><svg class="viz" id="p2" viewBox="0 0 420 225" width="630" height="337.5"></svg></div>
<div class="pic shot" data-shot="pic_sc3"><svg class="viz" id="p3" viewBox="0 0 420 225" width="630" height="337.5"></svg></div>
<div class="pic shot" data-shot="pic_sc4"><svg class="viz" id="p4" viewBox="0 0 420 225" width="630" height="337.5"></svg></div>
<div class="pic shot" data-shot="pic_room1"><svg class="viz" id="r1" viewBox="2 2 212 221" width="318" height="331.5"></svg></div>
<div class="pic shot" data-shot="pic_room3"><svg class="viz" id="r3" viewBox="2 2 212 221" width="318" height="331.5"></svg></div>
<div class="shot" data-shot="sweep_s1"><svg class="viz" id="sw1"></svg></div>
<div class="shot" data-shot="sweep_s2"><svg class="viz" id="sw2"></svg></div>
<div class="shot" data-shot="sweep_s3"><svg class="viz" id="sw3"></svg></div>
<div class="shot" data-shot="slam_s3_2"><svg class="viz" id="sl32"></svg></div>
"""
    script = f"""
<script>
(() => {{
'use strict';
{helpers}
{scen}
{sweep}
{slam}
{custom}
scenariosFigure();
[['p1','sc1'],['p2','sc2'],['p3','sc3'],['p4','sc4'],['r1','sc1'],['r3','sc3']].forEach(([d, src]) => {{ $(d).innerHTML = $(src).innerHTML; }});
sweepPanel($('sw1'), '1', {{ W: {W}, H: {H} }});
sweepPanel($('sw2'), '2', {{ W: {W}, H: {H} }});
sweepPanel($('sw3'), '3', {{ W: {W}, H: {H} }});
slamPanel($('sl32'), '0.3', '3', '2', {{ W: {W}, H: {H} }});
window.FIGS_DONE = true;
}})();
</script>"""
    return "<!doctype html><html><head><meta charset=utf-8>" + css + override + "</head><body>" + body + script + "</body></html>"


def eqs_html():
    divs = []
    for eid, tex, pt in EQS:
        px = pt * 96 / 72
        divs.append(f'<div class="e" data-shot="eq_{eid}" style="font-size:{px:.2f}px">\\(\\displaystyle {tex}\\)</div>')
    return f"""<!doctype html><html><head><meta charset=utf-8>
<style>body {{ margin: 0; padding: 20px; color: {INK}; background: transparent; }}
.e {{ display: inline-block; padding: 3px 2px; margin: 0 0 16px; line-height: 1; }}
.e mjx-container {{ margin: 0 !important; }}
.wrap {{ display: flex; flex-direction: column; align-items: flex-start; }}</style>
<script>window.MathJax = {{ tex: {{ inlineMath: [['\\\\(', '\\\\)']] }}, svg: {{ fontCache: 'local' }},
  startup: {{ pageReady: () => MathJax.startup.defaultPageReady().then(() => {{ window.EQS_DONE = true; }}) }} }};</script>
<script src="file://{MATHJAX}"></script></head>
<body><div class="wrap">{''.join(divs)}</div></body></html>"""


BUILD.mkdir(parents=True, exist_ok=True)
(BUILD / "figs.html").write_text(figs_html())
(BUILD / "eqs.html").write_text(eqs_html())
print("wrote", BUILD / "figs.html", BUILD / "eqs.html")
