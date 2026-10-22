> From: https://claude.ai/chat/22589345-3feb-4843-99e2-d5a9e1a715ba?artifact=ea73865e-7af5-443d-acdf-f81d87a54643

# you asked

message time: 2026-10-08 21:47:44

https://arxiv.org/pdf/2605.13266

what would it take to implement this into GTSAM

---

# claude response

Less than you'd expect. GTSAM's `develop` branch (checked today) already has most of the math. What's missing is the delay-specific glue.

**Already in GTSAM**
- `Gal3`: the Galilean group with Expmap, Logmap, Adjoint and Jacobians.
- `TangentLieGroup<Gal3>`: exactly the paper's symmetry group Gal(3) ⋉ gal(3), dimension 20. The docs even list it as a typical alias.
- `EquivariantFilter<M, Symmetry>`: a generic EqF. It has `predictWithTransition()` for passing your own discrete A and B, explicit-H updates, and the covariance reset/transport step.
- `AbcEquivariantFilterExample.cpp`: a worked EqF using the same biased-tangent-group pattern, on SO(3).
- `Gal3ImuEKF`, `GalileanPreintegration`, `GalileanImuFactor`: Galilean IMU propagation, but with time as absolute time, not as a delay.

**What you'd need to write**
1. **State and symmetry.** M = Gal3 × R¹⁰ (core state plus IMU and virtual biases), and a `Symmetry` functor for the state action φ (eq. 28). Note that the paper's bias vector is 10D, with the two virtual biases from Delama et al. 2025. GTSAM's existing preintegration uses the plain 6D physical bias, so you can't reuse `imuBias::ConstantBias` directly.
2. **Dynamics.** A new gravity term g_N whose time slot cancels, so δ stays constant (the -N F + F N terms). None of `Gal3ImuEKF`'s three modes do this. Then add the lift Λ (eqs. 30–31) and the closed-form A and B (eqs. 34–36) fed into `predictWithTransition`. You could start with automatic Jacobians and swap in the closed form later.
3. **Sliding preintegration buffer.** This is the main genuinely new component. It needs:
   - a ring buffer of `Gal3` matrices Υₖ over about 0.5 s, all right-multiplied each IMU step with the bias-corrected input (eq. 20)
   - lookup of Υ(δ̂) with interpolation
   - a raw IMU buffer to fetch ω at t − δ̂ for C₂

   Roughly 150 lines.
4. **Delayed GNSS update.** h(ξ) = P(F Υ(δ)⁻¹ ȳ₀) (eq. 38) with the explicit C (eqs. 40–42), passed to `updateWithVector`.
5. **Wrapping.** A class like `Gal3DelayEqF`, unit tests that check A, B, and C against numerical derivatives, a Monte Carlo example matching their 100–500 ms sim, and Python bindings in `navigation.i`.

**Main pitfall: conventions.** The paper's tangent order is (Ω, v, r, τ). GTSAM's `Gal3` internal ordering appears to match, but its constructor takes (R, **r, v**, t). Left versus right composition sides in `EquivariantFilter` also have to match how they wrote φ and the reset. Expect most of the debugging time here, so do the numerical-Jacobian tests first.

**Effort:** about 800–1,200 lines including tests. That's 1–2 weeks for someone comfortable with GTSAM's Lie group templates, mostly spent on items 2 and 3 plus convention checking.

**Alternative:** If you'd rather stay in GTSAM's smoothing world than filtering, treat δ as a variable in a fixed-lag smoother with a delayed `GPSFactor` built from the same buffered Υ(δ). A recent follow-up ([arXiv 2608.29514](https://arxiv.org/pdf/2608.29514)) does a sliding-window version on SGal(3) and discusses the consistency pitfalls of that route.

Sources:
- [Delama et al., Galilean State Estimation for INS with Unknown Time Delay](https://arxiv.org/pdf/2605.13266)
- [Sliding Window Filter on the Galilean Group](https://arxiv.org/pdf/2608.29514)
- GTSAM `develop`: `gtsam/navigation/`, `gtsam/base/TangentLieGroup.h`, `gtsam/geometry/Gal3.h`

---

# you asked

message time: 2026-10-09 14:38:03

building on this, i want to explore gal(3) and equivairant filter applications to UWB (ultra wideband radio). 

i have attached and linked beloew some readings on this: https://arxiv.org/pdf/2607.15807


my professor's opinion is that there are niches and quirks of using gal(3) and eqfs in robotics that people in UWB space dont know about, and would make a good paper while staying simple. any ideas

---

# claude response

I'll read the three papers, then pitch some paper ideas.

I'd build the paper around one gap: **UWB clock skew fits Gal(3) exactly, and nobody has used it there yet.** All of these ideas build on the Delama/Mahony delay-EqF work you just read.

## 1. Clock skew as the Galilean time-rate bias (strongest idea)
- **The gap.** Mahony et al. write that the κ entry in gal(3) "could be used to encode clock drift if required" and then leave it there. Delama's 10D bias already includes b_ρ, a time-rate bias that is *virtual* in INS. In UWB it is physical: crystal skew of about ±20 ppm. That works out to roughly 6 m of range error per ms of uncorrected reply time, which is the whole reason DS-TWR exists.
- **The model.** Each UWB node becomes a Galilean frame. Its time coordinate is the clock offset and b_ρ is the skew. The paper would be "UWB clocks are Galilean frames."
- **Prior art to beat.** Shalaby, Forbes et al. (T-RO 2024) solve multi-robot UWB with clock offset and skew. They use an EKF on SE₂(3) with offset and skew as separate Euclidean states, and they assume distances and skews stay constant during each ranging transaction.
- **What Gal(3) adds.** Clock-offset error and position error are coupled through velocity. This is the event transform p' = Qp + tb + q, shown in Mahony's Fig. 5, where timestamp uncertainty stretches the uncertainty along the direction of motion. Their EKF can't represent that.

## 2. Passive TDoA as Galilean events
This ties idea 1 to Dhekne's PnPLoc.
- Every overheard anchor message is an event (p, t) in homogeneous coordinates. The tag timestamps it in its own drifting clock.
- A TDoA reading is a difference of time components. That gives a measurement model with the same structure as Delama's GNSS model (their eq. 37) or Mahony's eq. 55.
- Add the phone's IMU and you have a privacy-preserving, synchronization-free tag running a Galilean EqF.
- I'd merge ideas 1 and 2 into one paper.

## 3. Motion during the ranging exchange (quick, clean side result)
- DS-TWR and Shalaby both assume nothing moves during the POLL/RESP/FINAL exchange. With TDMA schedules across many anchors (Raab et al. run about 300 ranges/s across 3 tags and 12 anchors), an exchange can span several ms. A drone at 10–20 m/s moves cm per ms.
- Galilean preintegration Υ(δ) between the timestamps t₀, t₄, t₆ gives an exact event model.
- **Honest caveat:** the error is a few cm, close to UWB noise. So this is a section or a workshop paper, not the main contribution.

## 4. Unknown UWB-to-IMU latency
- This is a direct port of the RSS paper, with a range measurement in place of GNSS position.
- It's the easiest of the four but the least novel. It's also the most likely to get scooped, since Delama co-authored both the Galilean delay paper and the Klagenfurt UWB AMR paper you linked.

## The quirk to make the headline: consistency
UWB people mostly don't know that EKFs go overconfident when they linearize along unobservable directions. In TDoA, global clock offset is unobservable (only clock differences are), and so are global translation and yaw. Raab et al. report outdoor NEES of about 40–860 with their EKF, and Kelly et al. showed time-offset states break EKF consistency. Equivariant error coordinates handle those directions properly. So make NEES versus a Shalaby-style EKF the main plot, not just RMSE. It's a simple, convincing result for a UWB audience.

**Practical notes**
- Start in simulation. Gal3 and TangentLieGroup<Gal3> already exist in GTSAM, so the group machinery is free.
- For real data, check whether Shalaby's dataset includes raw timestamps. The UTIAS UTIL TDoA dataset (UWB + IMU + mocap) is another option, from Kelly's lab, which co-wrote the Galilean paper.
- Move reasonably fast. Klagenfurt (Weiss/Delama) and ANU (Mahony/Ge, who already did an EqF for range-only SLAM, without clocks) are both one step from this. Clocks and passive TDoA are the part neither group has touched.

Sources:
- [Shalaby et al., Multi-Robot Relative Pose Estimation and IMU Preintegration Using Passive UWB Transceivers](https://arxiv.org/html/2304.03837v3)
- [Ge et al., Equivariant Filter Design for Range-only SLAM](https://arxiv.org/pdf/2503.03973)
- [Raab et al., Deployment-Ready UWB Localization for Industrial Ground Robots](https://arxiv.org/pdf/2607.15807)
- [Delama et al., Galilean State Estimation for INS with Unknown Time Delay](https://arxiv.org/pdf/2605.13266)
- [Clock Rigidity and Joint Position-Clock Estimation in UWB Sensor Networks](https://arxiv.org/pdf/2106.02199)
- Mahony, Kelly, Weiss, *Galilean Symmetry in Robotics* (your attached PDF, Sec. 2.5 and 6)
- Dhekne, *UWB Radios for Spatial Intelligence*, GetMobile 2025 (your attached PDF)

---

# you asked

message time: 2026-10-09 14:44:21

what is TDoA? can you ELI5 UWB

---

# claude response

**UWB, ELI5:** UWB is a radio that measures distance by timing very short pulses. A normal radio like Wi-Fi sends a long, smooth wave, so it's hard to tell exactly when a message arrived. UWB sends a pulse about a nanosecond long, which is a sharp click instead of a hum. A sharp click lets you timestamp its arrival very precisely.

Radio travels at the speed of light, about 30 cm per nanosecond. If you can time arrivals to a fraction of a nanosecond, you know distances to within roughly 10 cm. That's what AirTags and car keys use to say "your phone is 2.3 m away, that direction."

**The catch is clocks.** Each device has its own cheap crystal clock. They don't agree on what time it is (offset), and they tick at slightly different speeds (skew, roughly 20 parts per million). At light speed, a 1-nanosecond clock disagreement is 30 cm of error. So almost every UWB method is really a trick for cancelling clock errors.

**Two main tricks:**
- **Two-way ranging (TWR).** I shout, you shout back, and I time the round trip. I only use my own clock, so the offset cancels. Adding a third message (that's DS-TWR) also cancels most of the skew. The downside is that every tag has to transmit, which uses airtime, limits how many tags you can have, and reveals that the tag exists.
- **Time difference of arrival (TDoA).** Fixed anchors on the walls send pulses, and the tag just listens. It can't know when a pulse left, so it doesn't get absolute distances. But it can measure that anchor A's pulse arrived 3 ns before anchor B's, which means it's about 0.9 m closer to A than to B. Each of those differences puts you on a curve (a hyperbola). Intersect a few curves and you have your position. Because only differences matter, the tag's own clock offset cancels.

TDoA's appeal is that tags never transmit. So any number of phones can listen at once, and the building never learns who's there. Its usual cost is that the anchors need synchronized clocks. Dhekne's PnPLoc avoids that: the anchors range with each other using TWR, and the phone overhears those exchanges and works out the differences itself.

**Why this matters for your project:** every UWB measurement is really "a pulse left here at this time and arrived there at that time." That's an event (position plus time), which is exactly what Gal(3) is built to handle. Clock offset maps to the Galilean time coordinate, and clock skew to the time-rate term.

---

# you asked

message time: 2026-10-09 14:47:17

ok given this context what is the proposal

---

# claude response

## Proposal: Galilean equivariant filtering for synchronization-free UWB localization

**One-line pitch:** A UWB tag's clock offset and skew aren't nuisance parameters to bolt onto a pose filter. They are the time coordinate of a Galilean frame and its rate bias. Modelling them that way gives an equivariant filter that tracks pose and clock together, and stays consistent where EKFs don't.

**Problem.** Passive TDoA systems like PnPLoc let unlimited tags localize privately without synchronized anchors. Two things still go wrong:
- The tag's clock offset and skew corrupt every timestamp. At 20 ppm skew, 1 ms between overheard messages is 6 m of error.
- The tag moves between the receptions that get compared, but current methods treat each fix as static or bolt on an EKF. Shalaby et al. carry offset and skew as separate vectors next to SE₂(3) and assume nothing moves during a ranging transaction. EKFs in this space are known to be badly overconfident: Raab et al. report NEES of 40–860 against an ideal of about 1.

**Approach**
- **State.** The tag is a Galilean frame (R, v, p, τ) relative to the anchor network's time. τ is the clock offset. The bias vector holds the IMU biases plus b_ρ, the time-rate bias, which here is the physical clock skew. In Delama's INS paper that same slot is a purely mathematical ("virtual") bias.
- **Symmetry.** Gal(3) ⋉ gal(3), the same 20-dim group as Delama et al. This is `TangentLieGroup<Gal3>` in GTSAM, so the group machinery already exists.
- **Measurements.** The tag timestamps the anchor-to-anchor ranging packets it overhears (PnPLoc-style). Each reception is a spacetime event. The IMU-preintegrated Galilean increment links two receptions at different times, so motion between them is modelled exactly rather than ignored.
- **Filter.** A discrete-time EqF following Delama's Algorithm 1, with the GNSS update swapped for a passive-TDoA update.

**Contributions**
1. A model of UWB clocks as Galilean frames, in which clock skew is the time-rate bias. That's the κ slot Mahony et al. flag as possible but never use.
2. An EqF for passive TDoA plus IMU that jointly estimates pose, clock offset and skew online.
3. A consistency analysis showing the EqF stays calibrated (NEES ≈ 1) where EKFs don't, especially as clock offset becomes unobservable or speed increases.
4. Optional: a correction for motion during the ranging exchange, which affects fast platforms like drones.

**Experiments**
- **Simulation first.** Monte Carlo sweeps over skew, tag speed, anchor geometry and initial error.
- **Real data.** UWB anchors with raw timestamp access (DW1000/DW3000), a tag with an IMU (a phone or a small drone), and mocap for ground truth. Before collecting new data, check whether the UTIL TDoA dataset or Shalaby's data has what we need.
- **Baselines.**
  - PnPLoc's closed-form TDoA plus trilateration
  - a Shalaby-style SE₂(3) EKF with separate clock states
  - the same EKF without the skew states
- **Metrics.** Position RMSE, clock-offset/skew error, NEES, and convergence from bad initialization.

**Scope.** About 8–10 weeks:
- Weeks 1–3: derivation and simulation
- Weeks 4–6: EqF in GTSAM and baselines
- Weeks 7–10: real data and write-up

Possible venues are RA-L (rolling submissions), IROS 2027 (deadline around March), or IPIN to reach the UWB community directly.

**Risks**
- Klagenfurt (Weiss/Delama) and ANU (Mahony/Ge) are each one step from this. Leading with clocks and passive TDoA is the differentiation.
- Skew is only observable with enough time and motion, so the experiments need to show where that holds.
- Real data needs raw timestamp access, which depends on having the right UWB hardware.

**Questions for your professor**
- Is passive TDoA the right setting, or would two-way ranging be easier to get data for?
- What UWB hardware does the lab have?
- Which venue is the target?

---

# you asked

message time: 2026-10-09 14:52:30

can you do some lit review and confirm this hasnt been done/what we have to compare against? then draft an easy to read proposal

---

# claude response

Outline's up; now running the lit review.

Background's in; now running the lit review searches.

---

# you asked

message time: 2026-10-09 15:02:26

finish up and draft now

---

# claude response

Drafting the remaining sections now.

The proposal is drafted in the doc, ready for you to edit inline or comment on.

The lit review found no prior work that does this: nobody has built a Galilean or equivariant filter for UWB with clock offset and skew on the group. The closest papers are Shalaby et al. (T-RO 2024), which estimates clocks as separate states alongside pose, and Delama et al. (RSS 2026), which uses a Galilean filter but for GNSS delay with no clock skew. Shalaby is the main baseline to beat.

The review does have gaps:
- **Dataset check:** I stopped this one when you asked me to wrap up. So I haven't checked whether the UTIL dataset has the raw timestamps we'd need, and it's flagged in the doc.
- **Two older papers:** Ledergerber et al. (IROS 2015) and Hamer & D'Andrea (2018) may estimate UWB clocks in a plain EKF, but I couldn't open them. Read them before citing.

The biggest risk, also in the doc: PnPLoc's TDoA formula already cancels clock skew for a tag that isn't moving. The paper has to show the gain for a moving tag with IMU fusion, and in filter consistency.

---

# you asked

message time: 2026-10-15 01:58:59

the proposal is a bit too complicated - i like the idea, but i want to shorten it and make it easier to understand. also was thinking of using a drone somehow since collaborator wants to, any way to inrtegrate into teting plan

---

# claude response

Rewriting it shorter, with the drone as the main test.

Now the drone testing plan.

Updating the timeline for the drone work.

I've cut the proposal to about half its length, in plainer language, and made the drone the main test.

**What changed:**
- The background, approach and contributions sections are now two short sections, "The problem" and "The idea."
- The literature table is down to the four closest papers, each with one line on what it's missing.
- Risks are down to three, with a short list of questions for your professor and collaborator.

**How the drone fits:** the earlier draft's biggest risk was that existing methods already handle a tag that isn't moving. A drone answers that directly. You fly the same path at three speeds (hover, about 0.5 m/s, about 2 m/s), and the headline plot is error and filter consistency versus speed. Every method should tie at hover, and our filter should pull ahead as speed goes up.

The default setup is a small drone called a Crazyflie with a UWB add-on board, because the public UTIL dataset used that same setup. That lets you test on UTIL before flying anything yourself. If your collaborator has a bigger drone, you'd mount a UWB tag on it and use its built-in IMU instead.

The timeline now runs Oct 19 to Dec 20. The simulation check comes before any drone work, and a second check in week 4 confirms you can log raw UWB timestamps from the drone.

The doc is ready to read over, edit or comment on.

---

# you asked

message time: 2026-10-15 21:16:00

how does this differ from https://arxiv.org/pdf/2607.15807 and https://arxiv.org/pdf/2605.13266

---

# claude response

Neither paper does what we're proposing, but they sit on either side of it. Raab et al. is the UWB application with no clock or Galilean modelling. Delama et al. is the Galilean filter with no UWB or clock skew. We combine the two.

| | Raab et al. 2026 ([2607.15807](https://arxiv.org/pdf/2607.15807)) | Delama et al., RSS 2026 ([2605.13266](https://arxiv.org/pdf/2605.13266)) | Our proposal |
|---|---|---|---|
| Platform | Warehouse ground robot | Fixed-wing UAV | Indoor drone |
| Sensors | UWB two-way ranging plus wheel odometry | GNSS position plus IMU | Listen-only UWB (TDoA) plus IMU |
| Filter | Error-state EKF on a ground-surface model | Equivariant filter on the Galilean group | Equivariant filter on the Galilean group |
| Clocks | Not modelled; constant and range-dependent biases per anchor absorb timing errors | One constant GNSS delay; all clocks tick at the same rate | Tag clock offset plus skew, which grows over time |
| Main contribution | Automatic anchor calibration, terrain-aware fusion, a dataset | Joint navigation and delay estimation that stays consistent | Clock offset and skew on the group for UWB, tested across drone speeds |

**Against Raab et al.:** it's a different problem. Their tag transmits (two-way ranging), so clock offset mostly cancels and they never estimate a clock. Ours only listens, so the tag's clock error appears in every measurement. Their paper is useful to us as motivation: their best EKF still reaches outdoor NEES around 40 (ideal is about 1).

**Against Delama et al.:** this is the paper we build on. The filter machinery is largely theirs. The differences are:
- **What the time coordinate means.** Their δ is a fixed sensor latency. Our τ is the gap between two clocks, and it keeps changing because of skew.
- **Skew.** Their time-rate bias b_ρ is a mathematical device with no physical meaning. In ours it's the real crystal skew, and we estimate it.
- **Measurement.** They update on a GNSS position, which is close to linear in the state. We update on arrival timestamps of overheard packets, a nonlinear range-difference model they don't cover.

**The risk to plan for:** a reviewer could call this "Delama's filter with a different sensor." The paper has to show something their setup can't, which is what the skew estimate and the consistency-versus-speed plot are for. Delama and Weiss are authors on both papers, so their group is the most likely competitor, and probably a reviewer too.

The doc's "Why it's new" table currently lists Delama but not Raab. I can add a Raab row if you want both there.

---

# you asked

message time: 2026-10-15 22:19:37

can you confirm and do lit review and ensure no one has done this before

---

# claude response

I'll run a second, targeted sweep that covers what the first one couldn't verify, the new drone angle, and anything posted in the last few weeks.

Two searches are running in parallel: one on drone and listen-only UWB work that tracks clocks with standard filters, and one on anything new in the Galilean and equivariant-filter literature, including papers that cite Delama and Mahony. Once they're back, I'll update the "Why it's new" table in the doc.

---

# claude response

The Galilean/equivariant search is back: nothing found that does this, and the closest new paper (van Goor & Forbes, Sep 2026) uses UWB ranging with no clocks. Still waiting on the drone/UWB clock search before I update the doc.

---

# claude response

Both searches are back. Updating the doc with what changed.

I can't promise no one has done this, but I'm confident the core idea is unpublished. Two separate searches found nothing that does it. The new search does change what the paper can claim, though, and I've updated the doc to match.

**Not done anywhere I could find:** putting UWB clock offset and skew on the Galilean group, or into any equivariant filter. The closest papers miss one piece each:
- Delama (RSS 2026) uses the Galilean filter, but only for a fixed GNSS delay.
- van Goor & Forbes (Sep 2026) is an equivariant filter with UWB, but has no clocks.
- Shalaby (T-RO 2024) tracks UWB clocks, but as separate numbers next to the pose, and its drones transmit rather than only listen.

The only papers citing Delama or Mahony are their own groups' delay papers, and none touch clocks.

**Already done, which I hadn't flagged before:** tracking a listen-only tag's clock offset and skew inside a normal filter is established:
- Cano, Chidami & Le Ny (ICRA 2019)
- Navrátil et al. (2022)
- ETH quadcopter papers from 2015 and 2018

So the paper can't claim "first to estimate UWB clocks." It can claim the first Galilean or equivariant version. It can also claim the first test of whether these filters stay consistent as drone speed goes up; nobody reports that.

**Two practical findings that change the plan:**
- **UTIL won't work as test data.** It only stores processed range differences, not the raw timestamps the filter needs. MILUV, a public multi-quadcopter UWB dataset, has raw timestamps, though its drones transmit rather than only listen. Your own Crazyflie flights would need a small firmware change to log raw timestamps.
- **Clock offset can't be recovered from range differences.** Taking differences cancels it out, so the filter must work directly on raw arrival times.

**In the doc:** the "Why it's new" table now covers all of the above, including Raab. I also updated the testing setup (MILUV instead of UTIL, the Crazyflie's built-in filter added as a baseline), the risks (the offset problem, and that the Galilean model assumes all clocks tick at the same rate) and the references.

**Still unchecked:**
- I couldn't open the full texts of the two ETH quadcopter papers (Ledergerber 2015, Hamer & D'Andrea 2018). They're the most likely to overlap on the drone side, so read them before submitting.
- I couldn't confirm the exact title of the Cano ICRA 2019 paper, and it's marked as such in the references.
- Work under review that isn't on arXiv yet can't be searched.

---

# you asked

message time: 2026-10-16 13:33:24

UWB tags find their position by timing radio pulses, but every tag's cheap clock is a little wrong and drifts

is there any site that proves this

---

# claude response

Yes. These are the best sources to cite, best first:

1. **Bitcraze's TDoA documentation** (the Crazyflie UWB system you'd fly) is the clearest statement of it:
   - "each Anchor and Tag has its own clock that is completely un-synchronized to the other clocks"
   - "The clock frequency is drifting over time and this error must be compensated for in each unit."

   [TDoA principles](https://wiki.bitcraze.io/doc:lps:tdoa:principles) · [TDoA3 protocol](https://www.bitcraze.io/documentation/repository/lps-node-firmware/2019.09/tdoa3_protocol)

2. **Sidorenko et al., ["Decawave UWB Clock Drift Correction and Power Self-Calibration"](https://arxiv.org/abs/1902.11085)** (Sensors 2019) measures it on real DW1000 chips:
   - "it is not possible to manufacture exactly the same clock generators"
   - The two clocks differed by "about five parts per million".
   - With a 2 ms gap between messages, that produced "the distance error resulting from the clock drift about 1 m."

   This is probably the best one to cite in the paper, because it puts a number on the error.

3. **A [Qorvo developer forum thread](https://forum.qorvo.com/t/clock-drift-compensation-for-ranging/3262)** (Qorvo makes the DW1000 chip) has a user reporting a crystal with "10ppm accuracy" that needed trimming and drift correction. It's a forum post, so it's backup only.

4. **Dhekne's GetMobile article**, which you attached, says the three-message ranging exchange exists "to compensate for the different clock offsets and clock drifts between the ranging pair."

The "30 cm per nanosecond" figure is just the speed of light, so it doesn't need a source.

For a slightly stronger sentence in the doc: "Each UWB device has its own unsynchronized crystal clock; real DW1000 chips differ by about 5 ppm, enough to cause about 1 m of ranging error over a 2 ms exchange (Sidorenko et al., 2019)."

---

# you asked

message time: 2026-10-19 19:44:56

this project was approved. can this be easily tested in simulation? if so, whcih simulator?

---

# claude response

Yes, this is easy to test in simulation, and you don't need a robotics simulator for it. Write a small custom simulator in Python using GTSAM's Python bindings. That's also how the closest papers did it: Delama et al. used an analytic trajectory with synthetic 200 Hz IMU and 50 Monte Carlo runs, and Shalaby et al. used a 500-run Monte Carlo.

**Why not Gazebo or a drone simulator?** Your result depends on clock behaviour at the timestamp level: each device's offset, its skew, and exact pulse arrival times. The UWB simulators I found model distances, noise and walls blocking the signal, not clocks. For example, the [TU Graz Gazebo UWB plugin](https://www.tugraz.at/institute/iti/lehre/offene-projekte/modular-uwb-plugin-for-gazebo-ignition-towards-a-realistic-simulation-framework) is still a thesis proposal and doesn't mention clock drift. Drone physics also doesn't matter for a filter that only reads IMU and UWB data.

**What the simulator needs (a few hundred lines):**
1. **Flight path.** A smooth looping path at three speeds (hover, about 0.5 m/s, about 2 m/s).
2. **IMU.** GTSAM's Python bindings already include `Scenario` / `ScenarioRunner`, which turn a path into noisy IMU readings with bias.
3. **Clocks.** Each anchor and the tag get a random offset, plus a skew of a few ppm (Sidorenko measured about 5 ppm between real DW1000 chips) that drifts slowly.
4. **UWB timestamps.** Anchors broadcast on a schedule. For each packet, compute the true arrival time at the moving tag (distance divided by the speed of light). Then convert it to the tag's clock and add about 0.1 ns of noise.
5. **Monte Carlo.** About 50 runs per speed, recording error and NEES.

`gtsam.Gal3` is in the Python bindings, along with an example equivariant filter (`EqF.py`) to borrow structure from. The generic `EquivariantFilter` class isn't exposed in Python. So either write the Galilean filter's predict and update steps in Python on top of `gtsam.Gal3`, or do it in C++ later if you need speed.

**A realistic middle step before flying:** take the real drone path and IMU from a public dataset (MILUV or UTIL, which both have motion-capture ground truth) and simulate only the UWB timestamps on top. You get real drone motion and IMU noise, but full control over the clocks.

**Optional, if your collaborator wants it:** a Crazyflie software-in-the-loop simulator (Crazyswarm2 or CrazySim) would let the filter run inside the real firmware. That's integration work for after the method works, not for proving the idea.

I can write the simulator skeleton (path, IMU, clocks, UWB timestamps, and the Monte Carlo loop) as a starting point if you'd like.