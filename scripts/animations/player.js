"use strict";
const D = JSON.parse(document.getElementById("data").textContent),
  N = D.observed.length,
  S = D.split;
const $ = (id) => document.getElementById(id),
  clamp = (v) => Math.max(0, Math.min(1, v));
const steps = [...document.querySelectorAll(".step")],
  fills = [...document.querySelectorAll(".fill")];
const copy = [
  [
    "Fit only the pre-intervention data.",
    "The model learns the trend and seasonal pattern from 48 untreated months. The shaded band shows 94% uncertainty in the fitted mean.",
  ],
  [
    "Forecast the world without the intervention.",
    "Extend the fitted pattern beyond January 2024. The shaded band carries uncertainty in that untreated mean into the forecast.",
  ],
  [
    "The gap is the estimated impact.",
    "Subtract the counterfactual from each observed outcome. Error bars show the resulting 94% credible intervals.",
  ],
];
const average = D.impact.slice(S).reduce((a, b) => a + b, 0) / (N - S);
let position = 0,
  playing =
    window.parent === window &&
    !matchMedia("(prefers-reduced-motion: reduce)").matches,
  last = null,
  currentStage = -1,
  geometry;
const ns = "http://www.w3.org/2000/svg";
function el(tag, attrs = {}, parent = $("chart")) {
  const e = document.createElementNS(ns, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  parent.appendChild(e);
  return e;
}
function label(txt, x, y, attrs = {}, parent = $("chart")) {
  const e = el("text", { x, y, ...attrs }, parent);
  e.textContent = txt;
  return e;
}
function path(values, from, to, y) {
  return values
    .slice(from, to)
    .map(
      (v, j) =>
        (j ? "L" : "M") +
        geometry.x(from + j).toFixed(2) +
        "," +
        y(v).toFixed(2),
    )
    .join(" ");
}
function draw() {
  const chart = $("chart"),
    impact = $("impact-chart");
  chart.replaceChildren();
  impact.replaceChildren();
  const w = Math.round(chart.getBoundingClientRect().width),
    mobile = w < 550,
    h = mobile ? 300 : 330,
    L = 36,
    R = 16,
    T = 42,
    B = h - 32;
  chart.setAttribute("viewBox", `0 0 ${w} ${h}`);
  impact.setAttribute("viewBox", `0 0 ${w} 112`);
  const x = (i) => L + ((w - L - R) * i) / (N - 1),
    y = (v) => B - ((v - 30) / 75) * (B - T),
    iy = (v) => 78 - v * 1.8;
  geometry = { w, h, x, y, iy, cut: x(S - 0.5), B };
  const title = el("title", { id: "chart-title" });
  title.textContent = "Fit, forecast, then estimate intervention impact";
  const desc = el("desc", { id: "chart-desc" });
  desc.textContent =
    "A model is fitted to the first 48 months. Its counterfactual forecast is compared with 24 months of observed post-intervention data. Green bands show 94% credible intervals for the expected untreated outcome.";
  el("rect", {
    x: geometry.cut,
    y: T,
    width: w - R - geometry.cut,
    height: B - T,
    fill: "var(--accent)",
    opacity: 0.035,
  });
  [40, 60, 80, 100].forEach((v) => {
    el("line", { x1: L, x2: w - R, y1: y(v), y2: y(v), stroke: "#263c49" });
    label(v, L - 9, y(v) + 4, { "text-anchor": "end" });
  });
  [0, 24, 48, 71].forEach((i, j) =>
    label(["2020", "2022", "2024", "2025"][j], x(i), B + 22, {
      "text-anchor": i === 71 ? "end" : "middle",
    }),
  );
  el("line", {
    x1: geometry.cut,
    x2: geometry.cut,
    y1: 29,
    y2: B,
    stroke: "#819ba9",
    "stroke-dasharray": "4 5",
  });
  label("INTERVENTION", geometry.cut + 7, 20, {
    "font-size": 9,
    "letter-spacing": 1,
    style: "fill:#d4e2e8",
  });
  label(mobile ? "PRE" : "PRE-INTERVENTION", L, 20, {
    "font-size": 9,
    "letter-spacing": 1,
  });
  const defs = el("defs"),
    pre = el("clipPath", { id: "pre-clip" }, defs),
    post = el("clipPath", { id: "post-clip" }, defs);
  el("rect", { id: "pre-reveal", x: L - 3, y: 0, width: 0, height: h }, pre);
  el(
    "rect",
    { id: "post-reveal", x: x(S - 1), y: 0, width: 0, height: h },
    post,
  );

  const bandPoints = (from, to) =>
    D.pred_upper
      .slice(from, to)
      .map((v, j) => `${x(from + j)},${y(v)}`)
      .concat(
        D.pred_lower
          .slice(from, to)
          .map((v, j) => `${x(from + j)},${y(v)}`)
          .reverse(),
      )
      .join(" ");
  el("polygon", {
    id: "fit-band",
    points: bandPoints(0, S),
    fill: "var(--accent)",
    opacity: 0.28,
    "clip-path": "url(#pre-clip)",
  });
  el("polygon", {
    id: "forecast-band",
    points: bandPoints(S - 1, N),
    fill: "var(--accent)",
    opacity: 0.28,
    "clip-path": "url(#post-clip)",
  });
  el("path", {
    id: "fit",
    d: path(D.prediction, 0, S, y),
    fill: "none",
    stroke: "var(--accent)",
    "stroke-width": 2.6,
    "clip-path": "url(#pre-clip)",
  });
  el("path", {
    id: "forecast",
    d: path(D.prediction, S - 1, N, y),
    fill: "none",
    stroke: "var(--accent)",
    "stroke-width": 2.6,
    "stroke-dasharray": "5 4",
    "clip-path": "url(#post-clip)",
  });
  const prePoints = el("g", { id: "observed-pre" });
  for (let i = 0; i < S; i++)
    el(
      "circle",
      { cx: x(i), cy: y(D.observed[i]), r: 2.5, fill: "#ffffff" },
      prePoints,
    );
  const postPoints = el("g", { id: "observed-post" });
  for (let i = S; i < N; i++)
    el(
      "circle",
      { cx: x(i), cy: y(D.observed[i]), r: 2.5, fill: "#ffffff" },
      postPoints,
    );
  label(
    mobile ? "IMPACT" : "POINTWISE IMPACT",
    L,
    16,
    { "font-size": 9, "letter-spacing": 1 },
    impact,
  );
  label(
    "observed − counterfactual",
    w - R,
    16,
    { "text-anchor": "end", "font-size": 9 },
    impact,
  );
  el(
    "line",
    {
      x1: L,
      x2: w - R,
      y1: iy(0),
      y2: iy(0),
      stroke: "#52707e",
      "stroke-dasharray": "3 4",
    },
    impact,
  );
  label("0", L - 10, iy(0) + 4, { "text-anchor": "end" }, impact);
  el(
    "line",
    {
      x1: geometry.cut,
      x2: geometry.cut,
      y1: 26,
      y2: 90,
      stroke: "#52707e",
      "stroke-dasharray": "4 5",
    },
    impact,
  );
  const idefs = el("defs", {}, impact),
    iclip = el("clipPath", { id: "impact-clip" }, idefs);
  el(
    "rect",
    { id: "impact-reveal", x: x(S) - 5, y: 25, width: 0, height: 70 },
    iclip,
  );
  const bars = el(
    "g",
    { id: "impact-intervals", "clip-path": "url(#impact-clip)" },
    impact,
  );
  const cap = Math.min(3.5, ((w - L - R) / (N - 1)) * 0.28);
  for (let i = S; i < N; i++) {
    const barWidth = Math.max(2, ((w - L - R) / (N - 1)) * 0.6);
    el(
      "rect",
      {
        x: x(i) - barWidth / 2,
        y: Math.min(iy(0), iy(D.impact[i])),
        width: barWidth,
        height: Math.abs(iy(0) - iy(D.impact[i])),
        fill: "var(--accent)",
      },
      bars,
    );
    const lower = iy(D.impact_lower[i]),
      upper = iy(D.impact_upper[i]);
    el(
      "line",
      {
        x1: x(i),
        x2: x(i),
        y1: lower,
        y2: upper,
        stroke: "#ffffff",
        "stroke-width": 1.5,
      },
      bars,
    );
    for (const cy of [lower, upper])
      el(
        "line",
        {
          x1: x(i) - cap,
          x2: x(i) + cap,
          y1: cy,
          y2: cy,
          stroke: "#ffffff",
          "stroke-width": 1.5,
        },
        bars,
      );
  }

  label(
    "The comparison appears in stage 03.",
    L,
    48,
    { id: "impact-placeholder", "font-size": 11 },
    impact,
  );
  render();
}
function render() {
  if (!geometry) return;
  const stage = Math.min(2, Math.floor(position / 8));
  fills.forEach(
    (e, i) => (e.style.transform = `scaleX(${clamp((position - i * 8) / 8)})`),
  );
  if (stage !== currentStage) {
    steps.forEach((b, i) => {
      b.classList.toggle("active", i === stage);
      b.setAttribute("aria-pressed", i === stage ? "true" : "false");
    });
    $("headline").textContent = copy[stage][0];
    $("description").textContent = copy[stage][1];
    currentStage = stage;
  }
  const pre = clamp(position / 5.5),
    forecast = clamp((position - 8) / 5.5),
    observed = clamp((position - 16) / 5.5);
  $("pre-reveal").setAttribute(
    "width",
    (geometry.x(S - 1) - geometry.x(0) + 6) * pre,
  );
  $("post-reveal").setAttribute(
    "width",
    (geometry.x(N - 1) - geometry.x(S - 1) + 3) * forecast,
  );
  $("impact-reveal").setAttribute(
    "width",
    (geometry.x(N - 1) - geometry.x(S) + 10) * observed,
  );
  $("impact-placeholder").style.opacity = 1 - clamp((position - 16) * 2);
  $("metric").textContent =
    stage === 0
      ? "48"
      : stage === 1
        ? "24"
        : observed < 1
          ? "…"
          : `+${average.toFixed(1)}`;
  $("metric-label").innerHTML =
    stage === 0
      ? "months used to fit<br>the untreated pattern"
      : stage === 1
        ? "months forecast<br>without intervention"
        : "average monthly impact<br>outcome units · posterior mean";
  $("scrub").value = position;
  $("time").textContent =
    `${String(Math.floor(position)).padStart(2, "0")} / 24 s`;
  $("play").textContent = playing ? "Ⅱ" : "▶";
  $("play").setAttribute(
    "aria-label",
    playing ? "Pause animation" : "Play animation",
  );
}
function setPlay(value) {
  playing = value;
  last = null;
  render();
}
$("play").addEventListener("click", () => {
  if (position >= 24) position = 0;
  setPlay(!playing);
});
$("replay").addEventListener("click", () => {
  position = 0;
  setPlay(true);
});
steps.forEach((b, i) =>
  b.addEventListener("click", () => {
    position = i * 8 + 6;
    setPlay(false);
  }),
);
$("scrub").addEventListener("input", (e) => {
  position = Number(e.target.value);
  setPlay(false);
});
document.addEventListener("visibilitychange", () => {
  last = null;
});
function frame(now) {
  if (playing && !document.hidden) {
    if (last !== null) position = Math.min(24, position + (now - last) / 1000);
    if (position >= 24) playing = false;
    render();
  }
  last = now;
  requestAnimationFrame(frame);
}
new ResizeObserver(draw).observe($("chart"));
if (matchMedia("(prefers-reduced-motion: reduce)").matches) position = 24;
requestAnimationFrame(frame);

// The docs host starts playback once the player is in view. User controls take precedence.
let userInteracted = false;
document.addEventListener("pointerdown", () => {
  userInteracted = true;
});
document.addEventListener("keydown", () => {
  userInteracted = true;
});
window.addEventListener("message", (event) => {
  if (event.source !== window.parent || event.origin !== window.location.origin)
    return;
  if (
    event.data?.type === "causalpy-animation-start" &&
    !userInteracted &&
    !matchMedia("(prefers-reduced-motion: reduce)").matches
  )
    setPlay(true);
});
