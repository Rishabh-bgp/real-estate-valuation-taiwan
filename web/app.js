"use strict";
// In-browser inference for scikit-learn GradientBoostingRegressor models exported to JSON.
const CENTER = [25.0478, 121.5170], PING_M2 = 3.305785;
const MONTHS = ["Aug 2012","Sep 2012","Oct 2012","Nov 2012","Dec 2012","Jan 2013","Feb 2013","Mar 2013","Apr 2013","May 2013","Jun 2013","Jul 2013"];
const DIST_MIN = 23.38, DIST_MAX = 6488.0, BOUNDS = [[24.93207, 121.47353], [25.01459, 121.56627]];
let model, data, marker, loc = [24.9711, 121.53863];
const f32 = Math.fround;

function haversine(a, b, c, d) {
  const r = x => x * Math.PI / 180, dl = r(c - a), dg = r(d - b);
  const h = Math.sin(dl / 2) ** 2 + Math.cos(r(a)) * Math.cos(r(c)) * Math.sin(dg / 2) ** 2;
  return 6371.0088 * 2 * Math.asin(Math.sqrt(h));
}
function tree(t, x) { let n = 0; while (t.l[n] !== -1) n = f32(x[t.f[n]]) <= t.t[n] ? t.l[n] : t.r[n]; return t.v[n]; }
function gbr(m, x) { let s = 0; for (const t of m.trees) s += tree(t, x); return m.base + m.lr * s; }
function features(r) {
  return [r.transaction_date, r.house_age, r.dist_mrt, r.n_stores, r.latitude, r.longitude,
          Math.log1p(r.dist_mrt), haversine(r.latitude, r.longitude, CENTER[0], CENTER[1])];
}
function predict(r) { const x = features(r); return {point: Math.exp(gbr(model.point, x)), lo: Math.exp(gbr(model.lo, x)), hi: Math.exp(gbr(model.hi, x))}; }
window.predictRaw = predict;

const $ = id => document.getElementById(id);
const distFromSlider = v => Math.round(DIST_MIN * Math.pow(DIST_MAX / DIST_MIN, v));
function inputs() {
  return {transaction_date: 2012 + (8 + +$("transaction_date").value) / 12, house_age: +$("house_age").value,
          dist_mrt: distFromSlider(+$("dist_mrt").value), n_stores: +$("n_stores").value, latitude: loc[0], longitude: loc[1]};
}
const fmt = (v, d = 1) => v.toLocaleString("en-US", {minimumFractionDigits: d, maximumFractionDigits: d});
const IMPORTANCE = [["log distance to MRT", 2.768], ["distance to MRT", 2.686], ["house age", 1.698], ["latitude", 0.872],
  ["distance to Taipei Main Stn", 0.530], ["longitude", 0.403], ["convenience stores", 0.102], ["transaction date", 0.081]];
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
let median, shown = null, anim;
function countTo(v) {
  const el = $("price"); cancelAnimationFrame(anim);
  if (shown === null || reduced) { shown = v; el.textContent = fmt(v); return; }
  const from = shown, t0 = performance.now(), dur = 450;
  const step = t => { const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
    el.textContent = fmt(from + (v - from) * e); if (k < 1) anim = requestAnimationFrame(step); else { shown = v; el.textContent = fmt(v); } };
  anim = requestAnimationFrame(step);
}
function fill(i) { i.style.setProperty("--p", ((i.value - i.min) / (i.max - i.min) * 100) + "%"); }
function render() {
  const r = inputs(), p = predict(r), size = +$("size").value;
  $("o-house_age").textContent = r.house_age.toFixed(1) + " yr";
  $("o-dist_mrt").textContent = r.dist_mrt.toLocaleString() + " m";
  $("o-n_stores").textContent = r.n_stores;
  $("o-transaction_date").textContent = MONTHS[+$("transaction_date").value];
  $("o-size").textContent = size + " ping"; $("sizeLbl").textContent = size;
  document.querySelectorAll(".seg button").forEach(b => b.setAttribute("aria-checked", +b.dataset.v === size));
  document.querySelectorAll("input[type=range]").forEach(fill);
  countTo(p.point);
  $("interval").textContent = `${fmt(p.lo)} – ${fmt(p.hi)}`;
  $("perm2").textContent = "NT$ " + Math.round(p.point * 1e4 / PING_M2).toLocaleString();
  $("total").textContent = "NT$ " + fmt(p.point * size / 100, 2) + " M";
  const d = (p.point / median - 1) * 100; $("vsmed").textContent = (d >= 0 ? "+" : "") + d.toFixed(0) + "%";
  const sc = v => Math.min(100, Math.max(0, v / 120 * 100));
  $("band").style.left = sc(p.lo) + "%"; $("band").style.width = (sc(p.hi) - sc(p.lo)) + "%"; $("dot").style.left = sc(p.point) + "%";
}
function drawImportance() {
  const mx = IMPORTANCE[0][1], box = $("impChart");
  box.innerHTML = IMPORTANCE.map(([n, v]) => `<div class="imp-row"><span>${n}</span><span class="t"><i data-w="${v / mx * 100}"></i></span><b>${v.toFixed(2)}</b></div>`).join("");
  requestAnimationFrame(() => box.querySelectorAll("i").forEach(i => { i.style.width = i.dataset.w + "%"; }));
}

function color(v) { const t = Math.min(1, Math.max(0, (v - 10) / 60)); return `hsl(${168 - 164 * t} ${62 + 10 * t}% ${42 + 10 * t}%)`; }
// Esri Canvas basemaps: free, no API key; attribution required.
const ESRI = "https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_", TAIL = "/MapServer/tile/{z}/{y}/{x}";
const ATTR = 'Tiles &copy; <a href="https://www.esri.com/">Esri</a> &mdash; Esri, HERE, Garmin, &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';
let tiles, map;
function setTiles(theme) {
  if (!map) return; if (tiles) map.removeLayer(tiles);
  const k = theme === "dark" ? "Dark_Gray" : "Light_Gray";
  tiles = L.layerGroup([L.tileLayer(ESRI + k + "_Base" + TAIL, {maxZoom: 16, attribution: ATTR}),
                        L.tileLayer(ESRI + k + "_Reference" + TAIL, {maxZoom: 16})]).addTo(map);
  tiles.eachLayer(l => l.bringToBack());
}
function initMap() {
  map = L.map("map", {scrollWheelZoom: false}).fitBounds(BOUNDS, {padding: [20, 20]});
  setTiles(document.documentElement.dataset.theme);
  L.rectangle(BOUNDS, {color: "#e0b252", weight: 1, fill: false, dashArray: "4"}).addTo(map);
  data.forEach(r => L.circleMarker([r[4], r[5]], {radius: 5, weight: 1, color: "rgba(0,0,0,.35)", fillOpacity: .85, fillColor: color(r[6])})
    .bindTooltip(`${r[6]} × 10k NT$/ping`).addTo(map));
  marker = L.marker(loc, {draggable: true, keyboard: true, title: "Property location",
    icon: L.divIcon({className: "", html: '<div class="pin"></div>', iconSize: [22, 22], iconAnchor: [11, 11]})}).addTo(map);
  const clamp = ll => [Math.min(BOUNDS[1][0], Math.max(BOUNDS[0][0], ll.lat)), Math.min(BOUNDS[1][1], Math.max(BOUNDS[0][1], ll.lng))];
  const set = ll => { loc = clamp(ll); marker.setLatLng(loc); render(); };
  map.on("click", e => set(e.latlng)); marker.on("dragend", () => set(marker.getLatLng()));
}

function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  $("theme").firstElementChild.textContent = t === "dark" ? "☾" : "☀";
  $("theme").setAttribute("aria-label", t === "dark" ? "Switch to light mode" : "Switch to dark mode");
  document.querySelector('meta[name=theme-color]').content = t === "dark" ? "#0b1416" : "#f6f1e8";
  setTiles(t);
}
$("theme").addEventListener("click", () => {
  const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark"; localStorage.setItem("theme", t); applyTheme(t);
});
applyTheme(localStorage.getItem("theme") || "dark");
document.querySelectorAll("input[type=range]").forEach(i => i.addEventListener("input", render));
document.querySelectorAll(".seg button").forEach(b => b.addEventListener("click", () => { $("size").value = b.dataset.v; render(); }));

Promise.all([fetch("model.json").then(r => r.json()), fetch("data.json").then(r => r.json())]).then(([m, d]) => {
  model = m; data = d; const s = d.map(r => r[6]).sort((a, b) => a - b); median = (s[206] + s[207]) / 2;
  initMap(); render(); drawImportance(); document.body.dataset.ready = "1";
});
