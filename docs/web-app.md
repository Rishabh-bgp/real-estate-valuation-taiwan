# Web app
Live: https://rishabh-bgp.github.io/real-estate-valuation-taiwan/
* **Inference in JS:** `model.json` holds every tree (feature, threshold, children, leaf value) of the three GBR models.
  `app.js` walks each tree (casting features with `Math.fround`, as scikit-learn compares float32 features), sums
  `base + learning_rate × Σ leaves`, and exponentiates.
* **UI (redesigned Oct 2026):** bento-grid dashboard, dark mode by default with a light toggle (saved in localStorage), Inter + Noto Serif TC labels, mesh-gradient background with an original cloud-scroll pattern, original SVG Taipei skyline / temple-roof / lantern / lotus line art (no logos, no religious figures), animated count-up price, range bar, segmented size control, permutation-importance mini chart, `prefers-reduced-motion` respected. Leaflet map on Esri Canvas dark/light gray basemaps (no API key; attribution shown) with all 414 sales. The inference code (`tree`, `gbr`, `features`, `predict`) is unchanged.
  coloured by price; click or drag the pin to set the location (clamped to the study area); sliders for age, MRT
  distance (log scale), stores, month and size; price per ping, per m², total and comparison with the dataset median;
  dark/light toggle that remembers your choice.
* **Accessibility:** labelled controls with live values, `aria-live` estimate, keyboard-draggable marker, visible focus,
  reduced-motion support.
