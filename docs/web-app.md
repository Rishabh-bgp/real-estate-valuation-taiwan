# Web app
Live: https://rishabh-bgp.github.io/real-estate-valuation-taiwan/
* **Inference in JS:** `model.json` holds every tree (feature, threshold, children, leaf value) of the three GBR models.
  `app.js` walks each tree (casting features with `Math.fround`, as scikit-learn compares float32 features), sums
  `base + learning_rate × Σ leaves`, and exponentiates.
* **UI:** bento layout with glass tiles; Leaflet map with OpenStreetMap tiles (no API key) showing all 414 sales
  coloured by price; click or drag the pin to set the location (clamped to the study area); sliders for age, MRT
  distance (log scale), stores, month and size; price per ping, per m², total and comparison with the dataset median;
  dark/light toggle that remembers your choice.
* **Accessibility:** labelled controls with live values, `aria-live` estimate, keyboard-draggable marker, visible focus,
  reduced-motion support.
