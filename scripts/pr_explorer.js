"use strict";

function evaluateBiases(logits, labels, biases, classCount) {
  const confusion = new Int32Array(classCount * classCount);
  for (let i = 0; i < labels.length; i++) {
    const offset = i * classCount;
    let winner = 0;
    let best = logits[offset] + biases[0];
    for (let c = 1; c < classCount; c++) {
      const score = logits[offset + c] + biases[c];
      if (score > best) { best = score; winner = c; }
    }
    confusion[labels[i] * classCount + winner]++;
  }
  const support = Array(classCount).fill(0), predicted = Array(classCount).fill(0);
  for (let t = 0; t < classCount; t++) {
    for (let p = 0; p < classCount; p++) {
      support[t] += confusion[t * classCount + p];
      predicted[p] += confusion[t * classCount + p];
    }
  }
  const precision = [], recall = [], iou = [];
  let correct = 0;
  for (let c = 0; c < classCount; c++) {
    const tp = confusion[c * classCount + c];
    correct += tp;
    precision.push(predicted[c] ? tp / predicted[c] : 0);
    recall.push(support[c] ? tp / support[c] : 0);
    iou.push(tp / Math.max(support[c] + predicted[c] - tp, 1));
  }
  const mean = a => a.reduce((s, x) => s + x, 0) / classCount;
  return {precision, recall, iou, support, predicted, confusion: Array.from(confusion),
    oa: correct / labels.length, miou: mean(iou), mean_recall: mean(recall),
    mean_precision: mean(precision)};
}

async function startExplorer() {
  const data = EXPLORER_DATA, provenance = data.provenance;
  const names = provenance.classes, C = names.length;
  const colors = [
    "#1764ab", "#d14928", "#288445", "#8f43b1", "#b47b00",
    "#00858a", "#c53678", "#72513d", "#5364d5", "#697d20",
    "#ba3038", "#007bba", "#9b4c99", "#d06b00", "#00816a",
    "#6350a5", "#8b6b25", "#5b747d", "#b15764", "#353f69"
  ];
  const bytes = text => Uint8Array.from(atob(text), c => c.charCodeAt(0));
  const logits = new Float32Array(bytes(data.logits).buffer);
  const labels = bytes(data.labels), biases = new Float64Array(C);
  const baseline = evaluateBiases(logits, labels, biases, C);
  const baselineFlat = provenance.baseline.confusion.flat();
  if (baseline.confusion.some((v, i) => v !== baselineFlat[i])) throw Error("Baseline metric verification failed");
  let current = baseline, active = null, queued = false, initialized = false;
  let view = "val", testSnapshot = null, testLogits = null, testLabels = null, testBaseline = null;
  const rows = [], sliders = [], numbers = [];
  const pct = x => (x * 100).toFixed(2) + "%";
  const pp = x => (x > 0 ? "+" : "") + (x * 100).toFixed(2);
  const deltaClass = x => x > 1e-12 ? "delta-positive" : x < -1e-12 ? "delta-negative" : "";
  const metrics = [["miou", "Mean IoU"], ["oa", "Overall accuracy"], ["mean_recall", "Average recall"]];
  const summary = document.getElementById("summary");
  for (const [key, name] of metrics) {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${name}</td><td id="baseline-${key}"></td><td id="metric-${key}"></td><td id="delta-${key}"></td>`;
    summary.appendChild(row);
  }
  function selectClass(c) {
    active = c;
    rows.forEach((row, i) => row.classList.toggle("selected", i === c));
    document.getElementById("selection").textContent = `${c + 1}. ${names[c]} is highlighted. Its slider adjusts this class's logit on every parcel.`;
  }
  function setBias(c, value) {
    if (!Number.isFinite(value)) return;
    biases[c] = Math.max(-100, Math.min(100, value));
    view = "val";
    const range = Math.max(Number(document.getElementById("span").value), Math.ceil(Math.abs(biases[c])));
    sliders[c].min = -range; sliders[c].max = range;
    sliders[c].value = biases[c]; numbers[c].value = biases[c].toFixed(2);
    selectClass(c); schedule();
  }
  const controls = document.getElementById("controls");
  names.forEach((name, c) => {
    const row = document.createElement("div"); row.className = "class-row"; row.dataset.class = c;
    row.style.setProperty("--class-color", colors[c]);
    row.style.setProperty("--class-tint", colors[c] + "14");
    row.innerHTML = `<div class="row-top"><button class="class-name"><span class="number">${c + 1}</span>${name}</button><input class="bias" type="text" inputmode="decimal" value="0.00" aria-label="${name} score bias"></div>
      <div class="range-line"><input type="range" min="-5" max="5" step="0.01" value="0" aria-label="${name} score bias slider"><button class="reset-one" aria-label="Reset ${name} bias">Reset</button></div>
      <table class="class-metrics"><thead><tr><th>Precision</th><th>Recall</th><th>IoU change, pp</th><th>Predicted / true</th></tr></thead><tbody><tr><td class="p"></td><td class="r"></td><td class="diou"></td><td class="counts"></td></tr></tbody></table>`;
    controls.appendChild(row); rows.push(row);
    const slider = row.querySelector('input[type="range"]'), number = row.querySelector('.bias');
    sliders.push(slider); numbers.push(number);
    slider.addEventListener("input", () => setBias(c, Number(slider.value)));
    number.addEventListener("change", () => {
      if (number.value !== "" && Number.isFinite(Number(number.value))) setBias(c, Number(number.value));
      else number.value = biases[c].toFixed(2);
    });
    row.querySelector(".reset-one").addEventListener("click", () => setBias(c, 0));
    row.querySelector(".class-name").addEventListener("click", () => { selectClass(c); schedule(); });
  });
  const layout = {
    margin: {l: 66, r: 30, t: 20, b: 60},
    xaxis: {title: {text: "Recall"}, range: [-.035, 1.035], tickformat: ".0%", dtick: .1, gridcolor: "#e9eef3", zeroline: false},
    yaxis: {title: {text: "Precision"}, range: [-.035, 1.035], tickformat: ".0%", dtick: .1, gridcolor: "#e9eef3", zeroline: false},
    paper_bgcolor: "white", plot_bgcolor: "white", font: {family: "Arial", size: 12, color: "#35475a"},
    hovermode: "closest", showlegend: false, uirevision: "keep-zoom"
  };
  function traces() {
    const shown = displayed(), base = shown.baseline, adjusted = shown.current;
    const original = document.getElementById("show-baseline").checked;
    const x = [], y = [];
    for (let c = 0; c < C; c++) {
      x.push(base.recall[c], adjusted.recall[c], null);
      y.push(base.precision[c], adjusted.precision[c], null);
    }
    return [
      {x, y, type: "scatter", mode: "lines", line: {color: "#c1cbd4", width: 1}, hoverinfo: "skip", visible: original},
      {x: base.recall, y: base.precision, type: "scatter", mode: "markers",
        marker: {symbol: "circle-open", size: 10, color: colors, opacity: .55, line: {width: 1.5}},
        text: names, hovertemplate: "%{text}<br>Original recall: %{x:.2%}<br>Original precision: %{y:.2%}<extra></extra>", visible: original},
      {x: active === null ? [] : [adjusted.recall[active]], y: active === null ? [] : [adjusted.precision[active]],
        type: "scatter", mode: "markers", marker: {size: 24, color: "rgba(255,255,255,0.8)", line: {color: active === null ? "#357aa7" : colors[active], width: 1.5}}, hoverinfo: "skip"},
      {x: adjusted.recall, y: adjusted.precision, type: "scatter",
        mode: document.getElementById("show-labels").checked ? "markers+text" : "markers",
        text: names.map((_, i) => String(i + 1)), textposition: "top right",
        textfont: {size: 10, color: colors}, customdata: names,
        marker: {symbol: "cross-thin", size: 12, color: colors, line: {color: colors, width: 2}},
        hovertemplate: "%{customdata}<br>Recall: %{x:.2%}<br>Precision: %{y:.2%}<extra>Adjusted</extra>"}
    ];
  }
  function displayed() {
    return view === "test" && testSnapshot
      ? {baseline: testBaseline, current: testSnapshot.metrics} : {baseline, current};
  }
  function testMatchesCurrent() {
    return testSnapshot !== null && testSnapshot.biases.every((value, c) => value === biases[c]);
  }
  function evaluateTest() {
    if (testLogits === null) {
      const saved = data.test;
      testLabels = bytes(saved.labels);
      testLogits = saved.probabilities
        ? Float64Array.from(new Float32Array(bytes(saved.probabilities).buffer), Math.log)
        : new Float32Array(bytes(saved.logits).buffer);
      testBaseline = evaluateBiases(testLogits, testLabels, new Float64Array(C), C);
      const expected = saved.provenance.baseline.confusion.flat();
      if (testBaseline.confusion.some((value, i) => value !== expected[i])) throw Error("Test baseline metric verification failed");
    }
    const selected = Array.from(biases);
    testSnapshot = {biases: selected, metrics: evaluateBiases(testLogits, testLabels, selected, C),
      validation: evaluateBiases(logits, labels, selected, C), evaluated_at: new Date().toISOString()};
    view = "test";
    document.getElementById("view-test").disabled = false;
    schedule();
  }
  async function update() {
    queued = false;
    current = evaluateBiases(logits, labels, biases, C);
    const shown = displayed();
    for (const [key] of metrics) {
      document.getElementById("baseline-" + key).textContent = pct(shown.baseline[key]);
      document.getElementById("metric-" + key).textContent = pct(shown.current[key]);
      const cell = document.getElementById("delta-" + key);
      cell.textContent = pp(shown.current[key] - shown.baseline[key]);
      cell.className = deltaClass(shown.current[key] - shown.baseline[key]);
    }
    rows.forEach((row, c) => {
      row.querySelector(".p").textContent = pct(current.precision[c]);
      row.querySelector(".r").textContent = pct(current.recall[c]);
      const delta = row.querySelector(".diou"); delta.textContent = pp(current.iou[c] - baseline.iou[c]);
      delta.className = "diou " + deltaClass(current.iou[c] - baseline.iou[c]);
      row.querySelector(".counts").textContent = `${current.predicted[c]} / ${current.support[c]}`;
    });
    await Plotly.react("plot", traces(), layout, {responsive: true, displaylogo: false,
      modeBarButtonsToRemove: ["select2d", "lasso2d"], toImageButtonOptions: {format: "svg", filename: "multiclass-pr"}});
    document.getElementById("metric-split").textContent = view === "test" ? "Test metric" : "Validation metric";
    document.getElementById("view-validation").setAttribute("aria-pressed", String(view === "val"));
    document.getElementById("view-test").setAttribute("aria-pressed", String(view === "test"));
    const status = document.getElementById("test-status");
    status.hidden = view !== "test";
    status.textContent = testMatchesCurrent()
      ? "Test result for the current biases. Slider edits return to validation; this snapshot stays fixed."
      : "Test result for earlier biases. Click Evaluate on test set to evaluate the current configuration.";
    document.getElementById("status").textContent = view === "test"
      ? `Fold ${provenance.fold} test | ${testLabels.length} parcels`
      : `Fold ${provenance.fold} validation | ${labels.length} parcels`;
  }
  function schedule() {
    if (!queued) { queued = true; requestAnimationFrame(() => update().catch(fail)); }
  }
  function fail(error) { document.getElementById("load-error").textContent = error.message; console.error(error); }
  document.getElementById("reset").addEventListener("click", () => {
    view = "val"; biases.fill(0); sliders.forEach(s => s.value = 0); numbers.forEach(n => n.value = "0.00"); schedule();
  });
  document.getElementById("evaluate-test").addEventListener("click", () => {
    try { evaluateTest(); } catch (error) { fail(error); }
  });
  document.getElementById("view-validation").addEventListener("click", () => { view = "val"; schedule(); });
  document.getElementById("view-test").addEventListener("click", () => {
    if (testSnapshot) { view = "test"; schedule(); }
  });
  if (data.automatic) {
    const button = document.getElementById("automatic");
    button.hidden = false;
    button.title = "Validation mIoU search, constrained to validation accuracy at least as high as baseline";
    button.addEventListener("click", () => {
      data.automatic.biases.forEach((value, c) => {
        setBias(c, value);
        numbers[c].value = String(value);
      });
      active = null;
      rows.forEach(row => row.classList.remove("selected"));
      document.getElementById("selection").textContent =
        "Loaded the validation-selected automatic search. Adjust the sliders to explore from here.";
    });
  }
  document.getElementById("span").addEventListener("change", () => {
    sliders.forEach((s, c) => {
      const range = Math.max(Number(document.getElementById("span").value), Math.ceil(Math.abs(biases[c])));
      s.min = -range; s.max = range; s.value = biases[c];
    });
  });
  ["show-baseline", "show-labels"].forEach(id => document.getElementById(id).addEventListener("change", schedule));
  function configuration() {
    const result = {schema_version: 2, fold: provenance.fold, split: "val",
      baseline_cache_sha256: provenance.baseline_cache_sha256,
      predictions_sha256: provenance.predictions_sha256,
      rule: provenance.rule, biases: Object.fromEntries(names.map((n, c) => [n, biases[c]])),
      metrics: evaluateBiases(logits, labels, biases, C),
      status: "Exploratory validation configuration"};
    if (testSnapshot) result.test_evaluation = {
      fold: provenance.fold, split: "test", provenance: data.test.provenance,
      evaluated_at: testSnapshot.evaluated_at,
      biases: Object.fromEntries(names.map((name, c) => [name, testSnapshot.biases[c]])),
      metrics: testSnapshot.metrics, baseline: testBaseline,
      validation_at_evaluation: testSnapshot.validation, matches_current_biases: testMatchesCurrent()
    };
    return result;
  }
  document.getElementById("export").addEventListener("click", () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(configuration(), null, 2) + "\n"], {type: "application/json"}));
    const link = document.createElement("a"); link.href = url; link.download = `fold-${provenance.fold}-biases.json`;
    link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  await update();
  document.getElementById("plot").on("plotly_click", event => {
    const point = event.points[0];
    if (point.curveNumber === 3) { selectClass(point.pointNumber); rows[active].scrollIntoView({block: "nearest"}); schedule(); }
  });
  initialized = true;
  window.prExplorer = {evaluateBiases, setBias, configuration,
    get state() { return {baseline, current, biases: Array.from(biases), ready: initialized,
      view, testSnapshot, testMatchesCurrent: testMatchesCurrent()}; }};
}

startExplorer().catch(error => {
  document.getElementById("load-error").textContent = error.message;
  console.error(error);
});
