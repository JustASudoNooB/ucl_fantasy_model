  // ================================================================ views, profiles, lab (ufe/report/parts/20_app.js)
  const HOOKS = [];                                   // run after every matchday change
  const VIEWS = ["matchday", "players", "matches", "clubs", "my-team", "lab", "league", "guide"];
  const PUBLIC = !!(D.meta && D.meta.public);
  const hasLeague = !PUBLIC && (D.managers || []).length > 0;
  const mainEl = $("main");
  function viewOf(hash) {
    const h = String(hash || "").replace(/^#/, "");
    if (VIEWS.includes(h)) return [h, null];
    const el = h ? document.getElementById(h) : null;
    const sec = el && el.closest("main > section");
    return sec && sec.dataset.view ? [sec.dataset.view, el] : [store.get("view", "matchday"), null];
  }
  function showView(hash, scroll) {
    let [v, target] = viewOf(hash);
    if (!VIEWS.includes(v) || (v === "league" && !hasLeague)) v = "matchday";
    mainEl.dataset.v = v;
    $$("#views a").forEach(a => a.dataset.v === v ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current"));
    store.set("view", v);
    if (target) target.scrollIntoView({ block: "start" });
    else if (scroll) window.scrollTo(0, 0);
  }
  $("#views a[data-v='league']").hidden = !hasLeague;
  window.addEventListener("hashchange", () => showView(location.hash, true));
  showView(location.hash, false);
  if (!store.get("seen", false)) $("#welcome").hidden = false;
  $("#welcome-x").addEventListener("click", () => { store.set("seen", true); $("#welcome").hidden = true; });

  document.addEventListener("click", e => {
    if (e.target.closest("button, a, input, select, label, summary")) return;
    const el = e.target.closest("[data-pid]");
    if (el) openPlayer(el.getAttribute("data-pid"));
  });
  document.addEventListener("keydown", e => {
    if (e.key === "Enter" && e.target.getAttribute && e.target.getAttribute("data-pid") && !e.target.closest("dialog")) { e.preventDefault(); openPlayer(e.target.getAttribute("data-pid")); }
  });

  // ---------------------------------------------------------------- numbers
  const SEASON = (D.season && D.season.players) || {};
  const S_MDS = (D.season && D.season.mds) || [];
  const mdPlayers = () => D.matchdays[MD].players;
  MDS.forEach(m => D.matchdays[m].players.forEach(p => {              // ownership and points so far, for the table and the lab
    const s = SEASON[p.id], o = s && s.own != null ? s.own : (P[p.id] || {}).own;
    if (o != null) p.own = o;
    if (s) p.season = s.s[9];
  }));
  const METRICS = {
    xpts: { l: "Expected points", f: f2 }, price: { l: "Price (€m)", f: f1 }, value: { l: "Points per €1m", f: f2 },
    xpts_if_start: { l: "Points if he starts", f: f2 }, p_start_sim: { l: "Start chance", f: pct, pc: 1 },
    p_goal: { l: "Goal chance", f: pct, pc: 1 }, p_assist: { l: "Assist chance", f: pct, pc: 1 },
    xgoals: { l: "Expected goals", f: f2 }, xassists: { l: "Expected assists", f: f2 }, xrec: { l: "Ball recoveries", f: f1 },
    p_cs: { l: "Clean sheet chance", f: pct, pc: 1 }, p_haul: { l: "Chance of 10+ points", f: pct, pc: 1 }, p_blank: { l: "Chance of 2 or fewer", f: pct, pc: 1 },
    p10: { l: "Floor (10th percentile)", f: f1 }, p90: { l: "Ceiling (90th percentile)", f: f1 },
    own: { l: "Ownership (%)", f: v => f1(v) + "%" }, season: { l: "Points so far", f: v => String(Math.round(v)) },
    ppm: { l: "Points per 90 so far", f: f2 }, next3: { l: "Expected points, next 3 matchdays", f: f1 }
  };
  function mval(p, k) {
    const s = SEASON[p.id];
    if (k === "own") return s && s.own != null ? s.own : (P[p.id] || {}).own;
    if (k === "season") return s ? s.s[9] : null;
    if (k === "ppm") return s && s.s[0] >= 90 ? s.s[9] / (s.s[0] / 90) : null;
    if (k === "next3") { const e = D.ev[p.id], i = MDS.indexOf(MD); return e ? e.slice(i, i + 3).reduce((a, c) => a + c, 0) : null; }
    return p[k];
  }
  const mfmt = (k, v) => v == null ? "–" : METRICS[k].f(v);
  const median = a => { const s = a.slice().sort((x, y) => x - y), n = s.length; return n ? (n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2) : 0; };
  function ticks(lo, hi, n) {
    const span = (hi - lo) || 1, raw = span / (n || 5), mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= (n || 5) + 1) || mag * 10;
    const out = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
    return out;
  }
  const lin = (d0, d1, r0, r1) => v => r0 + (v - d0) / ((d1 - d0) || 1) * (r1 - r0);
  const tickLab = (k, v) => METRICS[k] && METRICS[k].pc ? Math.round(v * 100) + "%" : (Math.abs(v) >= 100 || Number.isInteger(v) ? String(v) : String(+v.toFixed(2)));
  // labels placed greedily in priority order, skipped when they would overlap one already placed
  function placeLabels(items, W, H) {
    const boxes = [], out = [];
    for (const it of items) {
      const w = it.text.length * 6.4 + 6, h = 14;
      const tries = [[it.x + 8, it.y + 4, "start"], [it.x - 8, it.y + 4, "end"], [it.x, it.y - 9, "middle"], [it.x, it.y + 18, "middle"]];
      for (const [x, y, anchor] of tries) {
        const x0 = anchor === "start" ? x : anchor === "end" ? x - w : x - w / 2, bx = { x0, y0: y - 11, x1: x0 + w, y1: y - 11 + h };
        if (bx.x0 < 2 || bx.x1 > W - 2 || bx.y0 < 2 || bx.y1 > H - 2) continue;
        if (boxes.some(b => bx.x0 < b.x1 && bx.x1 > b.x0 && bx.y0 < b.y1 && bx.y1 > b.y0)) continue;
        boxes.push(bx); out.push(`<text class="lab" x="${x.toFixed(1)}" y="${y.toFixed(1)}" text-anchor="${anchor}">${esc(it.text)}</text>`);
        break;
      }
    }
    return out.join("");
  }
  function axes(k, kx, ky, x0, x1, y0, y1, sx, sy, W, H, m) {
    let g = "";
    ticks(y0, y1, 5).forEach(v => { g += `<line class="gr" x1="${m.l}" x2="${W - m.r}" y1="${sy(v).toFixed(1)}" y2="${sy(v).toFixed(1)}"/><text x="${m.l - 8}" y="${(sy(v) + 4).toFixed(1)}" text-anchor="end">${tickLab(ky, v)}</text>`; });
    ticks(x0, x1, 7).forEach(v => { g += `<line class="gr" x1="${sx(v).toFixed(1)}" x2="${sx(v).toFixed(1)}" y1="${m.t}" y2="${H - m.b}"/><text x="${sx(v).toFixed(1)}" y="${H - m.b + 16}" text-anchor="middle">${tickLab(kx, v)}</text>`; });
    g += `<line class="ax" x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}"/><line class="ax" x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${H - m.b}"/>`;
    return g;
  }

  // ---------------------------------------------------------------- the lab: scatter
  const PRESETS = [
    ["Value", "price", "xpts", "Players above the pack at their price are the value picks; the dashed lines are the medians."],
    ["Differentials", "own", "xpts", "Top left is the differential corner: a strong forecast and few owners."],
    ["Threat v creativity", "xgoals", "xassists", "Right is goal threat, up is creativity. The top-right corner does both."],
    ["Floor v ceiling", "p10", "p90", "Up is the captain's corner (high ceiling). Right is the safe corner (high floor)."],
    ["Rotation risk", "p_start_sim", "xpts_if_start", "High but to the left: he scores well when he plays, and may not play."],
    ["Form v forecast", "ppm", "xpts", "Far right and low: he has outscored what the model expects next. Hot streaks rarely carry over."]
  ];
  let lab = Object.assign({ x: "price", y: "xpts", pos: "ALL", starters: true, mine: false }, store.get("lab", {}) || {});
  if (!METRICS[lab.x] || !METRICS[lab.y]) { lab.x = "price"; lab.y = "xpts"; }
  let labPts = [], labSel = null, labGeo = null;
  function setupLab() {
    const opts = Object.keys(METRICS).map(k => `<option value="${k}">${esc(METRICS[k].l)}</option>`).join("");
    $("#lab-x").innerHTML = opts; $("#lab-y").innerHTML = opts;
    $("#lab-presets").innerHTML = PRESETS.map((p, i) => `<button data-i="${i}">${esc(p[0])}</button>`).join("");
    $("#lab-pos").innerHTML = ["ALL", "DEF", "MID", "FWD", "GK"].map(p => `<button data-p="${p}">${p === "ALL" ? "All" : p}</button>`).join("");
    $$("#lab-presets button").forEach(b => b.addEventListener("click", () => { const p = PRESETS[+b.dataset.i]; lab.x = p[1]; lab.y = p[2]; labSel = null; renderLab(); }));
    $$("#lab-pos button").forEach(b => b.addEventListener("click", () => { lab.pos = b.dataset.p; labSel = null; renderLab(); }));
    $("#lab-x").addEventListener("change", e => { lab.x = e.target.value; labSel = null; renderLab(); });
    $("#lab-y").addEventListener("change", e => { lab.y = e.target.value; labSel = null; renderLab(); });
    $("#lab-starters").addEventListener("change", e => { lab.starters = e.target.checked; labSel = null; renderLab(); });
    $("#lab-mine").addEventListener("change", e => { lab.mine = e.target.checked; renderLab(); });
    $("#lab-find").addEventListener("input", () => renderLab());
    const svg = $("#lab-svg");
    const nearest = e => {
      if (!labGeo) return null;
      const r = svg.getBoundingClientRect(), x = (e.clientX - r.left) * labGeo.W / r.width, y = (e.clientY - r.top) * labGeo.H / r.height;
      let best = null, bd = 30 * 30;
      labPts.forEach(q => { const d = (q.cx - x) * (q.cx - x) + (q.cy - y) * (q.cy - y); if (d < bd) { bd = d; best = q; } });
      return best;
    };
    const tipFor = q => `<b>${esc(q.p.name)}</b> · ${esc(tShort(q.p.team))} · ${q.p.pos} · ${eurP(q.p)}<br>${esc(METRICS[lab.y].l)} <b>${mfmt(lab.y, q.vy)}</b><br>${esc(METRICS[lab.x].l)} <b>${mfmt(lab.x, q.vx)}</b>`;
    svg.addEventListener("pointermove", e => {
      const q = nearest(e);
      if (q) { svg.setAttribute("data-tip", tipFor(q)); if (labSel !== q.p.id) { labSel = q.p.id; markLab(); } }
      else svg.removeAttribute("data-tip");
    });
    svg.addEventListener("pointerleave", () => svg.removeAttribute("data-tip"));
    svg.addEventListener("click", e => {
      const q = nearest(e);
      if (!q) return;
      if (e.pointerType === "mouse" || labSel === q.p.id && svg.dataset.armed === q.p.id) { openPlayer(q.p.id); return; }
      labSel = q.p.id; svg.dataset.armed = q.p.id; markLab();
    });
  }
  function markLab() {
    const ring = $("#lab-ring"), q = labPts.find(x => x.p.id === labSel);
    if (ring) { if (q) { ring.setAttribute("cx", q.cx); ring.setAttribute("cy", q.cy); ring.removeAttribute("visibility"); } else ring.setAttribute("visibility", "hidden"); }
    const pre = PRESETS.find(p => p[1] === lab.x && p[2] === lab.y);
    $("#lab-read").innerHTML = q
      ? `<b>${esc(q.p.name)}</b> (${esc(tShort(q.p.team))}, ${q.p.pos}, ${eurP(q.p)}): ${esc(METRICS[lab.y].l.toLowerCase())} <b>${mfmt(lab.y, q.vy)}</b>, ${esc(METRICS[lab.x].l.toLowerCase())} <b>${mfmt(lab.x, q.vx)}</b>. <button class="plink" data-open="${q.p.id}"><u>Open his profile</u></button>`
      : (pre ? esc(pre[3]) : "Hover or tap a dot to read it.") + ` ${labPts.length} players shown; the Players tab has the same numbers as a table.`;
    const btn = $("#lab-read [data-open]");
    if (btn) btn.addEventListener("click", () => openPlayer(btn.dataset.open));
  }
  function renderLab() {
    store.set("lab", lab);
    $("#lab-x").value = lab.x; $("#lab-y").value = lab.y; $("#lab-starters").checked = !!lab.starters; $("#lab-mine").checked = !!lab.mine;
    const mine = new Set(!isExample && Array.isArray(squad) ? squad : []);
    $("#lab-mine-l").hidden = mine.size === 0;
    $$("#lab-presets button").forEach(b => { const p = PRESETS[+b.dataset.i]; b.setAttribute("aria-pressed", p[1] === lab.x && p[2] === lab.y); });
    $$("#lab-pos button").forEach(b => b.setAttribute("aria-pressed", b.dataset.p === lab.pos));
    const W = 960, H = 540, m = { l: 54, r: 20, t: 14, b: 46 };
    const find = $("#lab-find").value.trim().toLowerCase();
    const rows = mdPlayers().filter(p => (lab.pos === "ALL" || p.pos === lab.pos) && (!lab.starters || p.p_start_sim >= 0.6))
      .map(p => ({ p, vx: mval(p, lab.x), vy: mval(p, lab.y) })).filter(q => q.vx != null && q.vy != null && isFinite(q.vx) && isFinite(q.vy));
    const svg = $("#lab-svg");
    if (!rows.length) { svg.innerHTML = `<text x="${W / 2}" y="${H / 2}" text-anchor="middle">No players have both numbers yet. Try another pair or untick "Likely starters only".</text>`; labPts = []; labGeo = null; markLab(); return; }
    const ext = (a, pc) => { let lo = Math.min(...a), hi = Math.max(...a); const pad = (hi - lo || 1) * 0.05; lo = pc || lo >= 0 && lo - pad < 0 ? Math.max(0, lo - pad) : lo - pad; return [lo, hi + pad]; };
    const [x0, x1] = ext(rows.map(q => q.vx), METRICS[lab.x].pc), [y0, y1] = ext(rows.map(q => q.vy), METRICS[lab.y].pc);
    const sx = lin(x0, x1, m.l, W - m.r), sy = lin(y0, y1, H - m.b, m.t);
    rows.forEach(q => { q.cx = +sx(q.vx).toFixed(1); q.cy = +sy(q.vy).toFixed(1); q.hit = find && (q.p.name.toLowerCase().includes(find) || tShort(q.p.team).toLowerCase().includes(find)); });
    const order = { GK: 0, DEF: 1, MID: 2, FWD: 3 };
    rows.sort((a, c) => order[a.p.pos] - order[c.p.pos] || a.vy - c.vy);
    const mx = median(rows.map(q => q.vx)), my = median(rows.map(q => q.vy));
    let g = axes(null, lab.x, lab.y, x0, x1, y0, y1, sx, sy, W, H, m);
    g += `<line class="ref" x1="${sx(mx).toFixed(1)}" x2="${sx(mx).toFixed(1)}" y1="${m.t}" y2="${H - m.b}"/><line class="ref" x1="${m.l}" x2="${W - m.r}" y1="${sy(my).toFixed(1)}" y2="${sy(my).toFixed(1)}"/>`;
    g += `<text class="ttl" x="${W - m.r}" y="${H - 8}" text-anchor="end">${esc(METRICS[lab.x].l)} →</text><text class="ttl" x="${m.l}" y="${m.t - 2}" text-anchor="start" dy="-0">↑ ${esc(METRICS[lab.y].l)}</text>`;
    g += rows.map(q => `<circle class="dot pc-${q.p.pos}${find && !q.hit ? " dim" : ""}" cx="${q.cx}" cy="${q.cy}" r="${q.hit ? 6.5 : 5}"/>`).join("");
    if (lab.mine) g += rows.filter(q => mine.has(q.p.id)).map(q => `<circle cx="${q.cx}" cy="${q.cy}" r="9" fill="none" stroke="var(--accent)" stroke-width="2.5"/>`).join("");
    // labels: searched players, then my team, then the extremes on both axes
    const seen = new Set(), cand = [];
    const add = q => { if (!seen.has(q.p.id)) { seen.add(q.p.id); cand.push({ x: q.cx, y: q.cy, text: shortName(q.p.name) }); } };
    if (find) rows.filter(q => q.hit).slice(0, 25).forEach(add);
    if (lab.mine) rows.filter(q => mine.has(q.p.id)).forEach(add);
    rows.slice().sort((a, c) => c.vy - a.vy).slice(0, 14).forEach(add);
    rows.slice().sort((a, c) => c.vx - a.vx).slice(0, 6).forEach(add);
    if (lab.x === "own") rows.slice().sort((a, c) => a.vx - c.vx).filter(q => q.vy >= my).slice(0, 5).forEach(add);
    g += placeLabels(cand, W, H);
    g += `<circle id="lab-ring" class="ring" r="9" visibility="hidden"/>`;
    svg.innerHTML = g;
    labPts = rows; labGeo = { W, H };
    if (labSel && !rows.some(q => q.p.id === labSel)) labSel = null;
    $("#lab-legend").innerHTML = ["DEF", "MID", "FWD", "GK"].filter(p => lab.pos === "ALL" || lab.pos === p).map(p => `<span><i style="background:var(--c-${p.toLowerCase()})"></i>${({ DEF: "Defenders", MID: "Midfielders", FWD: "Forwards", GK: "Goalkeepers" })[p]}</span>`).join("")
      + `<span class="muted">Dashed lines: the median of each axis</span>` + (lab.mine ? `<span><i style="background:none;border:2.5px solid var(--accent)"></i>Your team</span>` : "");
    markLab();
  }

  // ---------------------------------------------------------------- head to head
  let h2h = store.get("h2h", null) || [];
  const byName = {};
  function setupH2H() {
    const top = mdPlayers().slice().sort((a, c) => c.xpts - a.xpts);
    top.forEach(p => { const k = `${p.name} (${tShort(p.team)})`; byName[k.toLowerCase()] = p.id; });
    $("#h2h-list").innerHTML = top.slice(0, 600).map(p => `<option value="${esc(`${p.name} (${tShort(p.team)})`)}"></option>`).join("");
    if (h2h.length !== 2 || !P[h2h[0]] || !P[h2h[1]]) h2h = [top[0].id, top[1].id];
    const lab2 = id => `${P[id].name} (${tShort(P[id].team)})`;
    $("#h2h-a").value = lab2(h2h[0]); $("#h2h-b").value = lab2(h2h[1]);
    [["#h2h-a", 0], ["#h2h-b", 1]].forEach(([sel, i]) => $(sel).addEventListener("change", e => {
      const v = e.target.value.trim().toLowerCase();
      const id = byName[v] || (Object.keys(byName).find(k => k.startsWith(v)) ? byName[Object.keys(byName).find(k => k.startsWith(v))] : null);
      if (id) { h2h[i] = id; store.set("h2h", h2h); e.target.value = lab2(id); renderH2H(); }
    }));
  }
  function renderH2H() {
    const [a, b] = h2h.map(id => mdPlayers().find(p => p.id === id));
    if (!a || !b) { $("#h2h-out").innerHTML = `<p class="fine">Pick two players above.</p>`; return; }
    const rowsK = ["xpts", "xpts_if_start", "p_start_sim", "p_goal", "p_assist", "p_cs", "xrec", "p_haul", "p90", "value", "next3", "season"];
    const head = p => `<div><span class="plink" data-pid="${p.id}" tabindex="0">${esc(shortName(p.name))}</span><small>${esc(tShort(p.team))} · ${p.pos} · ${eurP(p)}${mval(p, "own") != null ? ` · ${f1(mval(p, "own"))}% own` : ""}</small></div>`;
    const rows = rowsK.map(k => {
      const va = mval(a, k), vb = mval(b, k);
      if (va == null && vb == null) return "";
      const mx = Math.max(va || 0, vb || 0, 1e-9), wa = Math.max(0, (va || 0) / mx * 100), wb = Math.max(0, (vb || 0) / mx * 100);
      return `<div class="h2h-row"><div class="a${va > vb ? " win" : ""}"><b>${mfmt(k, va)}</b><span class="hbar" style="width:${wa.toFixed(0)}%;max-width:calc(100% - 56px)"></span></div><div class="m">${esc(METRICS[k].l)}</div><div class="b${vb > va ? " win" : ""}"><b>${mfmt(k, vb)}</b><span class="hbar" style="width:${wb.toFixed(0)}%;max-width:calc(100% - 56px)"></span></div></div>`;
    }).join("");
    const wins = rowsK.reduce((n, k) => { const va = mval(a, k), vb = mval(b, k); return va == null || vb == null || va === vb ? n : [n[0] + (va > vb), n[1] + (vb > va)]; }, [0, 0]);
    $("#h2h-out").innerHTML = `<div class="h2h-head">${head(a)}<div></div>${head(b)}</div><div class="h2h" style="margin-top:12px">${rows}</div>
      <p class="read" style="margin-top:12px">${esc(shortName(a.name))} leads ${wins[0]} rows, ${esc(shortName(b.name))} ${wins[1]}, for MD${MD}. Price and ownership are shown for context and do not count.</p>`;
  }

  // ---------------------------------------------------------------- club power map
  function renderClubMap() {
    const W = 960, H = 520, m = { l: 54, r: 20, t: 14, b: 46 };
    const st = Object.fromEntries((D.standings || []).map(r => [r.team, r]));
    const rows = Object.keys(T).map(t => ({ t, a: T[t].attack, d: T[t].defence })).filter(r => r.a != null && r.d != null);
    if (!rows.length) { $("#clubmap").hidden = true; return; }
    const pad = (a, f) => { const lo = Math.min(...a), hi = Math.max(...a), p = (hi - lo || 1) * f; return [lo - p, hi + p]; };
    const [x0, x1] = pad(rows.map(r => r.a), 0.07), [y0, y1] = pad(rows.map(r => r.d), 0.07);
    const sx = lin(x0, x1, m.l, W - m.r), sy = lin(y0, y1, H - m.b, m.t);
    let g = axes(null, null, null, x0, x1, y0, y1, sx, sy, W, H, m);
    g += `<text class="ttl" x="${W - m.r}" y="${H - 8}" text-anchor="end">Attack rating →</text><text class="ttl" x="${m.l}" y="${m.t - 2}">↑ Defence rating</text>`;
    rows.forEach(r => { r.cx = +sx(r.a).toFixed(1); r.cy = +sy(r.d).toFixed(1); r.s = st[r.t]; });
    g += rows.map(r => `<circle class="dot" tabindex="0" data-club="${r.t}" cx="${r.cx}" cy="${r.cy}" r="7" fill="var(--s1)" data-tip="${esc(`<b>${esc(T[r.t].name)}</b><br>attack ${f2(r.a)} · defence ${f2(r.d)}${r.s ? `<br>top 8: ${pct(r.s.p_top8)} · expected rank ${f1(r.s.exp_rank)}` : ""}`)}"/>`).join("");
    g += placeLabels(rows.slice().sort((a, c) => (c.a + c.d) - (a.a + a.d)).map(r => ({ x: r.cx, y: r.cy, text: tShort(r.t) })), W, H);
    const svg = $("#cm-svg");
    svg.innerHTML = g;
    $$("[data-club]", svg).forEach(c => c.addEventListener("click", () => { $("#pl-team").value = c.dataset.club; $("#pl-team").dispatchEvent(new Event("change")); location.hash = "#players"; }));
    const bestA = rows.slice().sort((a, c) => c.a - a.a)[0], bestD = rows.slice().sort((a, c) => c.d - a.d)[0];
    $("#cm-read").innerHTML = `Best attack: <b>${esc(T[bestA.t].name)}</b> (${f2(bestA.a)}). Best defence: <b>${esc(T[bestD.t].name)}</b> (${f2(bestD.d)}). Attackers from the right-hand side and defenders from the top are the ones to own; the fixture decides the rest.`;
  }

  // ---------------------------------------------------------------- fixture ticker
  let tkMode = store.get("tk", "att") === "def" ? "def" : "att";
  function renderTicker() {
    const mds = MDS.filter(m => m >= MD && D.matchdays[m]);
    $("#tk-mode").innerHTML = [["att", "For attackers"], ["def", "For defenders"]].map(([k, l]) => `<button data-k="${k}" aria-pressed="${k === tkMode}">${l}</button>`).join("");
    $$("#tk-mode button").forEach(b => b.addEventListener("click", () => { tkMode = b.dataset.k; store.set("tk", tkMode); renderTicker(); }));
    const st = Object.fromEntries((D.standings || []).map(r => [r.team, r]));
    const rows = Object.keys(T).map(t => {
      const cells = mds.map(m => { const f = fixtureOf(t, m); return f ? { opp: f.opp, home: f.home, v: tkMode === "att" ? (f.home ? f.f.xg_home : f.f.xg_away) : (f.home ? f.f.cs_home : f.f.cs_away) } : null; });
      return { t, cells, tot: cells.reduce((a, c) => a + (c ? c.v : 0), 0), n: cells.filter(Boolean).length };
    }).filter(r => r.n).sort((a, c) => c.tot - a.tot);
    const all = rows.flatMap(r => r.cells.filter(Boolean).map(c => c.v)), lo = Math.min(...all), hi = Math.max(...all);
    const fmt = v => tkMode === "att" ? f2(v) : pct(v);
    const head = `<tr><th>Club</th>${mds.map(m => `<th>MD${m} · ${esc(D.matchdays[m].deadline_ist.split(",")[0].replace(/ \d{4}$/, ""))}</th>`).join("")}<th>${tkMode === "att" ? "Goals, total" : "Clean sheets, total"}</th></tr>`;
    const body = rows.map(r => `<tr><td class="club" data-club="${r.t}">${esc(tShort(r.t))}${st[r.t] ? `<small>top 8 ${pct(st[r.t].p_top8)}</small>` : ""}</td>${r.cells.map(c => c
      ? `<td><div class="c"><i style="opacity:${(0.06 + 0.5 * (c.v - lo) / ((hi - lo) || 1)).toFixed(2)}"></i><span>${c.home ? "v" : "at"} ${esc(tShort(c.opp))}</span><b>${fmt(c.v)}</b></div></td>`
      : `<td><div class="c"><span class="muted">no match</span></div></td>`).join("")}<td class="tot">${tkMode === "att" ? f1(r.tot) : f2(r.tot)}</td></tr>`).join("");
    $("#tk-table").innerHTML = `<thead>${head}</thead><tbody>${body}</tbody>`;
    $("#tk-note").textContent = tkMode === "att" ? "Cell: the club's expected goals in that match." : "Cell: the club's chance of a clean sheet in that match.";
    $$("#tk-table [data-club]").forEach(c => c.addEventListener("click", () => { $("#pl-team").value = c.dataset.club; $("#pl-team").dispatchEvent(new Event("change")); location.hash = "#players"; }));
  }

  // ---------------------------------------------------------------- goals against the pre-deadline forecast, by club
  function renderHotCold() {
    const agg = {}, mds = [];
    (D.graded || []).forEach(g => { let used = false; (g.rows || []).forEach(r => {
      const m = /^(\d+)-(\d+)$/.exec(r.score || "");
      if (!m || !r.xg) return;
      used = true;
      [[r.home, +m[1], r.xg[0], +m[2], r.xg[1]], [r.away, +m[2], r.xg[1], +m[1], r.xg[0]]].forEach(([t, gf, xf, ga, xa]) => {
        const a = agg[t] || (agg[t] = { n: 0, gf: 0, xf: 0, ga: 0, xa: 0 });
        a.n++; a.gf += gf; a.xf += xf; a.ga += ga; a.xa += xa;
      });
    }); if (used) mds.push(g.md); });
    const teams = Object.keys(agg).filter(t => T[t]);
    if (teams.length < 8) { $("#hotcold").hidden = true; return; }
    $("#hotcold").hidden = false;
    const draw = (id, got, exp, verb) => {
      const rows = teams.map(t => ({ t, d: agg[t][got] - agg[t][exp], a: agg[t] })).sort((x, y) => y.d - x.d);
      const W = 460, rh = 17, top = 20, L = 96, R = 44, H = top + rows.length * rh + 6, mx = Math.max(1, ...rows.map(r => Math.abs(r.d))), c = L + (W - L - R) / 2, sc = (W - L - R) / 2 / mx;
      let g = `<line class="ax" x1="${c}" x2="${c}" y1="${top - 4}" y2="${H - 4}"/><text x="${c}" y="11" text-anchor="middle">as expected</text>`;
      rows.forEach((r, i) => {
        const y = top + i * rh, w = Math.max(Math.abs(r.d) * sc, 1), x = r.d >= 0 ? c : c - w;
        g += `<text x="${L - 8}" y="${y + 12}" text-anchor="end">${esc(tShort(r.t))}</text>
          <rect x="${x.toFixed(1)}" y="${y + 3}" width="${w.toFixed(1)}" height="${rh - 6}" rx="3" fill="var(${r.d >= 0 ? "--s2" : "--s1"})" data-tip="${esc(`<b>${esc(T[r.t].name)}</b><br>${verb} ${r.a[got]}, expected ${f1(r.a[exp])}<br>${r.a.n} matches`)}"/>
          <text class="lab" x="${(r.d >= 0 ? c + w + 5 : c - w - 5).toFixed(1)}" y="${y + 12}" text-anchor="${r.d >= 0 ? "start" : "end"}">${r.d >= 0 ? "+" : "−"}${f1(Math.abs(r.d))}</text>`;
      });
      const svg = $(id); svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.innerHTML = g;
    };
    draw("#hc-att", "gf", "xf", "scored"); draw("#hc-def", "ga", "xa", "conceded");
    const n = teams.reduce((a, t) => a + agg[t].n, 0) / 2;
    $("#hc-note").textContent = `Matchday${mds.length > 1 ? "s" : ""} ${mds.sort((a, b) => a - b).join(", ")}: ${n} matches, forecasts as saved at each deadline.`;
  }

  // ---------------------------------------------------------------- reliability
  function renderReliability() {
    const R = D.reliability || {}, M = R.models || {};
    const names = Object.keys(M);
    if (!names.length) { $("#reliability").hidden = true; return; }
    $("#reliability").hidden = false;
    const label = { statistical: "Statistical", network: "Neural network", bookmaker: "Bookmaker", trees: "Decision trees", combined: "Combined (feeds the picks)" };
    const sel = $("#rel-model");
    if (!sel.options.length) {
      sel.innerHTML = names.map(n => `<option value="${n}">${esc(label[n] || n)}</option>`).join("");
      sel.value = names.includes((D.models || {}).picks_model) ? D.models.picks_model : names[0];
      sel.addEventListener("change", renderReliability);
    }
    const d = M[sel.value] || {};
    $("#rel-note").textContent = `Matchdays ${(d.mds || []).join(", ")}, rebuilt with pre-deadline inputs.`;
    const W = 320, H = 300, m = { l: 42, r: 12, t: 12, b: 40 };
    const plot = (id, bins, hi, fmt, xl, yl) => {
      const pts = (bins || []).filter(b => b[0] > 0).map(b => ({ n: b[0], x: b[1] / b[0], y: b[2] / b[0] }));
      const top = hi || Math.max(1, ...pts.map(p => Math.max(p.x, p.y))) * 1.08;
      const sx = lin(0, top, m.l, W - m.r), sy = lin(0, top, H - m.b, m.t), nmax = Math.max(1, ...pts.map(p => p.n));
      let g = "";
      ticks(0, top, 4).forEach(v => { g += `<line class="gr" x1="${m.l}" x2="${W - m.r}" y1="${sy(v)}" y2="${sy(v)}"/><text x="${m.l - 6}" y="${sy(v) + 4}" text-anchor="end">${fmt(v)}</text><text x="${sx(v)}" y="${H - m.b + 15}" text-anchor="middle">${fmt(v)}</text>`; });
      g += `<line class="ref" x1="${sx(0)}" y1="${sy(0)}" x2="${sx(top)}" y2="${sy(top)}"/><line class="ax" x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}"/><line class="ax" x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${H - m.b}"/>`;
      g += `<text class="ttl" x="${W - m.r}" y="${H - 6}" text-anchor="end">${xl} →</text><text class="ttl" x="${m.l}" y="${m.t - 1}">↑ ${yl}</text>`;
      g += pts.map(p => `<circle class="dot" cx="${sx(p.x).toFixed(1)}" cy="${sy(p.y).toFixed(1)}" r="${(4 + 7 * Math.sqrt(p.n / nmax)).toFixed(1)}" fill="var(--s1)" fill-opacity=".85" data-tip="${esc(`forecast <b>${fmt(p.x)}</b> · happened <b>${fmt(p.y)}</b><br>${p.n} player-matches`)}"/>`).join("");
      $(id).innerHTML = g;
      return pts;
    };
    const ps = plot("#rel-start", d.cal_start, 1, v => Math.round(v * 100) + "%", "forecast", "started");
    const n = ps.reduce((a, p) => a + p.n, 0), gap = n ? ps.reduce((a, p) => a + p.n * Math.abs(p.x - p.y), 0) / n : 0;
    $("#rel-start-n").innerHTML = ps.length ? `${n.toLocaleString("en-GB")} player-matches in ${ps.length} groups; bigger dots hold more. Average gap between forecast and outcome: <b>${(gap * 100).toFixed(1)} points of probability</b>.` : "No line-ups graded yet.";
    const pp = plot("#rel-points", d.cal_points, 0, v => String(+v.toFixed(1)), "expected", "scored");
    const hiG = pp.length ? pp[pp.length - 1] : null;
    $("#rel-points-n").innerHTML = pp.length ? `UEFA's official points against the forecast. ${hiG && hiG.y > hiG.x ? `The top group scored <b>${f1(hiG.y)}</b> against <b>${f1(hiG.x)}</b> expected: the best players have beaten their forecast so far.` : "Groups sit close to the line."}` : "No official points on file yet.";
    // starters by position: dumbbell, expected v official per start
    const pos = d.pos_points || {};
    const pr = POS.filter(k => pos[k] && pos[k][0] > 0).map(k => ({ k, n: pos[k][0], e: pos[k][1] / pos[k][0], a: pos[k][2] / pos[k][0] }));
    if (pr.length) {
      const top = Math.max(...pr.map(r => Math.max(r.e, r.a))) * 1.15, sxx = lin(0, top, 64, W - 16), rowH = (H - 60) / pr.length;
      let g = "";
      ticks(0, top, 4).forEach(v => { g += `<line class="gr" x1="${sxx(v)}" x2="${sxx(v)}" y1="14" y2="${H - 40}"/><text x="${sxx(v)}" y="${H - 24}" text-anchor="middle">${+v.toFixed(1)}</text>`; });
      pr.forEach((r, i) => {
        const y = 14 + rowH * (i + 0.5);
        g += `<text x="56" y="${y + 4}" text-anchor="end" class="ttl">${r.k}</text><line x1="${sxx(r.e)}" x2="${sxx(r.a)}" y1="${y}" y2="${y}" stroke="var(--line-2)" stroke-width="3"/>
          <circle cx="${sxx(r.e).toFixed(1)}" cy="${y}" r="6" fill="var(--panel)" stroke="var(--s1)" stroke-width="2.5" data-tip="${r.k}: expected ${f2(r.e)} per start"/><circle class="dot" cx="${sxx(r.a).toFixed(1)}" cy="${y}" r="6.5" fill="var(--s1)" data-tip="${r.k}: scored ${f2(r.a)} per start (${r.n} starts)"/>
          <text class="lab" x="${(Math.max(sxx(r.e), sxx(r.a)) + 12).toFixed(1)}" y="${y + 4}">${f1(r.e)} → ${f1(r.a)}</text>`;
      });
      g += `<text class="ttl" x="${W - 16}" y="${H - 6}" text-anchor="end">points per start →</text>`;
      $("#rel-pos").innerHTML = g;
      $("#rel-pos-n").innerHTML = `Hollow dot: what the model expected of players who started. Solid dot: what they scored.`;
    } else { $("#rel-pos").innerHTML = ""; $("#rel-pos-n").textContent = "No starters graded yet."; }
  }

  // ---------------------------------------------------------------- player profile
  const dlg = $("#pp");
  const PARTS = [["Appearance", "--s0"], ["Goals", "--s1"], ["Assists", "--s2"], ["Clean sheet", "--s3"], ["Recoveries", "--s4"], ["Award", "--s5"]];
  const FLAG_OUT = ["Injured", "Suspended", "Not in squad", "I", "S", "NIS"];
  function fixtureOf(team, md) {
    const b = D.matchdays[md], f = b && b.fixtures.find(x => x.home === team || x.away === team);
    return f ? { opp: f.home === team ? f.away : f.home, home: f.home === team, date: f.date, f } : null;
  }
  function pmfChart(p) {
    if (!p.pmf) return `<p class="fine">No spread shown for players under 1 expected point.</p>`;
    const W = 460, H = 190, m = { l: 8, r: 8, t: 16, b: 26 }, n = p.pmf.length, bw = (W - m.l - m.r) / n, mxv = Math.max(...p.pmf);
    const sy = lin(0, mxv, H - m.b, m.t);
    let g = `<line class="ax" x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}"/>`;
    p.pmf.forEach((v, i) => {
      const pts = i - 2, x = m.l + i * bw, h = H - m.b - sy(v), col = pts <= 2 ? "var(--s0)" : pts >= 10 ? "var(--accent)" : "var(--s1)";
      if (v > 0) g += `<rect x="${(x + 1).toFixed(1)}" y="${sy(v).toFixed(1)}" width="${(bw - 2).toFixed(1)}" height="${Math.max(h, 1).toFixed(1)}" rx="2" fill="${col}"><title>${pts === 20 ? "20 or more" : pts} points: ${(v / 10).toFixed(1)}%</title></rect>`;
      if (pts % 5 === 0) g += `<text x="${(x + bw / 2).toFixed(1)}" y="${H - 9}" text-anchor="middle">${pts === 20 ? "20+" : pts}</text>`;
    });
    const mxX = m.l + (Math.min(Math.max(p.xpts, -2), 20) + 2 + 0.5) * bw;
    g += `<line class="ref" x1="${mxX.toFixed(1)}" x2="${mxX.toFixed(1)}" y1="${m.t - 4}" y2="${H - m.b}"/><text class="lab" x="${mxX.toFixed(1)}" y="${m.t - 6}" text-anchor="${mxX > W * 0.7 ? "end" : "start"}" dx="${mxX > W * 0.7 ? -4 : 4}">average ${f1(p.xpts)}</text>`;
    return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Chance of each points total">${g}</svg>
      <div class="key"><span><i class="sq" style="background:var(--s0)"></i>2 or fewer: <b>${pct(p.p_blank)}</b></span><span><i class="sq" style="background:var(--s1)"></i>3 to 9: <b>${pct(Math.max(0, 1 - p.p_blank - p.p_haul))}</b></span><span><i class="sq" style="background:var(--accent)"></i>10 or more: <b>${pct(p.p_haul)}</b></span></div>`;
  }
  function partsChart(p) {
    if (!p.c) return `<p class="fine">Not enough expected minutes to break his points down.</p>`;
    const pos = p.c.slice(0, 6).map(v => Math.max(v, 0)), tot = pos.reduce((a, c) => a + c, 0) || 1, other = p.c[6];
    return `<div class="parts" role="img" aria-label="Expected points by source">${pos.map((v, i) => v > 0.004 ? `<span style="width:${(v / tot * 100).toFixed(1)}%;background:var(${PARTS[i][1]})" title="${PARTS[i][0]}: ${f2(v)}"></span>` : "").join("")}</div>
      <div class="parts-key">${PARTS.map((q, i) => `<div><i style="background:var(${q[1]})"></i>${q[0]}<b>${f2(p.c[i])}</b></div>`).join("")}<div><i style="background:none;border:1px dashed var(--muted)"></i>Everything else<b>${other >= 0 ? "+" : "−"}${f2(Math.abs(other))}</b></div></div>
      <p class="fine">${p.pos === "GK" ? "Everything else is saves and penalty saves, less goals conceded and cards." : p.pos === "DEF" ? "Everything else is goals conceded and cards, less penalties won." : "Everything else is cards, missed penalties and penalties won."}</p>`;
  }
  function historyChart(id) {
    const s = SEASON[id];
    if (!s || !S_MDS.length) return `<p class="fine">No Champions League minutes on file this season.</p>`;
    const W = 460, H = 190, m = { l: 30, r: 10, t: 18, b: 26 }, n = S_MDS.length, vals = s.p.concat(s.x || []).filter(v => v != null);
    const top = Math.max(6, ...vals) * 1.15, lo = Math.min(0, ...vals), sy = lin(lo, top, H - m.b, m.t), bw = (W - m.l - m.r) / n;
    let g = "";
    ticks(lo, top, 4).forEach(v => { g += `<line class="gr" x1="${m.l}" x2="${W - m.r}" y1="${sy(v).toFixed(1)}" y2="${sy(v).toFixed(1)}"/><text x="${m.l - 6}" y="${(sy(v) + 4).toFixed(1)}" text-anchor="end">${v}</text>`; });
    g += `<line class="ax" x1="${m.l}" x2="${W - m.r}" y1="${sy(0).toFixed(1)}" y2="${sy(0).toFixed(1)}"/>`;
    S_MDS.forEach((md, i) => {
      const a = s.p[i], x = s.x ? s.x[i] : null, cx = m.l + (i + 0.5) * bw, w = Math.min(46, bw * 0.5);
      if (a != null) { const y0 = sy(Math.max(a, 0)), h = Math.abs(sy(a) - sy(0)); g += `<rect x="${(cx - w / 2).toFixed(1)}" y="${y0.toFixed(1)}" width="${w.toFixed(1)}" height="${Math.max(h, 1).toFixed(1)}" rx="3" fill="var(--s1)"><title>MD${md}: scored ${a}</title></rect><text class="lab" x="${cx.toFixed(1)}" y="${(sy(Math.max(a, 0)) - 5).toFixed(1)}" text-anchor="middle">${a}</text>`; }
      if (x != null) g += `<line x1="${(cx - w / 2 - 7).toFixed(1)}" x2="${(cx + w / 2 + 7).toFixed(1)}" y1="${sy(x).toFixed(1)}" y2="${sy(x).toFixed(1)}" stroke="var(--ink)" stroke-width="2.5" stroke-linecap="round"><title>MD${md}: expected ${f1(x)} before the deadline</title></line>`;
      g += `<text x="${cx.toFixed(1)}" y="${H - 8}" text-anchor="middle">MD${md}</text>`;
    });
    const C = D.season.cols, v = k => s.s[C.indexOf(k)];
    return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Points by matchday">${g}</svg>
      <div class="key"><span><i class="sq" style="background:var(--s1)"></i>Official points</span>${s.x ? `<span><i class="sq" style="background:var(--ink);height:3px;width:16px;border-radius:2px;vertical-align:3px"></i>Forecast before the deadline</span>` : ""}</div>
      <div class="statline"><span><b>${v("points")}</b> points</span><span><b>${v("mins")}</b> minutes</span><span><b>${v("goals")}</b> goals</span><span><b>${v("assists")}</b> assists</span><span><b>${v("cs")}</b> clean sheets</span><span><b>${v("recoveries")}</b> recoveries</span>${v("saves") ? `<span><b>${v("saves")}</b> saves</span>` : ""}<span><b>${v("motm")}</b> match awards</span>${v("yc") || v("rc") ? `<span><b>${v("yc")}</b> yellow, <b>${v("rc")}</b> red</span>` : ""}</div>`;
  }
  function percentiles(p) {
    const peers = mdPlayers().filter(q => q.pos === p.pos && q.p_start_sim >= 0.5);
    if (peers.length < 8) return "";
    const keys = p.pos === "GK" ? ["xpts", "p_cs", "p_start_sim", "value", "p90"] : ["xpts", "xgoals", "xassists", "xrec", "p_cs", "p_start_sim", "value", "p90"].filter(k => !(p.pos === "FWD" && k === "p_cs"));
    const short = { xpts: "Expected points", xgoals: "Goal threat", xassists: "Creativity", xrec: "Recoveries", p_cs: "Clean sheet", p_start_sim: "Start security", value: "Value", p90: "Ceiling" };
    return `<div class="pctl">${keys.map(k => {
      const v = p[k], below = peers.filter(q => q[k] < v).length, pc = Math.round(below / peers.length * 100);
      return `<div><span>${short[k]}</span><span class="tr"><i style="width:${pc}%"></i><u></u></span><b>${pc}</b></div>`;
    }).join("")}</div><p class="fine">Percentile among the ${peers.length} ${({ GK: "goalkeepers", DEF: "defenders", MID: "midfielders", FWD: "forwards" })[p.pos]} likely to start this matchday: 50 is the middle one, 99 the best.</p>`;
  }
  function openPlayer(id) {
    const p0 = P[id];
    if (!p0) return;
    const p = mdPlayers().find(x => x.id === id) || Object.assign({ xpts: 0 }, p0);
    const s = SEASON[id] || {}, own = s.own != null ? s.own : p0.own;
    const st = p0.st && !["Available", "None", ""].includes(String(p0.st)) ? String(p0.st) : (s.flag || "");
    const stLab = ({ I: "Injured", D: "Doubtful", S: "Suspended", NIS: "Not in squad" })[st] || st;
    const same = mdPlayers().filter(q => q.pos === p0.pos).sort((a, c) => c.xpts - a.xpts), rank = same.findIndex(q => q.id === id) + 1;
    const fx = fixtureOf(p0.team, MD);
    const kpi = (l, v, d, big) => `<div class="kpi${big ? " k1" : ""}"><span class="l">${l}</span><span class="v">${v}</span>${d ? `<span class="d">${d}</span>` : ""}</div>`;
    const next = MDS.map((m, i) => { const f = fixtureOf(p0.team, m), e = (D.ev[id] || [])[i]; return `<div class="${m === MD ? "on" : ""}"><small>MD${m}${f ? " · " + dayName(f.date) : ""}</small><b>${e != null ? f1(e) : "–"}</b><span>${f ? (f.home ? "v " : "at ") + esc(tShort(f.opp)) : "no match"}</span></div>`; }).join("");
    dlg.innerHTML = `<div class="sheet-head"><div><h2 id="pp-name">${esc(p0.name)}</h2>
        <div class="sub"><span class="pos-tag" style="margin:0">${p0.pos}</span><span>${esc((T[p0.team] || {}).name || p0.team)}</span><span>${eurP(p0)}</span>${own != null ? `<span>owned by ${f1(own)}%</span>` : ""}${stLab ? `<span class="flag${FLAG_OUT.includes(st) ? " out" : ""}">${esc(stLab)}</span>` : ""}</div>
        <div class="acts"><button class="btn" id="pp-add"></button><button class="btn" id="pp-h2h">Compare with another player</button></div></div>
        <button class="sheet-x" id="pp-x" aria-label="Close">×</button></div>
      <div class="sheet-body">
        <div class="kpis">
          ${kpi(`Expected points, MD${MD}${fx ? ` (${fx.home ? "v" : "at"} ${esc(tShort(fx.opp))})` : ""}`, f1(p.xpts), p.xpts_if_start != null ? `${f1(p.xpts_if_start)} if he starts · ${rank ? `#${rank} of ${same.length} ${p0.pos}` : ""}` : "", true)}
          ${kpi("Start chance", pct(p.p_start_sim), p.p_play != null ? `plays at all ${pct(p.p_play)}` : "")}
          ${kpi("Goal", pct(p.p_goal), p.xgoals != null ? `${f2(p.xgoals)} expected` : "")}
          ${kpi("Assist", pct(p.p_assist), p.xassists != null ? `${f2(p.xassists)} expected` : "")}
          ${p0.pos === "FWD" ? kpi("Recoveries", f1(p.xrec), "1 point per 3") : kpi("Clean sheet", pct(p.p_cs), p0.pos === "MID" ? "worth 1 point" : "worth 4 points")}
          ${kpi("10+ points", pct(p.p_haul), p.p90 != null ? `range ${f1(p.p10)} to ${f1(p.p90)}` : "")}
        </div>
        <div class="pp-grid">
          <div class="card"><h3>Chance of each score</h3>${pmfChart(p)}</div>
          <div class="card"><h3>Where the points come from</h3>${partsChart(p)}</div>
          <div class="card"><h3>Season so far</h3>${historyChart(id)}</div>
          <div class="card"><h3>Against his position</h3>${percentiles(p) || `<p class="fine">Too few likely starters to rank against.</p>`}</div>
        </div>
        <div class="card"><h3>Next matchdays</h3><div class="nextfx">${next}</div>${p.src ? `<p class="fine">Rates from: ${esc(p.src)}.</p>` : ""}</div>
      </div>`;
    $("#pp-x").addEventListener("click", () => dlg.close());
    const addBtn = $("#pp-add");
    const addLabel = () => {
      const mine = !isExample && squad.includes(id), full = !isExample && squad.length >= 15 && !mine;
      addBtn.textContent = mine ? "Remove from my team" : full ? "Your squad is full" : "Add to my team";
      addBtn.disabled = full;
    };
    addLabel();
    addBtn.addEventListener("click", () => {
      if (isExample) { squad = []; isExample = false; }          // the example squad makes way for a real one
      squad = squad.includes(id) ? squad.filter(x => x !== id) : squad.concat([id]);
      saveSquad(); renderTeam(); addLabel(); renderLab();
    });
    $("#pp-h2h").addEventListener("click", () => {
      h2h = [id, h2h[0] === id ? h2h[1] : h2h[0]]; store.set("h2h", h2h);
      const l2 = i => `${P[i].name} (${tShort(P[i].team)})`;
      $("#h2h-a").value = l2(h2h[0]); $("#h2h-b").value = l2(h2h[1]);
      renderH2H(); dlg.close(); location.hash = "#h2h";
    });
    if (!dlg.open) { if (dlg.showModal) dlg.showModal(); else dlg.setAttribute("open", ""); }
    dlg.focus();                                        // keep the Enter that opened it from pressing a button inside
    dlg.scrollTop = 0;
  }
  dlg.addEventListener("click", e => { if (e.target === dlg) dlg.close(); });

  // ---------------------------------------------------------------- guide
  function renderGuide() {
    const gen = new Date(D.meta.generated_at);
    $("#gd-sources").innerHTML = `Prices, ownership, official points and availability flags come from UEFA's own fantasy game feed. Club ratings use open league results; Premier League players' xG and xA come from the Fantasy Premier League public data, other leagues from published season tables. Predicted line-ups and bookmaker prices are read from public pages when they are on file. Built ${gen.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" })}, model ${esc(D.meta.model_version)}, ${Number(D.meta.n_sims).toLocaleString("en-GB")} simulations per match, ${D.meta.pool_size} players. This is a personal project with no link to UEFA.`;
  }

  function setupSearch() {
    const all = D.players_index.slice().sort((a, c) => ((D.ev[c.id] || [0])[0] || 0) - ((D.ev[a.id] || [0])[0] || 0));
    const key = p => `${p.name} (${tShort(p.team)})`;
    const idx = {};
    all.forEach(p => { idx[key(p).toLowerCase()] = p.id; });
    $("#find-list").innerHTML = all.map(p => `<option value="${esc(key(p))}"></option>`).join("");
    const box = $("#find");
    const go = () => {
      const v = box.value.trim().toLowerCase();
      if (!v) return;
      const id = idx[v] || (all.find(p => p.name.toLowerCase() === v) || all.find(p => p.name.toLowerCase().includes(v)) || {}).id;
      if (id) { box.value = ""; box.blur(); openPlayer(id); }
    };
    box.addEventListener("change", go);
    box.addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); go(); } });
  }
  setupLab(); setupH2H(); setupSearch(); renderClubMap(); renderHotCold(); renderReliability(); renderGuide();
  HOOKS.push(renderLab, renderH2H, renderTicker);
  if (PUBLIC) HOOKS.push(() => {                       // the shared edition gets a short notice in place of the price-list audit
    const g = new Date(D.meta.generated_at);
    $("#pool-notice").innerHTML = `<span aria-hidden="true">ⓘ</span><span><b>${Number(D.meta.pool_size).toLocaleString("en-GB")} players, ${Number(D.meta.n_sims).toLocaleString("en-GB")} simulations per match.</b> Updated ${g.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" })}. Prices and ownership are the official game's. Every number here is a probability or an average; the Guide tab says what each one means.</span>`;
  });
