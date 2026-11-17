  // ================================================================ templates (parts/30_templates.js, styles in parts/90_templates.css)
  // A template changes the Matchday hero (how your XI is shown) and the look of every tab. The numbers never change.
  const TEMPLATES = [
    { k: "cards", n: "Collector cards", d: "Every player as a card in his club's colours, graded by expected points. The captain's card is foil." },
    { k: "night", n: "Night match", d: "Your XI on a floodlit pitch, with the top projections scrolling on an LED board." },
    { k: "poster", n: "Matchday poster", d: "Big type on cobalt blue. The numbers are the picture." },
    { k: "market", n: "Market terminal", d: "Players quoted like stocks: a ticker, your positions, buy and sell signals." },
    { k: "board", n: "Tactics board", d: "The original: magnets and marker notes on a coach's whiteboard." }
  ];
  const TPL = () => { const t = document.documentElement.getAttribute("data-skin"); return TEMPLATES.some(x => x.k === t) ? t : "cards"; };
  const reduceMotion = () => window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---------------------------------------------------------------- shirts in club colours (config/kits.yaml)
  const KITS = D.kits || {};
  const lum = hex => { const n = parseInt(String(hex).slice(1), 16), c = [n >> 16, (n >> 8) & 255, n & 255].map(v => { v /= 255; return v <= .03928 ? v / 12.92 : Math.pow((v + .055) / 1.055, 2.4); }); return .2126 * c[0] + .7152 * c[1] + .0722 * c[2]; };
  const contrast = (a, b) => { const x = lum(a), y = lum(b); return (Math.max(x, y) + .05) / (Math.min(x, y) + .05); };
  const inits = n => { const w = String(n).split(" ").filter(Boolean); return (w.length > 1 ? w[0][0] + w[w.length - 1][0] : w[0].slice(0, 2)).toUpperCase(); };
  document.body.insertAdjacentHTML("beforeend", `<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>
    <linearGradient id="kitshade" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#fff" stop-opacity=".30"/><stop offset=".48" stop-color="#fff" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity=".26"/></linearGradient></defs></svg>`);
  function kit(team, initials, gk) {
    let [body, trim] = KITS[team] || ["#7d8a84", "#ffffff"];
    if (gk) [body, trim] = [trim, body];                      // keepers wear the reverse of the outfield shirt
    const ink = contrast(trim, body) >= 2.4 ? trim : (contrast("#ffffff", body) >= contrast("#111111", body) ? "#ffffff" : "#111111");
    const light = lum(body) > .6;
    const shirt = "M30 9 L47 4 Q60 15 73 4 L90 9 L116 31 L101 51 L90 43 L90 108 L30 108 L30 43 L19 51 L4 31 Z";
    return `<svg class="kit" viewBox="0 0 120 112" aria-hidden="true">
      <path d="${shirt}" fill="${body}"${light ? ' stroke="rgba(0,0,0,.22)" stroke-width="1.2"' : ""}/>
      <path d="M4 31 L19 51 L23.5 45 L8.6 25.4 Z M116 31 L101 51 L96.5 45 L111.4 25.4 Z" fill="${trim}"/>
      <path d="M47 4 Q60 15 73 4 L69 2.6 Q60 10.5 51 2.6 Z" fill="${trim}"/>
      <path d="M30 43 L30 100 M90 43 L90 100" stroke="rgba(0,0,0,.10)" stroke-width="1.4"/>
      <path d="${shirt}" fill="url(#kitshade)"/>
      <text x="60" y="78" text-anchor="middle" font-size="27" font-weight="800" fill="${ink}" style="font-family:var(--f-display);letter-spacing:.02em">${esc(initials)}</text>
    </svg>`;
  }

  // ---------------------------------------------------------------- the collectible card
  const tierOf = x => x >= 7 ? "foil" : x >= 5.5 ? "gold" : x >= 4.5 ? "silver" : "bronze";
  function pcard(p, o) {
    o = o || {};
    const x = p.xpts || 0, t = tierOf(x), def = p.pos === "GK" || p.pos === "DEF";
    return `<div class="pcard ${t}${o.cls ? " " + o.cls : ""}" data-pid="${p.id}" tabindex="0" role="button" aria-label="${esc(p.name)}, ${f1(x)} expected points"${o.style ? ` style="${o.style}"` : ""}>
      <div class="face">
        <div class="tl"><span class="xp">${f1(x)}</span><span class="meta"><b>${p.pos}</b><span>${esc(tShort(p.team))}</span><span>€${(+p.price).toFixed(1)}m</span></span></div>
        <div class="art">${kit(p.team, inits(p.name), p.pos === "GK")}</div>
        <span class="cname">${esc(shortName(p.name))}</span>
        <div class="st"><div><span>start</span><b>${pct(p.p_start_sim || 0)}</b></div><div><span>${def ? "clean sheet" : "goal"}</span><b>${pct((def ? p.p_cs : p.p_goal) || 0)}</b></div><div><span>10+</span><b>${pct(p.p_haul || 0)}</b></div></div>
        ${t === "foil" ? '<span class="foil"></span>' : ""}<span class="sheen"></span>${o.cap ? '<span class="capt">Captain</span>' : ""}
      </div></div>`;
  }
  function tilt(el) {                                          // the card leans toward the pointer and the foil follows it
    if (!el) return;
    const host = el.parentElement;
    host.addEventListener("pointermove", e => {
      const r = el.getBoundingClientRect(), mx = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), my = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height));
      el.style.setProperty("--mx", mx.toFixed(3)); el.style.setProperty("--my", my.toFixed(3));
      if (!reduceMotion()) el.style.setProperty("--tilt", `rotateY(${((mx - .5) * 22).toFixed(1)}deg) rotateX(${((.5 - my) * 18).toFixed(1)}deg)`);
    });
    host.addEventListener("pointerleave", () => el.style.setProperty("--tilt", "rotateY(0deg)"));
  }

  // ---------------------------------------------------------------- what every hero needs
  const leftTo = dl => { const ms = dl - Date.now(); if (!(ms > 0)) return "closed"; const d = Math.floor(ms / 864e5), h = Math.floor(ms / 36e5) % 24, m = Math.floor(ms / 6e4) % 60; return `${d}d ${String(h).padStart(2, "0")}h ${String(m).padStart(2, "0")}m`; };
  const formOf = ids => { const c = { DEF: 0, MID: 0, FWD: 0 }; ids.forEach(id => { const p = P[id]; if (p && c[p.pos] != null) c[p.pos]++; }); return `${c.DEF}-${c.MID}-${c.FWD}`; };
  function heroCtx() {
    const b = D.matchdays[MD], { sol, label } = currentSquad(), m = formation ? "formation" : mode();
    const pol = m === "best" ? b.best.policy : null;
    const by = Object.fromEntries(b.players.map(p => [p.id, p]));
    const pl = id => by[id] || Object.assign({ xpts: 0, p_start_sim: 0, p_goal: 0, p_haul: 0, p_cs: 0 }, P[id]);
    const xi = sol.xi.map(pl), bench = (sol.bench || []).map(pl), cap = pl(sol.captain);
    const own = m === "mine" && !!ME && squad.every(id => ME.squad.includes(id));
    const t = own ? (ME.transfers || []).filter(x => x.n > 0 && x.gain > 0.5).sort((a, c) => c.gain - a.gain)[0] : null;
    const held = new Set(xi.concat(bench).map(p => p.id));
    return {
      b, sol, label, m, pol, xi, bench, cap, own, pl, form: sol.formation || formOf(sol.xi),
      ev: pol ? pol.ev_active : sol.ev_xi, dl: new Date(b.deadline_iso),
      tx: t ? { out: t.sell.map(pl), inn: t.buy.map(pl), gain: t.gain, hits: t.hits } : null,
      targets: b.players.filter(p => !held.has(p.id) && p.ps !== "unknown").sort((a, c) => c.xpts - a.xpts).slice(0, 6),
      top: b.players.slice().sort((a, c) => c.xpts - a.xpts),
      name: m === "mine" ? (own ? (ME.team_name || "Your team") : "Your team") : m === "limitless" ? "Limitless" : m === "formation" ? `Best ${formation}` : "Best from scratch",
      ft: own ? ME.free_transfers : null, bank: own ? ME.bank : null
    };
  }
  function heroModes() {
    const b = D.matchdays[MD];
    const ms = (hasMine() ? [["mine", "My team"]] : []).concat([["best", "Best from scratch"]]).concat(b.limitless ? [["limitless", "Limitless"]] : []);
    return `<div class="hmodes" role="group" aria-label="Team shown">${ms.map(([k, l]) => `<button data-hm="${k}" aria-pressed="${mode() === k && !formation}">${l}</button>`).join("")}</div>`;
  }
  let heroKey = "";
  function countUp(el, to) {
    if (!el) return;
    if (reduceMotion()) { el.textContent = f1(to); return; }
    const t0 = performance.now();
    const step = t => { const k = Math.min(1, (t - t0) / 1000), e = 1 - Math.pow(1 - k, 3); el.textContent = f1(to * e); if (k < 1) requestAnimationFrame(step); };
    requestAnimationFrame(step);
  }
  const moveLine = c => c.tx ? `${c.tx.out.map(p => esc(shortName(p.name))).join(" + ")} out, ${c.tx.inn.map(p => esc(shortName(p.name))).join(" + ")} in` : "";

  // ---------------------------------------------------------------- 1. collector cards
  function heroCards(c) {
    const rest = c.xi.filter(p => p.id !== c.cap.id).sort((a, b) => b.xpts - a.xpts);
    const fan = rest.slice(0, 4), ang = [-16, -8, 8, 16], off = [-150, -76, 76, 150];
    let stage = fan.map((p, i) => pcard(p, { cls: "fan", style: `--w:clamp(128px,13.5vw,186px);left:calc(50% - var(--w)/2 + ${off[i] < 0 ? "-" : ""}min(${Math.abs(off[i])}px, ${(Math.abs(off[i]) / 9).toFixed(1)}vw));top:${10 + Math.abs(ang[i]) * 1.5}%;--rot:${ang[i]}deg;z-index:${i < 2 ? i + 1 : 4 - i}` })).join("");
    stage += pcard(c.cap, { cap: true, cls: "lead", style: "--w:clamp(184px,18.5vw,248px);left:calc(50% - var(--w)/2);top:2%;z-index:10" });
    const h = c.m === "mine" ? `Your hand for matchday ${MD}` : c.m === "best" ? `The best hand for matchday ${MD}` : `${c.name}, matchday ${MD}`;
    const trade = c.tx
      ? `<h3 class="ch-h">Cards worth pulling</h3><p class="ch-sub">The best trade for the next three matchdays${c.ft != null ? `, using your ${c.ft} free transfer${c.ft === 1 ? "" : "s"}` : ""}.</p>
         <div class="ch-trade"><div class="pair out">${c.tx.out.map(p => pcard(p)).join("")}</div><span class="arrow" aria-hidden="true">→</span><div class="pair">${c.tx.inn.map(p => pcard(p)).join("")}</div>
         <div class="gain">+${f1(c.tx.gain)}<small>expected points over three matchdays${c.tx.hits ? `, after a −${4 * c.tx.hits} hit` : ", no hit"}</small></div></div>`
      : `<h3 class="ch-h">Best cards outside this squad</h3><p class="ch-sub">The highest expected points this matchday among players you don't hold.</p><div class="ch-row">${c.targets.map(p => pcard(p)).join("")}</div>`;
    return `<div class="ch">
      <div class="ch-hero"><div class="ch-copy">${heroModes()}
        <h1 class="ch-title">${h}</h1>
        <p class="ch-lede">${c.m === "mine" ? `<b>${esc(c.name)}</b> in a ${c.form}. ` : `In a ${c.form}. `}Cards are graded by expected points: foil 7+, gold 5.5+, silver 4.5+, bronze below.</p>
        <div class="ch-tot"><div><span class="n" data-count="${c.ev}">${f1(c.ev)}</span><span class="l">expected points${c.pol ? " with the switch plan" : ", captain doubled"}</span></div>
          <div><span class="n">${pct(c.cap.p_haul || 0)}</span><span class="l">${esc(shortName(c.cap.name))} scores 10+</span></div>
          <div><span class="n hero-clock">${leftTo(c.dl)}</span><span class="l">to the deadline</span></div></div>
      </div><div class="ch-stage">${stage}</div></div>
      <h3 class="ch-h">The rest of the XI</h3><p class="ch-sub">Bench at the end, faded. Tap any card for the full profile.</p>
      <div class="ch-row">${rest.slice(4).map(p => pcard(p)).join("")}${c.bench.map(p => pcard(p, { cls: "benched" })).join("")}</div>
      ${trade}</div>`;
  }

  // ---------------------------------------------------------------- 2. night match
  function heroNight(c) {
    const lines = [["FWD", 22], ["MID", 47], ["DEF", 71], ["GK", 93]];
    let pls = "";
    lines.forEach(([k, y]) => { const row = c.xi.filter(p => p.pos === k), n = row.length; row.forEach((p, i) => {
      const x = n === 1 ? 50 : n === 2 ? 34 + 32 * i : 12 + (76 * i) / (n - 1), f = fixtureOf(p.team, MD);
      pls += `<div class="nm-pl${p.id === c.cap.id ? " cap" : ""}" data-pid="${p.id}" style="left:${x}%;top:${y}%"><div class="disc">${f1(p.xpts)}${p.id === c.cap.id ? '<span class="arm">C</span>' : ""}</div><div class="who">${esc(shortName(p.name))}</div><div class="op">${f ? `${f.home ? "v" : "at"} ${esc(tShort(f.opp))}, ` : ""}${pct(p.p_start_sim)} start</div></div>`;
    }); });
    const chalk = `<svg viewBox="0 0 680 800" preserveAspectRatio="none" aria-hidden="true"><g fill="none" stroke="rgba(238,246,234,.75)" stroke-width="3">
      <rect x="10" y="10" width="660" height="780"/><line x1="10" y1="400" x2="670" y2="400"/><circle cx="340" cy="400" r="92"/>
      <rect x="140" y="10" width="400" height="165"/><rect x="250" y="10" width="180" height="55"/><rect x="140" y="625" width="400" height="165"/><rect x="250" y="735" width="180" height="55"/>
      <path d="M 268 175 A 92 92 0 0 0 412 175"/><path d="M 268 625 A 92 92 0 0 1 412 625"/></g><circle cx="340" cy="400" r="5" fill="rgba(238,246,234,.8)"/></svg>`;
    const tick = c.top.slice(0, 16).map(p => `<span>${esc(shortName(p.name).toUpperCase())}<b>${f1(p.xpts)}</b></span>`).join("");
    const tiles = c.tx
      ? `<div class="nm-tile"><h3>Your move</h3><div class="swap"><span class="o">${c.tx.out.map(p => esc(shortName(p.name))).join(" + ")}</span> out, <span class="i">${c.tx.inn.map(p => esc(shortName(p.name))).join(" + ")}</span> in</div><p>${c.ft != null ? `Uses your <b>${c.ft} free transfer${c.ft === 1 ? "" : "s"}</b>, €${f1(c.bank)}m left in the bank.` : ""}</p></div>
         <div class="nm-tile"><h3>Worth</h3><div class="big">+${f1(c.tx.gain)}</div><p>expected points over the next three matchdays</p></div>`
      : `<div class="nm-tile"><h3>Bench</h3><p>${c.bench.map(p => `<b>${esc(shortName(p.name))}</b> ${f1(p.xpts)}`).join(", ")}</p></div>
         <div class="nm-tile"><h3>Best outside</h3><p>${c.targets.slice(0, 3).map(p => `<b>${esc(shortName(p.name))}</b> ${f1(p.xpts)}`).join(", ")}</p></div>`;
    return `<div class="nm">
      <div class="nm-head"><div>${heroModes()}<h1 class="nm-title">Matchday ${MD}</h1><div class="nm-who"><b>${esc(c.name)}</b>, ${c.form}, <span class="hero-clock nm-clock">${leftTo(c.dl)}</span> to the deadline</div></div>
        <div class="nm-score"><div class="n" data-count="${c.ev}">${f1(c.ev)}</div><div class="l">expected points, <b>${esc(shortName(c.cap.name))}</b> captain</div></div></div>
      <div class="nm-stadium"><div class="nm-turf">${chalk}${pls}</div><div class="nm-glow"></div></div>
      <div class="nm-led" aria-label="Top projections this matchday"><div class="run">${tick}${tick}</div></div>
      <div class="nm-tiles">${tiles}<div class="nm-tile"><h3>Captain</h3><div class="big light">${esc(shortName(c.cap.name))}</div><p>${f1(c.cap.xpts)} expected, ${pct(c.cap.p_haul || 0)} chance of 10 or more</p></div></div>
    </div>`;
  }

  // ---------------------------------------------------------------- 3. matchday poster
  function heroPoster(c) {
    const mine = new Set(c.xi.map(p => p.id)), top = c.top.slice(0, 12), mx = Math.max(...top.map(p => p.xpts), 1);
    const line = k => `<div class="po-line">${c.xi.filter(p => p.pos === k).sort((a, b) => b.xpts - a.xpts).map(p => `<span data-pid="${p.id}" class="${p.id === c.cap.id ? "c" : ""}" style="font-size:${(14 + p.xpts * 6).toFixed(0)}px">${esc(shortName(p.name))}<sup>${f1(p.xpts)}</sup></span>`).join("")}</div>`;
    return `<div class="po">
      <div class="po-top">${heroModes()}<span>Matchday ${MD}, ${esc(c.name)}, <span class="hero-clock">${leftTo(c.dl)}</span> to the deadline</span></div>
      <div class="po-big" aria-label="${f1(c.ev)} expected points">${f1(c.ev)}</div>
      <div class="po-cap"><span>expected points from the XI</span><span><b>${esc(shortName(c.cap.name))}</b> captain</span><span class="hot">${pct(c.cap.p_haul || 0)} chance he scores 10+</span></div>
      <div class="po-grid">
        <div><h2>The XI, sized by expected points</h2><div class="po-sheet">${line("FWD")}${line("MID")}${line("DEF")}${line("GK")}</div></div>
        <div><h2>The top twelve this matchday</h2><div class="po-race">${top.map(p => `<div class="${mine.has(p.id) ? "mine" : ""}" data-pid="${p.id}"><span>${esc(shortName(p.name))}</span><i style="transform:scaleX(${(p.xpts / mx).toFixed(3)})"></i><b>${f1(p.xpts)}</b></div>`).join("")}</div>
          <p class="po-note"><i></i>in this XI</p></div>
      </div>
      ${c.tx ? `<div class="po-deal"><div class="t"><s>${c.tx.out.map(p => esc(shortName(p.name))).join(" and ")}</s> out, ${c.tx.inn.map(p => esc(shortName(p.name))).join(" and ")} in</div><div class="g">+${f1(c.tx.gain)}<small>over three matchdays${c.ft != null ? `, ${c.ft} free transfers` : ""}</small></div></div>` : ""}
    </div>`;
  }

  // ---------------------------------------------------------------- 4. market terminal
  function heroMarket(c) {
    const delta = p => { const s = SEASON[p.id]; const x = s && s.x ? s.x.filter(v => v != null) : []; return x.length ? p.xpts - x[x.length - 1] : null; };
    const tick = c.top.slice(0, 18).map(p => { const d = delta(p); return `<span>${esc(shortName(p.name).toUpperCase())} <b>${f1(p.xpts)}</b>${d == null ? "" : ` <i class="${d >= 0 ? "up" : "dn"}">${d >= 0 ? "▲" : "▼"}${f1(Math.abs(d))}</i>`}</span>`; }).join("");
    const outIds = new Set(c.tx ? c.tx.out.map(p => p.id) : []);
    const spark = id => {
      const s = SEASON[id]; if (!s || !s.p || !s.p.length) return "";
      const v = s.p.map(x => x == null ? 0 : x), mx = Math.max(10, ...v), w = 70, h = 22;
      const pts = v.map((y, i) => `${(i * w / Math.max(1, v.length - 1)).toFixed(1)},${(h - 2 - Math.max(y, 0) / mx * (h - 4)).toFixed(1)}`).join(" ");
      return `<svg class="spark" viewBox="0 0 ${w} ${h}" aria-hidden="true"><polyline points="${pts}" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>`;
    };
    const last = id => { const s = SEASON[id]; const v = s && s.p ? s.p.filter(x => x != null) : []; return v.length ? v[v.length - 1] : "–"; };
    const tot = id => { const s = SEASON[id]; return s ? s.s[9] : "–"; };
    const row = (p, sig) => `<tr data-pid="${p.id}"><td class="tk"><b>${esc(shortName(p.name))}</b><small>${p.pos} ${esc(tShort(p.team))}</small></td><td>€${(+p.price).toFixed(1)}m</td><td class="x">${f1(p.xpts)}</td><td>${pct(p.p_start_sim || 0)}</td><td>${last(p.id)}</td><td>${tot(p.id)}</td><td class="sp">${spark(p.id)}</td><td><span class="sig ${sig.toLowerCase()}">${sig}</span></td></tr>`;
    const rows = c.xi.slice().sort((a, b) => b.xpts - a.xpts).map(p => row(p, p.id === c.cap.id ? "CAPTAIN" : outIds.has(p.id) ? "SELL" : "HOLD")).join("")
      + c.bench.map(p => row(p, outIds.has(p.id) ? "SELL" : "BENCH")).join("") + (c.tx ? c.tx.inn.map(p => row(p, "BUY")).join("") : "");
    const movers = key => Object.entries(SEASON).filter(([id, s]) => s[key] != null && P[id]).sort((a, b) => b[1][key] - a[1][key]).slice(0, 6)
      .map(([id, s]) => `<li data-pid="${id}"><span>${esc(shortName(P[id].name))} <small>${esc(tShort(P[id].team))}</small></span><b>${Number(s[key]).toLocaleString("en-GB")}</b></li>`).join("");
    const hasMov = Object.values(SEASON).some(s => s.tin != null);
    return `<div class="mk">
      <div class="mk-tape" aria-label="Top projections, change against the last forecast"><div class="run">${tick}${tick}</div></div>
      <div class="mk-head"><div>${heroModes()}<h1 class="mk-title">Matchday ${MD} positions</h1><p class="mk-sub">${esc(c.name)}, ${c.form}. Deadline in <span class="hero-clock">${leftTo(c.dl)}</span>.</p></div>
        <div class="mk-quote"><span class="l">expected return</span><span class="n" data-count="${c.ev}">${f1(c.ev)}</span><span class="l">captain ${esc(shortName(c.cap.name))}, ${pct(c.cap.p_haul || 0)} chance of 10+</span></div></div>
      <div class="mk-grid">
        <div class="mk-panel"><h3>Positions</h3><div class="scroll-x"><table class="mk-table"><thead><tr><th>Player</th><th>Price</th><th>xPts</th><th>Start</th><th>Last MD</th><th>Season</th><th>Form</th><th>Signal</th></tr></thead><tbody>${rows}</tbody></table></div></div>
        <div class="mk-side">
          ${c.tx ? `<div class="mk-panel mk-order"><h3>Order ticket</h3><p><span class="sig sell">SELL</span> ${c.tx.out.map(p => esc(shortName(p.name))).join(", ")}</p><p><span class="sig buy">BUY</span> ${c.tx.inn.map(p => esc(shortName(p.name))).join(", ")}</p><p class="gain">+${f1(c.tx.gain)}</p><p class="fine">expected points over three matchdays${c.ft != null ? `, ${c.ft} free transfers, €${f1(c.bank)}m in the bank` : ""}</p></div>` : ""}
          ${hasMov ? `<div class="mk-panel"><h3>Most bought</h3><ol class="mk-mov">${movers("tin")}</ol></div><div class="mk-panel"><h3>Most sold</h3><ol class="mk-mov dn">${movers("tout")}</ol></div>` : ""}
        </div>
      </div></div>`;
  }

  // ---------------------------------------------------------------- the hero, called by renderBoard
  function renderTplHero() {
    const box = $("#tpl-hero"), t = TPL();
    if (!box) return;
    if (t === "board") { box.innerHTML = ""; return; }
    const c = heroCtx();
    box.innerHTML = t === "night" ? heroNight(c) : t === "poster" ? heroPoster(c) : t === "market" ? heroMarket(c) : heroCards(c);
    $$("[data-hm]", box).forEach(b => b.addEventListener("click", () => { boardMode = b.dataset.hm; formation = null; renderBoard(); renderReadout(); renderChecks(); }));
    if (t === "cards") tilt($(".ch-stage .lead", box));
    const key = t + MD;
    if (key !== heroKey) {                                     // the one moment of motion per template and matchday
      heroKey = key;
      const n = $("[data-count]", box);
      if (t === "night" && !reduceMotion()) { box.classList.remove("boot"); void box.offsetWidth; box.classList.add("boot"); setTimeout(() => countUp(n, +n.dataset.count), 1200); }
      else if (t === "poster" && !reduceMotion()) { box.classList.remove("boot"); void box.offsetWidth; box.classList.add("boot"); }
      else countUp(n, +n.dataset.count);
    }
  }
  setInterval(() => { const dl = new Date(D.matchdays[MD].deadline_iso); $$(".hero-clock").forEach(e => { e.textContent = leftTo(dl); }); }, 30000);

  // ---------------------------------------------------------------- the Players tab as cards
  let plView = store.get("plview", "cards") === "table" ? "table" : "cards";
  function renderPlayerCards(rows) {
    const seg = $("#pl-viewseg"), grid = $("#pl-cards"), tbl = $("#pl-table").closest(".scroll-x");
    const cardsOn = TPL() === "cards";
    seg.hidden = !cardsOn;
    const on = cardsOn && plView === "cards";
    grid.hidden = !on; tbl.hidden = on;
    seg.innerHTML = [["cards", "Cards"], ["table", "Table"]].map(([k, l]) => `<button data-v="${k}" aria-pressed="${plView === k}">${l}</button>`).join("");
    $$("button", seg).forEach(b => b.addEventListener("click", () => { plView = b.dataset.v; store.set("plview", plView); renderPlayers(); }));
    if (on) grid.innerHTML = rows.slice(0, plLimit).map(p => pcard(p)).join("") || `<p class="fine">No players match these filters. Clear the search or raise the price limit.</p>`;
  }

  // ---------------------------------------------------------------- the template menu
  const PREV = {
    cards: `<span class="pv pv-cards"><i style="--r:-9deg"></i><i style="--r:8deg"></i><i class="f"><b>7.3</b></i></span>`,
    night: `<span class="pv pv-night"><i></i><b>65.5</b><u>RAPHINHA 8.2&nbsp;&nbsp;MBAPPÉ 7.3</u></span>`,
    poster: `<span class="pv pv-poster"><b>65.5</b><small>expected points</small></span>`,
    market: `<span class="pv pv-market"><u>RAPH 8.2 ▲0.4&nbsp;&nbsp;MBAP 7.3 ▼0.1</u><b>65.5</b><small>SELL YAMAL&nbsp;&nbsp;BUY PAZ</small></span>`,
    board: `<span class="pv pv-board"><i></i><i></i><i></i><b>Matchday 5</b></span>`
  };
  const tplDlg = document.createElement("dialog");
  tplDlg.className = "skins"; tplDlg.setAttribute("aria-labelledby", "skins-h");
  document.body.appendChild(tplDlg);
  function renderTplMenu() {
    const cur = TPL();
    tplDlg.innerHTML = `<div class="skins-head"><h2 id="skins-h">Choose a template</h2><button class="sheet-x" aria-label="Close">×</button></div>
      <div class="skins-grid">${TEMPLATES.map(t => `<button class="skin-opt" data-skin="${t.k}" aria-pressed="${t.k === cur}">${PREV[t.k]}<strong>${esc(t.n)}</strong><span class="d">${esc(t.d)}</span></button>`).join("")}</div>
      <p class="skin-foot">Only the look changes; every number stays the same. Your choice is kept in this browser.</p>`;
    $(".sheet-x", tplDlg).addEventListener("click", () => tplDlg.close());
    $$(".skin-opt", tplDlg).forEach(b => b.addEventListener("click", () => {
      window.ufeSkin(b.dataset.skin); store.set("skin", b.dataset.skin);
      $$(".skin-opt", tplDlg).forEach(x => x.setAttribute("aria-pressed", x === b));
      renderBoard(); renderReadout(); renderPlayers();
    }));
  }
  $("#skin-btn").addEventListener("click", () => { TEMPLATES.forEach(x => window.ufeSkin(x.k, true)); renderTplMenu(); tplDlg.showModal ? tplDlg.showModal() : tplDlg.setAttribute("open", ""); });
  tplDlg.addEventListener("click", e => { if (e.target === tplDlg) tplDlg.close(); });
