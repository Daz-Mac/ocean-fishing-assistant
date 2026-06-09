// Ocean Fishing Assistant — Lovelace Custom Card
// Only enable verbose console logging when raw_output_enabled is set

/* ---- Styles ---- */
const styles = `
  :host { display: block; }
  .card { padding: 16px; font-family: var(--primary-font-family, Roboto); font-size: 14px; line-height: 1.4; }
  .loading, .unavailable, .config-msg { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 32px 16px; color: var(--secondary-text-color); text-align: center; gap: 8px; }
  .loading { min-height: 60px; }

  .header { display: flex; justify-content: space-between; padding: 8px 0; margin-bottom: 8px; border-bottom: 1px solid var(--divider-color, rgba(0,0,0,0.08)); }
  .header-title { font-weight: 600; font-size: 15px; color: var(--primary-text-color); }
  .header-moon { font-size: 13px; color: var(--secondary-text-color); }

  .score-bar { position: relative; height: 48px; border-radius: 24px; overflow: hidden; background: linear-gradient(to right, #e53935 0%, #e53935 33%, #fdd835 33%, #fdd835 66%, #43a047 66%, #43a047 100%); display: flex; align-items: center; justify-content: center; margin-bottom: 12px; }
  .score-fill { position: absolute; right: 0; top: 0; bottom: 0; background: rgba(255,255,255,0.4); border-radius: 0 24px 24px 0; }
  .score-num { position: relative; font-size: 22px; font-weight: 700; color: #fff; text-shadow: 0 1px 3px rgba(0,0,0,0.3); z-index: 1; }
  .score-label { text-align: center; margin: -8px 0 12px; font-size: 13px; font-weight: 600; }

  .conditions { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 8px; margin-bottom: 12px; padding: 10px; background: var(--secondary-background-color, rgba(0,0,0,0.02)); border-radius: 10px; }
  .c-item { display: flex; align-items: center; gap: 6px; font-size: 12px; }
  .c-label { font-size: 10px; color: var(--secondary-text-color); }
  .c-value { font-weight: 500; color: var(--primary-text-color); }

  .section-title { font-size: 12px; font-weight: 600; color: var(--secondary-text-color); margin: 8px 0 4px; }
  .row { display: flex; align-items: center; gap: 6px; padding: 3px 0; font-size: 12px; border-bottom: 1px solid var(--divider-color, rgba(0,0,0,0.05)); }
  .row:last-child { border-bottom: none; }
  .rdate { color: var(--secondary-text-color); width: 48px; flex-shrink: 0; }
  .rper { color: var(--secondary-text-color); width: 48px; flex-shrink: 0; font-size: 11px; }
  .rbar-wrap { flex: 1; height: 10px; background: var(--secondary-background-color, #f0f0f0); border-radius: 5px; overflow: hidden; }
  .rbar { height: 100%; border-radius: 5px; }
  .rscore { font-weight: 600; width: 24px; text-align: right; }
  .empty { font-size: 12px; color: var(--secondary-text-color); padding: 8px; text-align: center; }

  .footer { border-top: 1px solid var(--divider-color, rgba(0,0,0,0.08)); padding-top: 6px; margin-top: 4px; font-size: 11px; color: var(--secondary-text-color); text-align: center; opacity: 0.7; }
`;

function barColor(s) {
  if (s == null || s < 50) return '#e53935';
  if (s < 70) return '#fdd835';
  return '#43a047';
}

function formatTime(ts) {
  if (!ts) return null;
  const d = new Date(ts);
  if (isNaN(d.getTime())) return null;
  return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
}
const factorLabels = { tide:'Tide', wind:'Wind', wind_direction:'Wind Dir', waves:'Waves', time:'Time', pressure:'Pressure', season:'Season', moon:'Moon', temperature:'Temp' };

function buildCard(a, config) {
  const title = config.title || 'Ocean Fishing';
  const showForecast = config.show_forecast !== false;
  const forecastDays = Math.min(Math.max(parseInt(config.forecast_days, 10) || 2, 1), 5);
  const scoreRaw = parseInt(a.state, 10);
  const isError = a.state === 'unavailable' || isNaN(scoreRaw);
  const attrs = a.attributes || {};
  const hasCachedData = Object.keys(attrs.remainder_of_today_periods || {}).length > 0 || Object.keys(attrs.next_5_day_periods || {}).length > 0;

  // Unavailable with no cached data — render minimal error card
  if (isError && !hasCachedData) {
    return `
      <ha-card>
        <div class="card">
          <div class="header">
            <span class="header-title">🎣 ${title || 'Ocean Fishing'}</span>
          </div>
          <div class="unavailable">
            <ha-icon icon="hass:alert" style="--mdc-icon-size:32px;color:var(--error-color,#e53935)"></ha-icon>
            <div style="font-size:13px;font-weight:500;margin-top:4px">Sensor data unavailable</div>
            <div style="font-size:11px;color:var(--secondary-text-color);margin-top:2px">Open-Meteo or World Tides API may be unreachable</div>
            <div style="font-size:10px;color:var(--secondary-text-color);margin-top:8px">Check your internet connection and API key configuration</div>
          </div>
        </div>
      </ha-card>
    `;
  }

  const score = isError ? (attrs.current_forecast?.score_100 || 0) : scoreRaw;
  const cf = attrs.current_forecast || {};
  const comps = cf.components || {};
  const moon = attrs.moon_phase_name || (comps.moon ? comps.moon.moon_phase_name : null);
  const profile = attrs.profile_used || {};
  const nextHigh = attrs.next_high_tide || null;
  const label = score >= 70 ? 'Excellent' : score >= 50 ? 'Fair' : 'Poor';
  const pct = 100 - Math.min(Math.max(score || 0, 0), 100);
  // All 9 factor scores from current_forecast components
  const compsList = [
    { k:'tide', i:'hass:waves', l:'Tide', v: comps.tide?.tide_phase },
    { k:'wind', i:'hass:weather-windy', l:'Wind', v: comps.wind?.wind_speed },
    { k:'wind_direction', i:'hass:compass', l:'Wind Dir', v: comps.wind_direction?.wind_direction_deg != null ? comps.wind_direction.wind_direction_deg + '°' : null },
    { k:'waves', i:'hass:wave', l:'Waves', v: comps.waves?.wave_height },
    { k:'pressure', i:'hass:gauge', l:'Pressure', v: comps.pressure?.pressure_delta },
    { k:'temperature', i:'hass:thermometer', l:'Temp', v: comps.temperature?.temperature },
    { k:'season', i:'hass:calendar', l:'Season', v: null },
    { k:'moon', i:'hass:brightness-7', l:'Moon', v: comps.moon?.moon_phase_name },
    { k:'time', i:'hass:clock', l:'Time', v: null },
  ];
  const safetyVals = cf.safety_values || {};
  const safetyItems = [];
  if (safetyVals.wind_gust) safetyItems.push({ l:'Gust', v: safetyVals.wind_gust });
  if (safetyVals.swell_period_s != null) safetyItems.push({ l:'Swell', v: safetyVals.swell_period_s + 's' });
  if (safetyVals.precipitation_probability != null) safetyItems.push({ l:'Precip', v: safetyVals.precipitation_probability + '%' });

  // Forecast periods
  const today = attrs.remainder_of_today_periods || {};
  const next5 = attrs.next_5_day_periods || {};
  const pMap = { period_00_06:'00-06', period_06_12:'06-12', period_12_18:'12-18', period_18_24:'18-24' };
  const rows = [];
  for (const [periodName, periodData] of Object.entries(today))
    rows.push({ d:'Today', p: pMap[periodName]||periodName, s: periodData.score_100, t: periodData.tide_phase, c: periodData.components, b: periodData.spring_tide_bonus || 0 });
  for (const d of Object.keys(next5).sort().slice(0, forecastDays)) {
    const label = new Date(d).toLocaleDateString(undefined, {month:'short', day:'numeric'});
    for (const [pn, pd] of Object.entries(next5[d]))
      rows.push({ d: label, p: pMap[pn]||pn, s: pd.score_100, t: pd.tide_phase, c: pd.components, b: pd.spring_tide_bonus || 0 });
  }
  const todayCount = Object.entries(today).length;
  const display = rows.slice(0, todayCount + forecastDays * 4);
  // Store data for click handler (as JSON on container)
  const rowsData = JSON.stringify(display.map(r => ({ d: r.d, p: r.p, s: r.s, t: r.t, c: r.c, b: r.b })));

  return `
    <ha-card>
      <div class="card">
        ${isError ? `<div style="background:rgba(229,57,53,0.1);border:1px solid rgba(229,57,53,0.3);border-radius:6px;padding:8px;margin-bottom:8px;font-size:11px;display:flex;align-items:center;gap:6px">
          <ha-icon icon="hass:alert" style="color:#e53935;--mdc-icon-size:16px"></ha-icon>
          <span>Weather/tide data temporarily unavailable — showing cached data</span>
        </div>` : ''}
        <div class="header">
          <span class="header-title">🎣 ${title || 'Ocean Fishing'}</span>
          ${moon ? `<span class="header-moon">🌙 ${moon}</span>` : ''}
        </div>

        <div class="score-bar">
          <div class="score-fill" style="width:${pct}%"></div>
          <span class="score-num">${score}</span>
        </div>
        <div class="score-label" style="color:${barColor(score)}">Now: ${label} &middot; ${score}/100</div>

        <div class="section-title" style="margin-bottom:4px">Now &mdash; Score Breakdown</div>
        <div class="conditions" style="margin-bottom:4px">
          ${compsList.map(c => {
            const sc = comps[c.k]?.score_100;
            const scColor = barColor(sc);
            const scDisplay = sc != null ? `${sc}` : '--';
            return `<div class="c-item">
              <ha-icon icon="${c.i}"></ha-icon>
              <div style="flex:1;min-width:0">
                <div class="c-label">${c.l}</div>
                <div class="c-value" style="display:flex;justify-content:space-between;align-items:center">
                  <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:70px">${c.v || ''}</span>
                  <span style="font-weight:700;margin-left:4px;flex-shrink:0;color:${scColor}">${scDisplay}</span>
                </div>
              </div>
            </div>`;
          }).join('')}
        </div>
        ${safetyItems.length ? `<div class="conditions" style="grid-template-columns:repeat(${Math.min(safetyItems.length,3)},1fr);margin-bottom:12px;padding:6px 10px">
          ${safetyItems.map(s => `<div style="font-size:11px;text-align:center"><span style="color:var(--secondary-text-color)">${s.l}</span><br><span style="font-weight:500">${s.v}</span></div>`).join('')}
        </div>` : ''}

        ${showForecast ? (display.length ? `
        <div class="section-title">Forecast <span style="font-weight:400;font-size:11px;color:var(--secondary-text-color)">(tap row for details)</span></div>
        <div id="forecast-container" data-rows='${rowsData}'>
        ${display.map((r, i) => {
          return `<div class="row" data-idx="${i}">
            <span class="rdate">${r.d}</span>
            <span class="rper">${r.p}</span>
            <div class="rbar-wrap"><div class="rbar" style="width:${Math.min(r.s||0,100)}%;background:${barColor(r.s)}"></div></div>
            <span class="rscore" style="color:${barColor(r.s)}">${r.s != null ? r.s : '--'}</span>
          </div>
          <div class="row-detail" id="detail-${i}" style="display:none;font-size:11px;padding:6px 8px;background:var(--secondary-background-color,rgba(0,0,0,0.03));border-radius:6px;margin-bottom:4px"></div>`;
        }).join('')}
        </div>
        ` : `<div class="empty">No forecast data</div>`) : ''}

        <div class="footer">${formatTime(nextHigh?.timestamp) ? `🌊 Next high tide: ${formatTime(nextHigh.timestamp)} &middot; ` : ''}${profile.common_name ? `${profile.common_name}${profile.scientific_name ? ` (${profile.scientific_name})` : ''} &middot; ` : ''}Data: Open-Meteo, World Tides</div>
      </div>
    </ha-card>
  `;
}

/* ---- Card element ---- */
class OceanFishingCard extends HTMLElement {
  set hass(hass) { this._hass = hass; this._render(); }

  setConfig(config) {
    this._config = config || {};
    if (!config || !config.entity) {
      // Config will be set later by the editor
    }
  }

  constructor() {
    super();
    this._hass = null;
    this._config = {};
    this._shadow = this.attachShadow({ mode: 'open' });
    this._shadow.innerHTML = `<style>${styles}</style>`;
    this._expandedRow = null;
  }

  _render() {
    if (!this._config || !this._config.entity) {
      this._shadow.innerHTML = `<style>${styles}</style><ha-card><div class="config-msg">Configure entity in card settings</div></ha-card>`;
      return;
    }
    if (!this._hass) {
      this._shadow.innerHTML = `<style>${styles}</style><ha-card><div class="loading">Loading...</div></ha-card>`;
      return;
    }
    const stateObj = this._hass.states[this._config.entity];
    if (!stateObj) {
      this._shadow.innerHTML = `<style>${styles}</style><ha-card><div class="config-msg">Sensor not found: ${this._config.entity}</div></ha-card>`;
      return;
    }

    if (stateObj.attributes.raw_output_enabled) {
      console.log('[OceanFishingCard] Rendering with state:', stateObj.state, 'attrs:', Object.keys(stateObj.attributes));
      console.log('[OceanFishingCard] today periods:', Object.keys(stateObj.attributes.remainder_of_today_periods || {}).length);
      console.log('[OceanFishingCard] next5 days:', Object.keys(stateObj.attributes.next_5_day_periods || {}).length);
    }

    const html = `<style>${styles}</style>` + buildCard(stateObj, this._config);
    this._shadow.innerHTML = html;

    // Set up click handlers for forecast rows
    const container = this._shadow.getElementById('forecast-container');
    if (container) {
      let rowsData = [];
      try { rowsData = JSON.parse(container.dataset.rows || '[]'); } catch (_) {}
      container.addEventListener('click', (e) => {
        const row = e.target.closest('.row');
        if (!row) return;
        const idx = parseInt(row.dataset.idx, 10);
        if (isNaN(idx) || !rowsData[idx]) return;
        const detail = this._shadow.getElementById(`detail-${idx}`);
        if (!detail) return;
        if (detail.style.display !== 'none') {
          detail.style.display = 'none';
          this._expandedRow = null;
          return;
        }
        // Build detail content from components
        const r = rowsData[idx];
        let content = `<div style="display:grid;grid-template-columns:1fr 1fr;gap:2px 8px">`;
        if (r.c) {
          for (const [fk, fv] of Object.entries(r.c)) {
            const name = factorLabels[fk] || fk;
            const sc = fv.score_100 != null ? fv.score_100 : '--';
            const color = barColor(sc);
            content += `<div style="display:flex;justify-content:space-between">
              <span>${name}</span>
              <span style="color:${color};font-weight:500">${sc}</span>
            </div>`;
          }
        } else {
          content += `<div style="grid-column:1/-1;color:var(--secondary-text-color)">No detail data</div>`;
        }
        content += `</div>`;
        if (r.t) {
          const ts = r.c && r.c.tide && r.c.tide.score_100;
          const tc = ts != null ? barColor(ts) : '';
          content += `<div style="margin-top:4px;font-size:10px;display:flex;align-items:center;gap:4px">
            <span style="color:var(--secondary-text-color)">Tide: ${r.t} ${r.t === 'rising' ? '↑' : r.t === 'falling' ? '↓' : ''}</span>
            ${ts != null ? `<div style="flex:1;height:6px;background:var(--secondary-background-color,#f0f0f0);border-radius:3px;overflow:hidden;max-width:50px"><div style="height:100%;width:${Math.min(ts,100)}%;background:${tc};border-radius:3px"></div></div><span style="color:${tc};font-weight:500">${ts}</span>` : ''}
          </div>`;
        }
        if (r.b) {
          content += `<div style="margin-top:2px;font-size:10px;color:#FFD700">🌊 Spring Tide +${r.b}</div>`;
        }
        detail.style.display = 'block';
        this._expandedRow = idx;
      });

      // Restore expanded row after re-render
      if (this._expandedRow !== null) {
        const detail = this._shadow.getElementById(`detail-${this._expandedRow}`);
        if (detail && rowsData[this._expandedRow]) {
          const r = rowsData[this._expandedRow];
          let content = `<div style="display:grid;grid-template-columns:1fr 1fr;gap:2px 8px">`;
          if (r.c) {
            for (const [fk, fv] of Object.entries(r.c)) {
              const name = factorLabels[fk] || fk;
              const sc = fv.score_100 != null ? fv.score_100 : '--';
              const color = barColor(sc);
              content += `<div style="display:flex;justify-content:space-between">
                <span>${name}</span>
                <span style="color:${color};font-weight:500">${sc}</span>
              </div>`;
            }
          }
          content += `</div>`;
          if (r.t) {
            const ts = r.c && r.c.tide && r.c.tide.score_100;
            const tc = ts != null ? barColor(ts) : '';
            content += `<div style="margin-top:4px;font-size:10px;display:flex;align-items:center;gap:4px">
              <span style="color:var(--secondary-text-color)">Tide: ${r.t} ${r.t === 'rising' ? '↑' : r.t === 'falling' ? '↓' : ''}</span>
              ${ts != null ? `<div style="flex:1;height:6px;background:var(--secondary-background-color,#f0f0f0);border-radius:3px;overflow:hidden;max-width:50px"><div style="height:100%;width:${Math.min(ts,100)}%;background:${tc};border-radius:3px"></div></div><span style="color:${tc};font-weight:500">${ts}</span>` : ''}
            </div>`;
          }
          if (r.b) {
            content += `<div style="margin-top:2px;font-size:10px;color:#FFD700">🌊 Spring Tide +${r.b}</div>`;
          }
          detail.style.display = 'block';
        } else {
          this._expandedRow = null;
        }
      }
    }
  }

  getCardSize() { return 3; }

  static async getConfigForm() {
    return {
      schema: [
        { name: 'entity', selector: { entity: { domain: 'sensor' } } },
        { name: 'title', selector: { text: {} } },
        { name: 'show_forecast', selector: { boolean: {} } },
        { name: 'forecast_days', selector: { number: { min: 1, max: 5, mode: 'slider' } } },
      ],
    };
  }
  static getStubConfig() { return { show_forecast: true, forecast_days: 2 }; }
}

customElements.define('ocean-fishing-card', OceanFishingCard);
window.customCards = window.customCards || [];
window.customCards.push({ type: 'ocean-fishing-card', name: 'Ocean Fishing Assistant', description: 'Fishing conditions', preview: false });

// Registered — no startup log needed unless raw_output_enabled is active
