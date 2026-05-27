// Ocean Fishing Assistant — Lovelace Custom Card
// Uses native Custom Elements v1 API — no LitElement dependency.
// Compatible with HA 2024.x through 2026.x.
// Register as custom:ocean-fishing-card in Lovelace dashboard.

const CARD_VERSION = '1.0.0';

/* ---- Styles ---- */
const styles = `
  :host { display: block; }
  .card { padding: 16px; font-family: var(--primary-font-family, Roboto); font-size: 14px; line-height: 1.4; }

  /* States */
  .loading, .unavailable, .config-msg {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    padding: 32px 16px; color: var(--secondary-text-color); text-align: center; gap: 8px;
  }
  .loading { min-height: 60px; }
  .unavailable ha-icon { --mdc-icon-size: 32px; color: var(--error-color, #e53935); }
  .hint { font-size: 12px; color: var(--secondary-text-color); opacity: 0.7; }

  /* Header */
  .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid var(--divider-color, rgba(0,0,0,0.08)); }
  .header-left { display: flex; align-items: center; gap: 8px; }
  .header-icon { font-size: 20px; }
  .header-title { font-weight: 600; font-size: 15px; color: var(--primary-text-color); }
  .header-moon { font-size: 13px; color: var(--secondary-text-color); }

  /* Score bar */
  .score-section { margin-bottom: 12px; }
  .score-bar { position: relative; height: 48px; border-radius: 24px; overflow: hidden; display: flex; align-items: center; justify-content: center; }
  .score-fill { position: absolute; right: 0; top: 0; bottom: 0; background: rgba(255,255,255,0.4); border-radius: 0 24px 24px 0; transition: width 0.3s ease; }
  .score-num { position: relative; font-size: 22px; font-weight: 700; color: #fff; text-shadow: 0 1px 3px rgba(0,0,0,0.3); z-index: 1; }
  .score-label { text-align: center; margin-top: 4px; font-size: 13px; }
  .dominant-factor { text-align: center; font-size: 12px; color: var(--warning-color, #e53935); margin-top: 2px; }

  /* Conditions grid */
  .conditions { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 12px; margin-bottom: 12px; padding: 10px; background: var(--secondary-background-color, rgba(0,0,0,0.02)); border-radius: 10px; }
  .c-item { display: flex; align-items: center; gap: 8px; font-size: 13px; }
  .c-item ha-icon { --mdc-icon-size: 18px; color: var(--state-icon-color, #666); flex-shrink: 0; }
  .c-label { font-size: 11px; color: var(--secondary-text-color); line-height: 1.2; }
  .c-value { font-weight: 500; color: var(--primary-text-color); font-size: 13px; }

  /* Chart */
  .chart-section { margin-bottom: 12px; }
  .section-title { font-size: 12px; font-weight: 600; color: var(--secondary-text-color); margin-bottom: 6px; }
  .chart-fallback { font-size: 12px; border: 1px dashed var(--divider-color, #ccc); border-radius: 8px; padding: 8px; }
  .chart-row { display: flex; justify-content: space-between; padding: 3px 0; border-bottom: 1px solid var(--divider-color, rgba(0,0,0,0.05)); }
  .chart-row:last-child { border-bottom: none; }
  .chart-row-time { color: var(--secondary-text-color); font-size: 11px; }
  .chart-row-score { font-weight: 500; }
  .chart-more { text-align: center; font-size: 11px; color: var(--secondary-text-color); padding: 4px; }

  /* Forecast */
  .forecast-section { margin-bottom: 8px; }
  .f-row { display: flex; align-items: center; gap: 6px; padding: 4px 0; font-size: 12px; border-bottom: 1px solid var(--divider-color, rgba(0,0,0,0.05)); }
  .f-row:last-child { border-bottom: none; }
  .f-date { color: var(--secondary-text-color); width: 52px; flex-shrink: 0; }
  .f-period { color: var(--secondary-text-color); width: 52px; flex-shrink: 0; font-size: 11px; }
  .f-bar-wrap { flex: 1; height: 10px; background: var(--secondary-background-color, #f0f0f0); border-radius: 5px; overflow: hidden; }
  .f-bar { height: 100%; border-radius: 5px; transition: width 0.3s ease; }
  .f-score { font-weight: 600; width: 28px; text-align: right; }
  .empty-data { font-size: 12px; color: var(--secondary-text-color); padding: 8px; text-align: center; }

  /* Footer */
  .footer { border-top: 1px solid var(--divider-color, rgba(0,0,0,0.08)); padding-top: 8px; font-size: 11px; color: var(--secondary-text-color); text-align: center; opacity: 0.7; }
`;

/* ---- Card HTML template (called each render) ---- */
function buildCard(stateObj, attrs) {
  const score = stateObj ? parseInt(stateObj.state, 10) : null;
  const cf = attrs.current_forecast || {};
  const comps = cf.components || {};
  const a = attrs;
  const profile = a.profile_used || {};

  return `
    <ha-card>
      <div class="card">
        ${_header(a, comps, cf)}
        ${_scoreBar(score, cf)}
        ${_conditions(a, comps, cf)}
        ${_chart(comps)}
        ${_forecast(a)}
        ${_footer(profile)}
      </div>
    </ha-card>
  `;
}

function _header(a, comps, cf) {
  const moon = a.moon_phase_name || (comps.moon ? comps.moon.moon_phase_name : null);
  return `<div class="header">
    <div class="header-left"><span class="header-icon">🎣</span><span class="header-title">Ocean Fishing</span></div>
    ${moon ? `<span class="header-moon">🌙 ${moon}</span>` : ''}
  </div>`;
}

function _scoreBar(score, cf) {
  const label = score == null ? '--' : score >= 70 ? 'Excellent' : score >= 50 ? 'Fair' : 'Poor';
  const color = score == null ? '#888' : score >= 70 ? '#43a047' : score >= 50 ? '#fdd835' : '#e53935';
  const pct = 100 - (score || 0);
  const dominant = _dominantFactor(cf);
  return `<div class="score-section">
    <div class="score-bar" style="background:linear-gradient(to right,#e53935 0%,#e53935 33%,#fdd835 33%,#fdd835 66%,#43a047 66%,#43a047 100%)">
      <div class="score-fill" style="width:${pct}%"></div>
      <span class="score-num">${score != null ? score : '--'}</span>
    </div>
    <div class="score-label" style="color:${color};font-weight:600">${label}</div>
    ${dominant ? `<div class="dominant-factor">${dominant.icon} ${dominant.label}</div>` : ''}
  </div>`;
}

function _dominantFactor(cf) {
  if (!cf || !cf.components) return null;
  if (cf.safety_capped) return { label: 'Capped by safety limit', key: 'wind', icon: '⚠️' };
  let lowest = null, lowestScore = 11;
  const labels = { wind:'Wind', waves:'Waves', tide:'Tide', time:'Time', pressure:'Pressure', season:'Season', moon:'Moon', temperature:'Temperature', wind_direction:'Wind Dir' };
  for (const [k, v] of Object.entries(cf.components)) {
    const sc = v && v.score_100 != null ? v.score_100 : 10;
    if (sc < lowestScore) { lowestScore = sc; lowest = k; }
  }
  if (!lowest || lowestScore >= 70) return null;
  return { label: `Dragged by: ${labels[lowest] || lowest}`, key: lowest, icon: '⬇️' };
}

function _conditions(a, comps, cf) {
  const items = [
    { icon: 'hass:waves', label: 'Tide', value: cf.tide_phase || (comps.tide ? comps.tide.tide_phase : '--') },
    { icon: 'hass:weather-windy', label: 'Wind', value: a.current_wind_speed || (comps.wind ? comps.wind.wind_speed : '--') },
    { icon: 'hass:wave', label: 'Waves', value: a.current_wave_height || (comps.waves ? comps.waves.wave_height : '--') },
    { icon: 'hass:thermometer', label: 'Temp', value: a.current_temperature || (comps.temperature ? comps.temperature.temperature : '--') },
    { icon: 'hass:gauge', label: 'Pressure', value: comps.pressure ? comps.pressure.pressure_delta : '--' },
    { icon: 'hass:timeline', label: 'Swell', value: a.current_swell_period_s != null ? a.current_swell_period_s + ' s' : '--' },
  ];
  return `<div class="conditions">${items.map(i => `<div class="c-item"><ha-icon icon="${i.icon}"></ha-icon><div><div class="c-label">${i.label}</div><div class="c-value">${i.value}</div></div></div>`).join('')}</div>`;
}

function _chart(comps) {
  return `<div class="chart-section">
    <div class="section-title">Tide & Score — Next 48h</div>
    <div class="chart-fallback">Enable raw data in integration options to see chart</div>
  </div>`;
}

function _forecast(a) {
  const today = a.remainder_of_today_periods || {};
  const next5 = a.next_5_day_periods || {};
  const periods = [];
  const pMap = { period_00_06:'00-06h', period_06_12:'06-12h', period_12_18:'12-18h', period_18_24:'18-24h', dawn:'Dawn', dusk:'Dusk' };
  for (const [d, pmap] of Object.entries(today))
    for (const [pn, pd] of Object.entries(pmap))
      periods.push({ date:'Today', period: pMap[pn]||pn, score: pd.score_100 });
  for (const [d, pmap] of Object.entries(next5)) {
    const label = new Date(d).toLocaleDateString(undefined, {month:'short', day:'numeric'});
    for (const [pn, pd] of Object.entries(pmap))
      periods.push({ date: label, period: pMap[pn]||pn, score: pd.score_100 });
  }
  const slice = periods.slice(0, 8);
  if (!slice.length) return `<div class="forecast-section"><div class="section-title">Period Forecasts</div><div class="empty-data">No forecast data</div></div>`;
  const barColor = s => s >= 70 ? '#43a047' : s >= 50 ? '#fdd835' : '#e53935';
  return `<div class="forecast-section">
    <div class="section-title">Period Forecasts</div>
    ${slice.map(p => `<div class="f-row"><span class="f-date">${p.date}</span><span class="f-period">${p.period}</span><div class="f-bar-wrap"><div class="f-bar" style="width:${p.score||0}%;background:${barColor(p.score)}"></div></div><span class="f-score" style="color:${barColor(p.score)}">${p.score != null ? p.score : '--'}</span></div>`).join('')}
  </div>`;
}

function _footer(profile) {
  const name = profile.common_name || '';
  const sci = profile.scientific_name || '';
  return `<div class="footer">${name ? `${name}${sci ? ` (${sci})` : ''} · ` : ''}Open-Meteo</div>`;
}

/* ---- Card element (native Custom Element) ---- */
class OceanFishingCard extends HTMLElement {
  set hass(hass) { this._hass = hass; this._render(); }
  set config(config) { this._config = config; this._render(); }
  get hass() { return this._hass; }
  get config() { return this._config; }

  constructor() {
    super();
    this._hass = null;
    this._config = null;
    this._shadow = this.attachShadow({ mode: 'open' });
  }

  setConfig(config) {
    if (!config || !config.entity) throw new Error('Entity must be specified');
    this._config = config;
  }

  _render() {
    const root = this._shadow;
    if (!this._config || !this._config.entity) {
      root.innerHTML = '<ha-card><div class="config-msg">Configure entity in card settings</div></ha-card>';
      return;
    }
    if (!this._hass) {
      root.innerHTML = '<ha-card><div class="loading">Loading...</div></ha-card>';
      return;
    }
    const stateObj = this._hass.states[this._config.entity];
    if (!stateObj) {
      root.innerHTML = `<ha-card><div class="unavailable"><ha-icon icon="hass:alert"></ha-icon><div>Sensor unavailable</div><div class="hint">Check entity: ${this._config.entity}</div></div></ha-card>`;
      return;
    }
    // Ensure styles are present
    if (!root.querySelector('style')) {
      const style = document.createElement('style');
      style.textContent = styles;
      root.prepend(style);
    }
    // If ha-card is not available, use a simple div
    root.innerHTML = (root.querySelector('style')?.outerHTML || `<style>${styles}</style>`) + buildCard(stateObj, stateObj.attributes);
  }

  getCardSize() { return 3; }
}

/* ---- Register ---- */
customElements.define('ocean-fishing-card', OceanFishingCard);

window.customCards = window.customCards || [];
window.customCards.push({
  type: 'ocean-fishing-card',
  name: 'Ocean Fishing Assistant',
  description: 'Fishing conditions dashboard — score, tide, forecast',
  preview: false,
});
