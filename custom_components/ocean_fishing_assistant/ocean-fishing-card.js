// Ocean Fishing Assistant — Lovelace Custom Card
// Minimal implementation for debugging

console.log('[OceanFishingCard] Module loaded');

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
  .rtide { font-size: 11px; color: var(--secondary-text-color); width: 14px; text-align: center; }
  .empty { font-size: 12px; color: var(--secondary-text-color); padding: 8px; text-align: center; }

  .footer { border-top: 1px solid var(--divider-color, rgba(0,0,0,0.08)); padding-top: 6px; margin-top: 4px; font-size: 11px; color: var(--secondary-text-color); text-align: center; opacity: 0.7; }
`;

function barColor(s) {
  if (s == null || s < 50) return '#e53935';
  if (s < 70) return '#fdd835';
  return '#43a047';
}

function buildCard(a, title) {
  const score = parseInt(a.state, 10);
  const attrs = a.attributes || {};
  const cf = attrs.current_forecast || {};
  const comps = cf.components || {};
  const moon = attrs.moon_phase_name || (comps.moon ? comps.moon.moon_phase_name : null);
  const profile = attrs.profile_used || {};
  const label = score >= 70 ? 'Excellent' : score >= 50 ? 'Fair' : 'Poor';
  const pct = 100 - Math.min(Math.max(score || 0, 0), 100);

  // Conditions
  const conds = [
    { i:'hass:waves', l:'Tide', v: cf.tide_phase || '--' },
    { i:'hass:weather-windy', l:'Wind', v: attrs.current_wind_speed || '--' },
    { i:'hass:wave', l:'Waves', v: attrs.current_wave_height || '--' },
    { i:'hass:thermometer', l:'Temp', v: attrs.current_temperature || '--' },
  ];

  // Forecast periods
  const today = attrs.remainder_of_today_periods || {};
  const next5 = attrs.next_5_day_periods || {};
  const pMap = { period_00_06:'00-06', period_06_12:'06-12', period_12_18:'12-18', period_18_24:'18-24' };
  const rows = [];
  for (const [periodName, periodData] of Object.entries(today))
    rows.push({ d:'Today', p: pMap[periodName]||periodName, s: periodData.score_100, t: periodData.tide_phase });
  for (const d of Object.keys(next5).sort().slice(0, 3)) {
    const label = new Date(d).toLocaleDateString(undefined, {month:'short', day:'numeric'});
    for (const [pn, pd] of Object.entries(next5[d]))
      rows.push({ d: label, p: pMap[pn]||pn, s: pd.score_100, t: pd.tide_phase });
  }
  const display = rows.slice(0, 8);

  return `
    <ha-card>
      <div class="card">
        <div class="header">
          <span class="header-title">🎣 ${title || 'Ocean Fishing'}</span>
          ${moon ? `<span class="header-moon">🌙 ${moon}</span>` : ''}
        </div>

        <div class="score-bar">
          <div class="score-fill" style="width:${pct}%"></div>
          <span class="score-num">${score}</span>
        </div>
        <div class="score-label" style="color:${barColor(score)}">${label} · ${score}/100</div>

        <div class="conditions">
          ${conds.map(c => `<div class="c-item"><ha-icon icon="${c.i}"></ha-icon><div><div class="c-label">${c.l}</div><div class="c-value">${c.v}</div></div></div>`).join('')}
        </div>

        ${display.length ? `
        <div class="section-title">Forecast</div>
        ${display.map(r => {
          const tide = r.t === 'rising' ? '↑' : r.t === 'falling' ? '↓' : '';
          return `<div class="row"><span class="rdate">${r.d}</span><span class="rper">${r.p}</span><div class="rbar-wrap"><div class="rbar" style="width:${Math.min(r.s||0,100)}%;background:${barColor(r.s)}"></div></div><span class="rscore" style="color:${barColor(r.s)}">${r.s != null ? r.s : '--'}</span>${tide ? `<span class="rtide">${tide}</span>` : ''}</div>`;
        }).join('')}
        ` : `<div class="empty">No forecast data</div>`}

        <div class="footer">${profile.common_name ? `${profile.common_name}${profile.scientific_name ? ` (${profile.scientific_name})` : ''} · ` : ''}Data: Open-Meteo, World Tides</div>
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

    console.log('[OceanFishingCard] Rendering with state:', stateObj.state, 'attrs:', Object.keys(stateObj.attributes));
    console.log('[OceanFishingCard] today periods:', Object.keys(stateObj.attributes.remainder_of_today_periods || {}).length);
    console.log('[OceanFishingCard] next5 days:', Object.keys(stateObj.attributes.next_5_day_periods || {}).length);

    const html = `<style>${styles}</style>` + buildCard(stateObj, this._config.title);
    this._shadow.innerHTML = html;
  }

  getCardSize() { return 3; }

  static async getConfigForm() {
    return {
      schema: [
        { name: 'entity', selector: { entity: { domain: 'sensor' } } },
        { name: 'title', selector: { text: {} } },
      ],
    };
  }
  static getStubConfig() { return {}; }
}

customElements.define('ocean-fishing-card', OceanFishingCard);
window.customCards = window.customCards || [];
window.customCards.push({ type: 'ocean-fishing-card', name: 'Ocean Fishing Assistant', description: 'Fishing conditions', preview: false });

console.log('[OceanFishingCard] Registered');
