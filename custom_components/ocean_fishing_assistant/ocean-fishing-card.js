// Ocean Fishing Assistant — Lovelace Custom Card
// Single-file LitElement, no build step. Requires HA 2024.x+
// Register as custom:ocean-fishing-card in Lovelace dashboard

// Resolve LitElement, html, and css from HA's module system.
// In HA 2026.x, these are available via window.LitElement after the frontend loads.
// Fallback: use the prototype chain of known HA custom elements.
let LitElement = window.LitElement;
let html = window.html;
let css = window.css;

if (!LitElement || typeof html !== 'function' || typeof css !== 'function') {
  // Fallback: try prototype chain of known HA custom elements
  try {
    const base = customElements.get('home-assistant-main')
      || customElements.get('ha-panel-lovelace')
      || customElements.get('hui-view');
    if (base) {
      const proto = Object.getPrototypeOf(Object.getPrototypeOf(base));
      LitElement = proto.constructor;
      if (typeof html !== 'function') html = proto.html;
      if (typeof css !== 'function') css = proto.css;
    }
  } catch (_) { /* fallback failed */ }
}

if (typeof html !== 'function' || typeof css !== 'function') {
  console.warn('Ocean Fishing Card: Could not resolve html/css template functions — card not registered.');
} else {
  // -- start of card implementation (guarded by successful LitElement resolution) --

class OceanFishingCard extends LitElement {
  static get properties() {
    return {
      _hass: { type: Object },
      config: { type: Object },
      _error: { type: String },
    };
  }

  static getConfigElement() {
    return document.createElement('ocean-fishing-card-editor');
  }

  static getStubConfig() {
    return {};
  }

  setConfig(config) {
    if (!config.entity) {
      throw new Error('Entity must be specified');
    }
    this.config = config;
  }

  get _stateObj() {
    return this._hass ? this._hass.states[this.config.entity] : null;
  }

  get _attrs() {
    return this._stateObj ? this._stateObj.attributes : {};
  }

  get _current() {
    return this._attrs.current_forecast || {};
  }

  get _score() {
    return this._stateObj ? parseInt(this._stateObj.state, 10) : null;
  }

  get _scoreLabel() {
    const s = this._score;
    if (s === null) return '--';
    if (s >= 70) return 'Excellent';
    if (s >= 50) return 'Fair';
    return 'Poor';
  }

  get _scoreColor() {
    const s = this._score;
    if (s === null) return 'var(--secondary-text-color)';
    if (s >= 70) return '#43a047';
    if (s >= 50) return '#fdd835';
    return '#e53935';
  }

  get _dominantFactor() {
    const cf = this._current;
    if (!cf || !cf.components) return null;
    const comps = cf.components;

    // If safety-capped, show that first
    if (cf.safety_capped) return { label: 'Capped by safety limit', key: 'wind', icon: '⚠️' };

    // Find lowest scoring component
    let lowest = null;
    let lowestScore = 11;
    for (const [key, val] of Object.entries(comps)) {
      const sc = val && val.score_100 != null ? val.score_100 : 10;
      if (sc < lowestScore) {
        lowestScore = sc;
        lowest = key;
      }
    }
    if (!lowest || lowestScore >= 70) return null;
    const labels = { wind: 'Wind', waves: 'Waves', tide: 'Tide', time: 'Time', pressure: 'Pressure',
                     season: 'Season', moon: 'Moon', temperature: 'Temperature', wind_direction: 'Wind Dir' };
    return { label: `Dragged by: ${labels[lowest] || lowest}`, key: lowest, icon: '⬇️' };
  }

  get _conditions() {
    const cf = this._current;
    const comps = cf && cf.components ? cf.components : {};
    const a = this._attrs;
    return [
      { icon: 'hass:waves', label: 'Tide', value: comps.tide ? comps.tide.tide_phase : (a.current_forecast ? a.current_forecast.tide_phase : '--') },
      { icon: 'hass:weather-windy', label: 'Wind', value: a.current_wind_speed || (comps.wind ? comps.wind.wind_speed : '--') },
      { icon: 'hass:wave', label: 'Waves', value: a.current_wave_height || (comps.waves ? comps.waves.wave_height : '--') },
      { icon: 'hass:thermometer', label: 'Temp', value: a.current_temperature || (comps.temperature ? comps.temperature.temperature : '--') },
      { icon: 'hass:gauge', label: 'Pressure', value: comps.pressure ? comps.pressure.pressure_delta : '--' },
      { icon: 'hass:timeline', label: 'Swell', value: a.current_swell_period_s != null ? `${a.current_swell_period_s} s` : (comps.waves ? '--' : '--') },
    ];
  }

  get _forecastPeriods() {
    const today = this._attrs.remainder_of_today_periods || {};
    const next5 = this._attrs.next_5_day_periods || {};
    const periods = [];
    const now = new Date();

    for (const [dateKey, pmap] of Object.entries(today)) {
      for (const [pname, pdata] of Object.entries(pmap)) {
        periods.push({ date: 'Today', period: this._periodLabel(pname), score: pdata.score_100, safety: pdata.safety });
      }
    }

    for (const [dateKey, pmap] of Object.entries(next5)) {
      for (const [pname, pdata] of Object.entries(pmap)) {
        const d = new Date(dateKey);
        const label = d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
        periods.push({ date: label, period: this._periodLabel(pname), score: pdata.score_100, safety: pdata.safety });
      }
    }
    return periods.slice(0, 8);
  }

  _periodLabel(pname) {
    const map = { period_00_06: '00-06h', period_06_12: '06-12h', period_12_18: '12-18h', period_18_24: '18-24h',
                  dawn: 'Dawn', dusk: 'Dusk' };
    return map[pname] || pname;
  }

  render() {
    if (!this.config || !this.config.entity) {
      return html`<ha-card><div class="config-msg">Configure entity in card settings</div></ha-card>`;
    }
    if (!this._hass) {
      return html`<ha-card><div class="loading"><ha-circular-progress indeterminate></ha-circular-progress></div></ha-card>`;
    }
    if (!this._stateObj) {
      return html`<ha-card>
        <div class="unavailable">
          <ha-icon icon="hass:alert"></ha-icon>
          <div>Sensor unavailable</div>
          <div class="hint">Check entity ID: ${this.config.entity}</div>
        </div>
      </ha-card>`;
    }

    const score = this._score;
    const cf = this._current;
    const comps = cf && cf.components ? cf.components : {};
    const dominant = this._dominantFactor;
    const conditions = this._conditions;
    const periods = this._forecastPeriods;
    const profile = this._attrs.profile_used || {};
    const moonName = this._attrs.moon_phase_name || (comps.moon ? comps.moon.moon_phase_name : null);

    return html`
      <ha-card>
        <!-- Header -->
        <div class="header">
          <div class="header-left">
            <span class="header-icon">🎣</span>
            <span class="header-title">${this.config.title || 'Ocean Fishing'}</span>
          </div>
          ${moonName ? html`<span class="header-moon">🌙 ${moonName}</span>` : ''}
        </div>

        <!-- Score bar (hero element) -->
        <div class="score-section">
          <div class="score-bar" style="background: linear-gradient(to right, #e53935 0%, #e53935 33%, #fdd835 33%, #fdd835 66%, #43a047 66%, #43a047 100%);">
            <div class="score-fill" style="width: ${100 - (score || 0)}%;"></div>
            <span class="score-num">${score != null ? score : '--'}</span>
          </div>
          <div class="score-label" style="color: ${this._scoreColor}; font-weight: 600;">${this._scoreLabel}</div>
          ${dominant ? html`<div class="dominant-factor">${dominant.icon} ${dominant.label}</div>` : ''}
        </div>

        <!-- Current conditions (2-column grid) -->
        <div class="conditions">
          ${conditions.map(c => html`
            <div class="condition-item">
              <ha-icon icon="${c.icon}"></ha-icon>
              <div>
                <div class="condition-label">${c.label}</div>
                <div class="condition-value">${c.value != null ? c.value : 'No data'}</div>
              </div>
            </div>
          `)}
        </div>

        <!-- Tide + Score chart -->
        <div class="chart-section">
          <div class="section-title">Tide & Score — Next 48h</div>
          ${this._renderChart()}
        </div>

        <!-- Period forecasts -->
        <div class="forecast-section">
          <div class="section-title">Period Forecasts</div>
          ${periods.length > 0 ? html`
            <div class="forecast-table">
              ${periods.map(p => html`
                <div class="forecast-row">
                  <span class="forecast-date">${p.date}</span>
                  <span class="forecast-period">${p.period}</span>
                  <div class="forecast-bar-wrapper">
                    <div class="forecast-bar" style="width: ${p.score || 0}%; background: ${this._barColor(p.score)};"></div>
                  </div>
                  <span class="forecast-score" style="color: ${this._barColor(p.score)};">${p.score != null ? p.score : '--'}</span>
                </div>
              `)}
            </div>
          ` : html`<div class="empty-data">No forecast data available</div>`}
        </div>

        <!-- Footer -->
        <div class="footer">
          ${profile.common_name ? html`${profile.common_name}${profile.scientific_name ? ` (${profile.scientific_name})` : ''} · ` : ''}
          Open-Meteo
        </div>
      </ha-card>
    `;
  }

  _renderChart() {
    // Try ha-chart-base; fall back to text table on error
    try {
      const perTs = this._getPerTimestampData();
      if (!perTs || perTs.length < 2) {
        return this._renderChartFallback(perTs);
      }
      return html`<div id="chart-container" style="height: 100px;"></div>`;
    } catch (e) {
      return this._renderChartFallback(this._getPerTimestampData());
    }
  }

  _getPerTimestampData() {
    // Build per-timestamp data from sensor attributes
    // Requires raw data exposure to be enabled
    const pts = this._attrs.per_timestamp_forecasts;
    if (!pts || !Array.isArray(pts)) {
      // Fallback: use period forecasts for simpler text table
      return this._forecastPeriods.map(p => ({
        time: p.date + ' ' + p.period,
        score: p.score,
      }));
    }
    return pts.slice(0, 48).map(e => ({
      time: e.timestamp,
      score: e.score_100,
      tide_phase: e.tide_phase,
    }));
  }

  _renderChartFallback(data) {
    if (!data || data.length === 0) {
      return html`<div class="chart-fallback">No tide data</div>`;
    }
    return html`
      <div class="chart-fallback">
        ${data.slice(0, 6).map(d => html`
          <div class="chart-row">
            <span class="chart-row-time">${d.time}</span>
            <span class="chart-row-score" style="color: ${this._barColor(d.score)};">${d.score != null ? `${d.score}/100` : '--'}</span>
          </div>
        `)}
        ${data.length > 6 ? html`<div class="chart-more">+${data.length - 6} more periods</div>` : ''}
      </div>
    `;
  }

  _barColor(score) {
    if (score == null) return 'var(--secondary-text-color)';
    if (score >= 70) return '#43a047';
    if (score >= 50) return '#fdd835';
    return '#e53935';
  }

  firstUpdated() {
    this._renderChartAsync();
  }

  updated(changedProps) {
    if (changedProps.has('_hass') || changedProps.has('config')) {
      this._renderChartAsync();
    }
  }

  async _renderChartAsync() {
    await this.updateComplete;
    const container = this.shadowRoot && this.shadowRoot.getElementById('chart-container');
    if (!container) return;

    const perTs = this._getPerTimestampData();
    if (!perTs || perTs.length < 2 || !perTs[0].tide_phase) {
      return;
    }

    try {
      // Attempt to render ha-chart-base
      const chartBase = customElements.get('ha-chart-base');
      if (!chartBase) {
        container.innerHTML = '';
        container.appendChild(this._buildChartFallback(perTs));
        return;
      }

      const chart = document.createElement('ha-chart-base');
      chart.data = this._buildChartData(perTs);
      chart.setAttribute('theme', 'dark');
      container.innerHTML = '';
      container.appendChild(chart);
    } catch (e) {
      container.innerHTML = '';
      container.appendChild(this._buildChartFallback(perTs));
    }
  }

  _buildChartData(perTs) {
    return {
      datasets: [
        {
          label: 'Tide',
          data: perTs.map((e, i) => ({
            x: new Date(e.time || i).getTime(),
            y: e.tide_phase === 'high' ? 100 : e.tide_phase === 'rising' ? 75 : e.tide_phase === 'falling' ? 25 : 50,
          })),
          borderColor: '#2196F3',
          backgroundColor: 'rgba(33, 150, 243, 0.1)',
          fill: true,
          tension: 0.3,
          pointRadius: 2,
        },
        {
          label: 'Score',
          data: perTs.map((e, i) => ({
            x: new Date(e.time || i).getTime(),
            y: e.score || 0,
          })),
          backgroundColor: perTs.map(e => {
            const s = e.score;
            if (s >= 70) return 'rgba(67, 160, 71, 0.6)';
            if (s >= 50) return 'rgba(253, 216, 53, 0.6)';
            return 'rgba(229, 57, 53, 0.6)';
          }),
          type: 'bar',
          order: 1,
        },
      ],
    };
  }

  _buildChartFallback(perTs) {
    const div = document.createElement('div');
    div.className = 'chart-fallback';
    const slice = perTs.slice(0, 6);
    for (const d of slice) {
      const row = document.createElement('div');
      row.className = 'chart-row';
      const time = document.createElement('span');
      time.className = 'chart-row-time';
      time.textContent = d.time;
      const score = document.createElement('span');
      score.className = 'chart-row-score';
      score.textContent = d.score != null ? `${d.score}/100` : '--';
      score.style.color = this._barColor(d.score);
      row.appendChild(time);
      row.appendChild(score);
      div.appendChild(row);
    }
    if (perTs.length > 6) {
      const more = document.createElement('div');
      more.className = 'chart-more';
      more.textContent = `+${perTs.length - 6} more periods`;
      div.appendChild(more);
    }
    return div;
  }

  static get styles() {
    return css`
      :host { display: block; }
      ha-card {
        padding: 16px;
        font-family: var(--primary-font-family, Roboto);
        font-size: 14px;
        line-height: 1.4;
      }

      /* Config / empty / unavailable states */
      .config-msg, .loading, .unavailable {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 32px 16px;
        color: var(--secondary-text-color);
        text-align: center;
        gap: 8px;
      }
      .config-msg { min-height: 80px; }
      .loading { min-height: 60px; }
      .unavailable ha-icon { --mdc-icon-size: 32px; color: var(--error-color, #e53935); }
      .hint { font-size: 12px; color: var(--secondary-text-color); opacity: 0.7; }

      /* Header */
      .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--divider-color, rgba(0,0,0,0.08));
      }
      .header-left { display: flex; align-items: center; gap: 8px; }
      .header-icon { font-size: 20px; }
      .header-title { font-weight: 600; font-size: 15px; color: var(--primary-text-color); }
      .header-moon { font-size: 13px; color: var(--secondary-text-color); }

      /* Score bar */
      .score-section { margin-bottom: 12px; }
      .score-bar {
        position: relative;
        height: 48px;
        border-radius: 24px;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
      }
      .score-fill {
        position: absolute;
        right: 0;
        top: 0;
        bottom: 0;
        background: rgba(255,255,255,0.4);
        border-radius: 0 24px 24px 0;
        transition: width 0.3s ease;
      }
      .score-num {
        position: relative;
        font-size: 22px;
        font-weight: 700;
        color: #fff;
        text-shadow: 0 1px 3px rgba(0,0,0,0.3);
        z-index: 1;
      }
      .score-label {
        text-align: center;
        margin-top: 4px;
        font-size: 13px;
      }
      .dominant-factor {
        text-align: center;
        font-size: 12px;
        color: var(--warning-color, #e53935);
        margin-top: 2px;
      }

      /* Conditions grid */
      .conditions {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 6px 12px;
        margin-bottom: 12px;
        padding: 10px;
        background: var(--secondary-background-color, rgba(0,0,0,0.02));
        border-radius: 10px;
      }
      .condition-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 13px;
      }
      .condition-item ha-icon {
        --mdc-icon-size: 18px;
        color: var(--state-icon-color, #666);
        flex-shrink: 0;
      }
      .condition-label { font-size: 11px; color: var(--secondary-text-color); line-height: 1.2; }
      .condition-value { font-weight: 500; color: var(--primary-text-color); font-size: 13px; }

      /* Chart */
      .chart-section { margin-bottom: 12px; }
      .section-title { font-size: 12px; font-weight: 600; color: var(--secondary-text-color); margin-bottom: 6px; }
      .chart-fallback {
        font-size: 12px;
        border: 1px dashed var(--divider-color, #ccc);
        border-radius: 8px;
        padding: 8px;
      }
      .chart-row {
        display: flex;
        justify-content: space-between;
        padding: 3px 0;
        border-bottom: 1px solid var(--divider-color, rgba(0,0,0,0.05));
      }
      .chart-row:last-child { border-bottom: none; }
      .chart-row-time { color: var(--secondary-text-color); font-size: 11px; }
      .chart-row-score { font-weight: 500; }
      .chart-more { text-align: center; font-size: 11px; color: var(--secondary-text-color); padding: 4px; }

      /* Forecast table */
      .forecast-section { margin-bottom: 8px; }
      .forecast-table { }
      .forecast-row {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 4px 0;
        font-size: 12px;
        border-bottom: 1px solid var(--divider-color, rgba(0,0,0,0.05));
      }
      .forecast-row:last-child { border-bottom: none; }
      .forecast-date { color: var(--secondary-text-color); width: 52px; flex-shrink: 0; }
      .forecast-period { color: var(--secondary-text-color); width: 52px; flex-shrink: 0; font-size: 11px; }
      .forecast-bar-wrapper { flex: 1; height: 10px; background: var(--secondary-background-color, #f0f0f0); border-radius: 5px; overflow: hidden; }
      .forecast-bar { height: 100%; border-radius: 5px; transition: width 0.3s ease; }
      .forecast-score { font-weight: 600; width: 28px; text-align: right; }
      .empty-data { font-size: 12px; color: var(--secondary-text-color); padding: 8px; text-align: center; }

      /* Footer */
      .footer {
        border-top: 1px solid var(--divider-color, rgba(0,0,0,0.08));
        padding-top: 8px;
        font-size: 11px;
        color: var(--secondary-text-color);
        text-align: center;
        opacity: 0.7;
      }
    `;
  }
}

// Card editor for Lovelace UI config
class OceanFishingCardEditor extends LitElement {
  static get properties() {
    return { _hass: {}, _config: {}, _error: {} };
  }

  setConfig(config) {
    this._config = config || {};
  }

  render() {
    if (!this._hass) return html``;

    // Filter to sensors from our domain
    const entities = Object.keys(this._hass.states).filter(eid =>
      eid.startsWith('sensor.') && this._hass.states[eid].attributes &&
      (eid.includes('rosia_bay') || eid.includes('mackeral') || eid.includes('fishing'))
    );

    return html`
      <div class="editor">
        <paper-dropdown-menu label="Entity" .disabled=${entities.length === 0}>
          <paper-listbox slot="dropdown-content" .selected=${entities.indexOf(this._config.entity)}
            @iron-select=${e => this._valueChanged('entity', entities[e.detail.selected])}>
            ${entities.length > 0
              ? entities.map(eid => html`<paper-item>${eid}</paper-item>`)
              : html`<paper-item>No fishing sensor found</paper-item>`
            }
          </paper-listbox>
        </paper-dropdown-menu>
        <paper-input label="Title (optional)" .value=${this._config.title || ''}
          @value-changed=${e => this._valueChanged('title', e.detail.value)}>
        </paper-input>
      </div>
    `;
  }

  _valueChanged(key, value) {
    const newConfig = { ...this._config, [key]: value };
    if (!value) delete newConfig[key];
    this._config = newConfig;
    const event = new CustomEvent('config-changed', {
      detail: { config: newConfig },
      bubbles: true,
      composed: true,
    });
    this.dispatchEvent(event);
  }
}

// Register card and editor
customElements.define('ocean-fishing-card', OceanFishingCard);
customElements.define('ocean-fishing-card-editor', OceanFishingCardEditor);

// Register with HA's custom card registry
window.customCards = window.customCards || [];
window.customCards.push({
  type: 'ocean-fishing-card',
  name: 'Ocean Fishing Assistant',
  description: 'Fishing conditions dashboard — score, tide, forecast',
  preview: false,
});

} // end of LitElement guard
