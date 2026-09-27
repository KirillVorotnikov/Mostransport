/* global window */
const TramForecast = (() => {
  const ROUTES = [1, 5, 7, 11, 12, 17, 25, 26, 28, 50];
  const MAP_ROUTES = new Set(["1", "7", "11", "12", "17", "25", "26", "28", "50"]);
  const state = {
    loaded: false,
    failed: false,
    facts: new Map(), // Сюда складываем исторические факты
    models: {
      selective: new Map(),
      lgbm: new Map(),
    },
    activeModel: "selective", // Текущая выбранная модель
    history: new Map(),
    effects: null,
  };
  let pending = null;

  function parseSemicolon(text) {
    const lines = String(text || "").replace(/^\uFEFF/, "").trim().split(/\r?\n/);
    if (!lines.length) return [];
    const headers = lines[0].split(";");
    return lines.slice(1).filter(Boolean).map((line) => {
      const cells = line.split(";");
      const row = {};
      headers.forEach((header, index) => {
        row[header] = cells[index] ?? "";
      });
      return row;
    });
  }

  function cellKey(route, date, hour) {
    return `${Number(route)}|${date}|${Number(hour)}`;
  }

  function historyKey(route, weekday, hour) {
    return `${Number(route)}|${weekday}|${Number(hour)}`;
  }

  function putCell(route, date, hour, value, status, targetMap) {
    const key = cellKey(route, date, hour);
    targetMap.set(key, { value, status });
    if (status === "fact") {
      const weekday = weekdayOf(date);
      const bucketKey = historyKey(route, weekday, hour);
      const bucket = state.history.get(bucketKey) || [];
      bucket.push({ date, value });
      state.history.set(bucketKey, bucket);
    }
  }

  function weekdayOf(iso) {
    const [year, month, day] = iso.split("-").map(Number);
    return new Date(year, month - 1, day).getDay();
  }

  function formatDate(year, month, day) {
    return `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
  }

  function daysInMonth(year, month) {
    return new Date(year, month, 0).getDate();
  }

  function median(values) {
    if (!values.length) return null;
    const sorted = [...values].sort((left, right) => left - right);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  }

  async function load() {
    if (state.loaded || state.failed) return state;
    if (pending) return pending;
    pending = Promise.all([
      fetch("./data/forecast_selective.csv", { cache: "no-store" }).then((r) => r.ok ? r.text() : "").catch(() => ""),
      fetch("./data/forecast_lgbm.csv", { cache: "no-store" }).then((r) => r.ok ? r.text() : "").catch(() => ""),
      fetch("./data/kaggle_mstrans/labels_day_train.csv", { cache: "no-store" }).then((response) => response.text()),
      fetch("./data/kaggle_mstrans/labels_day_test.csv", { cache: "no-store" }).then((response) => response.text()),
      fetch("./data/kaggle_enrichment/scenario_effects.json", { cache: "no-store" }).then((response) => response.json()),
    ]).then(([selectiveText, lgbmText, trainText, testText, effects]) => {
      if (selectiveText) {
        parseSemicolon(selectiveText).forEach((row) => {
          putCell(row.route, row.date, row.hour, Number(row.prediction), "forecast", state.models.selective);
        });
      }
      if (lgbmText) {
        parseSemicolon(lgbmText).forEach((row) => {
          putCell(row.route, row.date, row.hour, Number(row.prediction), "forecast", state.models.lgbm);
        });
      }
      [trainText, testText].forEach((text) => {
        parseSemicolon(text).forEach((row) => {
          putCell(row.route, row.date, row.hour, Number(row.boardings), "fact", state.facts);
        });
      });
      state.effects = effects;
      state.loaded = true;
      return state;
    }).catch((error) => {
      state.failed = true;
      pending = null;
      throw error;
    });
    return pending;
  }

  function cell(route, date, hour) {
    const key = cellKey(route, date, hour);
    const fact = state.facts.get(key);
    if (fact) return fact; // Факты всегда в приоритете
    const modelMap = state.models[state.activeModel] || state.models.selective;
    return modelMap.get(key) || { value: null, status: "missing" };
  }

  function typical(route, date, hour) {
    const bucket = state.history.get(historyKey(route, weekdayOf(date), hour)) || [];
    const values = bucket.filter((item) => item.date !== date).map((item) => item.value);
    return values.length >= 3 ? median(values) : null;
  }

  function tone(value, usual) {
    if (value == null) return "missing";
    if (usual == null || usual <= 0) return "ok";
    const ratio = value / usual;
    if (ratio > 1.15) return "hot";
    if (ratio > 1) return "warm";
    return "ok";
  }

  function formatCount(value) {
    if (value == null || Number.isNaN(value)) return "—";
    return Math.round(value).toLocaleString("ru-RU");
  }

  function hourRange(hourFrom, hourTo) {
    const from = Math.max(0, Math.min(23, Number(hourFrom) || 0));
    const to = Math.max(0, Math.min(23, Number(hourTo) || 0));
    const start = Math.min(from, to);
    const end = Math.max(from, to);
    return Array.from({ length: end - start + 1 }, (_, index) => start + index);
  }

  function statusOf(items) {
    if (!items.length) return { code: "missing", label: "нет данных" };
    const known = items.filter((item) => item.status !== "missing");
    if (!known.length) return { code: "missing", label: "нет данных" };
    const facts = known.every((item) => item.status === "fact");
    const forecasts = known.every((item) => item.status === "forecast");
    const complete = known.length === items.length;
    if (facts && complete) return { code: "fact", label: "факт" };
    if (facts) return { code: "partial-fact", label: "факт, не все часы" };
    if (forecasts && complete) return { code: "forecast", label: "прогноз" };
    if (forecasts) return { code: "partial-forecast", label: "прогноз, не все часы" };
    return { code: "estimate", label: "оценка" };
  }

  function blankRow(period, route) {
    return {
      period,
      route: String(route),
      value: null,
      usual: null,
      status: "missing",
      statusLabel: "нет данных",
      covered: 0,
      expected: 0,
      tone: "missing",
    };
  }

  function rollup(period, route, dates, hours) {
    const items = [];
    dates.forEach((date) => {
      hours.forEach((hour) => items.push(cell(route, date, hour)));
    });
    const known = items.filter((item) => item.value != null && Number.isFinite(item.value));
    const status = statusOf(items);
    if (!known.length) return blankRow(period, route);
    const usualParts = [];
    dates.forEach((date) => {
      hours.forEach((hour) => {
        const usual = typical(route, date, hour);
        if (cell(route, date, hour).value != null && usual != null) usualParts.push(usual);
      });
    });
    const value = known.reduce((sum, item) => sum + item.value, 0);
    const usual = usualParts.length ? usualParts.reduce((sum, item) => sum + item, 0) : null;
    return {
      period,
      route: String(route),
      value,
      usual,
      status: status.code,
      statusLabel: status.label,
      covered: known.length,
      expected: items.length,
      tone: tone(value, usual),
    };
  }

  function rowsFor(view) {
    const route = Number(view.route);
    const hours = hourRange(view.hourFrom, view.hourTo);
    const [year, month] = view.date.split("-").map(Number);
    if (view.horizon === "month") {
      return Array.from({ length: daysInMonth(year, month) }, (_, index) => {
        const date = formatDate(year, month, index + 1);
        return rollup(date, route, [date], hours);
      });
    }
    if (view.horizon === "year") {
      return Array.from({ length: 12 }, (_, index) => {
        const monthNumber = index + 1;
        const dates = Array.from({ length: daysInMonth(year, monthNumber) }, (_, day) => formatDate(year, monthNumber, day + 1));
        return rollup(`${year}-${String(monthNumber).padStart(2, "0")}`, route, dates, hours);
      });
    }
    return hours.map((hour) => {
      const item = cell(route, view.date, hour);
      const usual = item.value == null ? null : typical(route, view.date, hour);
      const status = item.value == null ? { code: "missing", label: "нет данных" } : statusOf([item]);
      return {
        period: `${view.date} ${String(hour).padStart(2, "0")}:00`,
        hour,
        route: String(route),
        value: item.value,
        usual,
        status: status.code,
        statusLabel: status.label,
        covered: item.value == null ? 0 : 1,
        expected: 1,
        tone: tone(item.value, usual),
      };
    });
  }

  function parseManual(value) {
    if (value == null || String(value).trim() === "") return null;
    const number = Number(String(value).replace(",", "."));
    if (!Number.isFinite(number) || number < 0.5 || number > 2) return null;
    return Math.round(number * 1000) / 1000;
  }

  function resolveScenario(route, scenario) {
    const weatherName = scenario.weather === "heavy" ? "heavy" : scenario.weather === "rain" ? "rain" : scenario.weather === "snow" ? "snow" : null;
    const weather = weatherName ? state.effects?.weather?.[weatherName]?.[String(route)] : null;
    const eventManual = scenario.eventOn ? parseManual(scenario.eventCoeff) : null;
    const seasonManual = scenario.seasonMode === "custom" ? parseManual(scenario.seasonCoeff) : null;
    if (eventManual != null || seasonManual != null) {
      const coefficient = eventManual ?? seasonManual;
      return {
        coefficient,
        confidence: "гипотеза",
        apply: coefficient !== 1,
        note: "Свой коэффициент — гипотеза. Он заменяет автоматическую поправку и не перемножается с погодой или сезоном.",
      };
    }
    if ((scenario.eventOn && eventManual == null) || (scenario.seasonMode === "custom" && seasonManual == null)) {
      return {
        coefficient: 1,
        confidence: "эффект не оценён",
        apply: false,
        note: "Свой коэффициент не применён: нужно число от 0,50 до 2,00. Автоматические эффекты при этом не перемножаются.",
      };
    }
    if (weather?.apply) {
      const label = weatherName === "heavy" ? "Сильный дождь" : weatherName === "snow" ? "Снег" : "Дождь";
      return {
        coefficient: weather.coefficient,
        confidence: weather.confidence,
        apply: true,
        note: `${label}: ${weather.targetHours} часов за ${weather.targetDays} дней января–октября 2025 против всех наблюдаемых часов того же маршрута. Это оценка, не точный эффект.`,
      };
    }
    if (weatherName) {
      return {
        coefficient: 1,
        confidence: "эффект не оценён",
        apply: false,
        note: "Для этой погоды и маршрута нет надёжной оценки. Коэффициент 1,00.",
      };
    }
    return {
      coefficient: 1,
      confidence: "базовый",
      apply: false,
      note: state.effects?.season?.note || "Базовый прогноз без дополнительной поправки.",
    };
  }

  function present(view) {
    const scenario = resolveScenario(view.route, view.scenario || {});
    const rows = rowsFor(view).map((row) => {
      const scenarioValue = row.value == null ? null : Math.round(row.value * scenario.coefficient);
      const baseValue = row.value == null ? null : Math.round(row.value);
      return { ...row, baseValue, scenarioValue };
    });
    const known = rows.filter((row) => row.baseValue != null);
    const totalBase = known.reduce((sum, row) => sum + row.baseValue, 0);
    const totalScenario = known.reduce((sum, row) => sum + row.scenarioValue, 0);
    const covered = rows.reduce((sum, row) => sum + row.covered, 0);
    const expected = rows.reduce((sum, row) => sum + row.expected, 0);
    const codes = new Set(known.map((row) => row.status));
    let label = "нет данных";
    if (known.length) {
      const factual = [...codes].every((code) => code === "fact" || code === "partial-fact");
      const predicted = [...codes].every((code) => code === "forecast" || code === "partial-forecast");
      const incomplete = covered < expected;
      if (factual) label = incomplete ? "факт, не все часы" : "факт";
      else if (predicted) label = incomplete ? "прогноз, не все часы" : "прогноз";
      else label = "оценка";
    }
    if (scenario.apply) label = scenario.confidence === "гипотеза" ? "сценарий, гипотеза" : "сценарий";
    return {
      rows,
      totalBase: known.length ? totalBase : null,
      totalScenario: known.length ? totalScenario : null,
      covered,
      expected,
      statusLabel: label,
      scenario,
      horizon: view.horizon,
    };
  }

  function peakHour(route, date, hourFrom, hourTo) {
    const rows = rowsFor({ route, date, horizon: "day", hourFrom, hourTo });
    return rows.filter((row) => row.value != null).sort((left, right) => right.value - left.value)[0] || null;
  }

  function watchlist(date, startHour, span) {
    return ROUTES.map((route) => {
      const options = [];
      for (let shift = 0; shift < span; shift += 1) {
        const hour = startHour + shift;
        const item = cell(route, date, hour);
        const usual = typical(route, date, hour);
        const deviation = item.value != null && usual != null && usual > 0 ? (item.value - usual) / usual : null;
        options.push({ route: String(route), date, hour, value: item.value, status: item.status, usual, deviation });
      }
      return options.sort((left, right) => (right.deviation ?? -Infinity) - (left.deviation ?? -Infinity))[0];
    }).sort((left, right) => (right.deviation ?? -Infinity) - (left.deviation ?? -Infinity));
  }

  function hasGeometry(route) {
    return MAP_ROUTES.has(String(route));
  }

  return {
    ROUTES,
    MAP_ROUTES,
    state,
    load,
    cell,
    typical,
    tone,
    formatCount,
    hourRange,
    present,
    peakHour,
    watchlist,
    hasGeometry,
    parseManual,
  };
})();

window.TramForecast = TramForecast;
