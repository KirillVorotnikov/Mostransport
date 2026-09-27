/* global L, JSZip */

const MOSCOW = [55.7558, 37.62];
const KRASNOGORSK = [55.8204707, 37.3196942];
const KRASNOGORSK_INSET_PX = 30;
const MOSCOW_MKAD_BOUNDS = [
  [55.40, 36.90],
  [56.10, 39.40],
];
const MOSCOW_MAP_ZOOM = 10.5;
const LINE_HIT_WEIGHT = 18;
const LINE_HOVER_LINGER_MS = 240;
const DEFAULT_ENDPOINT = "https://apidata.mos.ru/v1/datasets/3221/features";
const PAGE_SIZE = 1000;
const MAP_STATE_KEY = "tram-map-state";

const state = {
  map: null,
  layers: new Map(),
  stopsLayer: null,
  feed: null,
  routes: [],
  selectedId: null,
  hoveredId: null,
  hoverTimer: 0,
  stationStopId: null,
  stationCards: new Map(),
  stationCardZ: 1200,
  lineScale: 1,
  showStops: true,
  showMapLoad: false,
  usingDemo: false,
  mapFramed: false,
  forecast: {
    model: "selective",
    horizon: "day",
    date: "2025-11-03",
    hourFrom: 0,
    hourTo: 23,
    selectedHour: 8,
    playing: false,
    playTimer: 0,
    segment: "",
    trips: 0,
    scenario: {
      weather: "base",
      eventOn: false,
      eventPlace: "",
      eventTime: "",
      eventCoeff: "",
      seasonMode: "base",
      seasonCoeff: "",
    },
  },
};

const $ = (selector) => document.querySelector(selector);

document.addEventListener("DOMContentLoaded", () => {
  setupMap();
  bindEvents();
  renderFeed(makeDemoFeed(), true);
  setStatus("Готово · набор 3221 GeoJSON");
  loadLocalGeoJson({ silent: true }).catch(() => {});
  if (window.TramFacts) {
    TramFacts.load().then(() => {
      if (!state.selectedId) return;
      const route = state.routes.find((item) => item.route_id === state.selectedId);
      if (route) renderDemand(route);
      const stops = stopsForRoute(state.selectedId);
      state.stationCards.forEach((card, stopId) => {
        const stop = stops.find((item) => item.stop_id === stopId);
        if (!stop || !route) return;
        const station = TramFacts.stationFor(stop.stop_name, route.route_short_name, stop.stop_lat, stop.stop_lon);
        fillStationCard(card, station, stop);
      });
    }).catch(() => {});
  }
  if (window.TramForecast) {
    TramForecast.load().then(() => {
      renderRouteList($("#route-search").value);
      const route = state.routes.find((item) => item.route_id === state.selectedId);
      if (route) renderDemand(route);
      syncForecastTimeDisplay(); // <-- ДОБАВИТЬ
    }).catch(() => {
      renderRouteList($("#route-search").value);
      const route = state.routes.find((item) => item.route_id === state.selectedId);
      if (route) renderDemand(route);
      syncForecastTimeDisplay(); // <-- ДОБАВИТЬ
    });
  } else {
    syncForecastTimeDisplay(); // <-- ДОБАВИТЬ
  }
});

function setupMap() {
  const cityBounds = L.latLngBounds(MOSCOW_MKAD_BOUNDS);
  state.map = L.map("map", {
    zoomControl: false,
    preferCanvas: true,
    zoomSnap: 0.5,
    minZoom: MOSCOW_MAP_ZOOM,
    maxZoom: 16,
    maxBounds: cityBounds,
    maxBoundsViscosity: 1,
    zoomAnimation: true,
    fadeAnimation: false,
    markerZoomAnimation: false,
    attributionControl: false,
    dragging: true,
    scrollWheelZoom: true,
    doubleClickZoom: true,
    boxZoom: false,
    keyboard: false,
    touchZoom: true,
  }).setView(MOSCOW, MOSCOW_MAP_ZOOM);
  L.control.zoom({ position: "topleft" }).addTo(state.map);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    updateWhenIdle: true,
    keepBuffer: 3,
  }).addTo(state.map);
  state.stopsLayer = L.layerGroup().addTo(state.map);
  window.addEventListener("resize", () => {
    state.map.invalidateSize({ pan: false });
  });
  setTimeout(() => state.map.invalidateSize({ pan: false }), 250);
}

function frameMap() {
  const zoom = MOSCOW_MAP_ZOOM;
  const size = state.map.getSize();
  if (!size.x || !size.y) return false;
  const anchor = state.map.project(KRASNOGORSK, zoom);
  const latitude = state.map.project(MOSCOW, zoom);
  const center = state.map.unproject(L.point(anchor.x - KRASNOGORSK_INSET_PX + size.x / 2, latitude.y), zoom);
  state.map.setView(center, zoom, { animate: false });
  return true;
}

function frameInitialMap() {
  if (state.mapFramed) return;
  const place = () => {
    if (state.mapFramed) return;
    state.map.invalidateSize({ pan: false });
    if (!frameMap()) {
      requestAnimationFrame(place);
      return;
    }
    state.mapFramed = true;
  };
  requestAnimationFrame(place);
}

function bindEvents() {
  $("#route-search").addEventListener("input", (event) => renderRouteList(event.target.value));
  $("#close-details").addEventListener("click", () => {
    $("#route-details").classList.add("hidden");
    state.selectedId = null;
    renderMap();
    renderRouteList($("#route-search").value);
  });
  $("#stations-toggle").addEventListener("click", () => {
    setStationsOpen(!$("#stations-panel").classList.contains("open"));
  });
  $("#line-size").addEventListener("input", (event) => {
    state.lineScale = Number(event.target.value) || 1;
    applyLineScale();
  });
  bindEdgeResize($("#sidebar-resize"), {
    target: $(".sidebar"),
    read: () => $(".sidebar").getBoundingClientRect().width,
    write: (width) => {
      const sidebar = $(".sidebar");
      sidebar.style.width = `${width}px`;
      sidebar.style.flexBasis = `${width}px`;
      state.map.invalidateSize({ pan: false });
    },
    next: (start, dx) => clamp(start + dx, 220, 560),
  });
  bindEdgeResize($("#route-card-resize"), {
    target: $("#route-details"),
    read: () => $("#route-details").getBoundingClientRect().width,
    write: (width) => {
      const card = $("#route-details");
      card.classList.add("is-sized");
      card.style.setProperty("--route-card-width", `${width}px`);
      $(".map-area").style.setProperty("--route-card-width", `${width}px`);
    },
    next: (start, dx) => clamp(start - dx, 280, Math.max(320, $(".map-area").getBoundingClientRect().width - 48)),
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "/" && document.activeElement.tagName !== "INPUT") {
      event.preventDefault();
      $("#route-search").focus();
    }
  });
  bindForecastControls();
  window.addEventListener("pagehide", () => {
    if (!state.usingDemo) saveMapState();
  });
}

async function loadData() {
  const button = $("#load-data");
  const endpoint = $("#endpoint").value.trim() || DEFAULT_ENDPOINT;
  const apiKey = $("#api-key").value.trim();
  button.disabled = true;
  button.innerHTML = '<span class="button-icon">⋯</span> Загружаю данные';
  setStatus("Получение данных…");

  try {
    const records = await fetchAllRows(endpoint, apiKey);
    const feed = buildGtfsFeed(records);
    if (!feed.routes.length) throw new Error("В ответе не найдены трамвайные маршруты");
    renderFeed(feed, false);
    showToast(`Загружено из data.mos.ru: ${feed.routes.length} маршрутов`);
    setStatus(`data.mos.ru · 3221 · ${feed.routes.length} трамвайных маршрутов`);
  } catch (error) {
    console.warn("Не удалось загрузить data.mos.ru", error);
    try {
      await loadLocalGeoJson({ silent: true });
      showToast("API недоступен — использован локальный GeoJSON");
    } catch {
      try {
        await loadLocalGtfs({ silent: true });
        showToast("API и GeoJSON недоступны — использована локальная GTFS-выгрузка");
      } catch {
        renderFeed(makeDemoFeed(), true);
        showToast("Источники недоступны — показана демонстрационная геометрия.");
        setStatus("Демо-данные · источники недоступны");
      }
    }
  } finally {
    button.disabled = false;
    button.innerHTML = '<span class="button-icon">↻</span> Загрузить данные';
  }
}

async function loadLocalGtfs(options = {}) {
  const button = $("#load-local-gtfs");
  if (!options.silent) {
    button.disabled = true;
    button.textContent = "Загружаю локальный GTFS…";
    setStatus("Чтение moscow-tram-gtfs.zip…");
  }
  try {
    const response = await fetch("./moscow-tram-gtfs.zip", { cache: "no-store" });
    if (!response.ok) throw new Error(`Локальный ZIP не найден: HTTP ${response.status}`);
    const feed = await parseGtfsZip(await response.arrayBuffer());
    if (!feed.routes.length) throw new Error("В локальном GTFS нет routes.txt");
    renderFeed(feed, false);
    setStatus(`Локальный GTFS · ${feed.routes.length} маршрутов`);
    if (!options.silent) showToast(`Загружено из ZIP: ${feed.routes.length} маршрутов`);
    return feed;
  } finally {
    if (!options.silent) {
      button.disabled = false;
      button.innerHTML = '<span class="button-icon">▣</span> Открыть локальный GTFS ZIP';
    }
  }
}

async function loadLocalGeoJson(options = {}) {
  const button = $("#load-local-geojson");
  if (!options.silent) {
    button.disabled = true;
    button.textContent = "Загружаю GeoJSON…";
    setStatus("Чтение data/tram_routes_3221.geojson…");
  }
  try {
    const response = await fetch("./data/tram_routes_3221.geojson", { cache: "no-store" });
    if (!response.ok) throw new Error(`Локальный GeoJSON не найден: HTTP ${response.status}`);
    const payload = await response.json();
    const records = (payload.features || []).map((feature) => ({
      ...(feature.properties?.attributes || feature.properties || feature.attributes || {}),
      geometry: feature.geometry,
    }));
    const feed = buildGtfsFeed(records);
    if (!feed.routes.length) throw new Error("В локальном GeoJSON нет трамвайных маршрутов");
    renderFeed(feed, false);
    setStatus(`Локальный GeoJSON · ${feed.routes.length} трамвайных маршрутов`);
    if (!options.silent) showToast(`Загружено из GeoJSON: ${feed.routes.length} маршрутов`);
    return feed;
  } finally {
    if (!options.silent) {
      button.disabled = false;
      button.innerHTML = '<span class="button-icon">⌁</span> Открыть актуальный GeoJSON';
    }
  }
}

async function parseGtfsZip(buffer) {
  if (!window.JSZip) throw new Error("JSZip не загружен");
  const zip = await JSZip.loadAsync(buffer);
  const read = async (name) => {
    const file = zip.file(name);
    return file ? parseCsv(await file.async("text")) : [];
  };
  const [agency, routes, stops, trips, stopTimes, calendar, shapes] = await Promise.all([
    read("agency.txt"), read("routes.txt"), read("stops.txt"), read("trips.txt"),
    read("stop_times.txt"), read("calendar.txt"), read("shapes.txt"),
  ]);
  return {
    agency,
    routes: routes.map((route, index) => ({
      ...route,
      route_type: Number(route.route_type || 0),
      route_color: routeColor(index),
    })),
    stops: stops.map((stop) => ({ ...stop, stop_lat: Number(stop.stop_lat), stop_lon: Number(stop.stop_lon) })),
    trips: trips.map((trip) => ({ ...trip, direction_id: Number(trip.direction_id || 0) })),
    stopTimes: stopTimes.map((item) => ({ ...item, stop_sequence: Number(item.stop_sequence || 0) })),
    calendar,
    shapes: shapes.map((shape) => ({
      ...shape,
      shape_pt_lat: Number(shape.shape_pt_lat),
      shape_pt_lon: Number(shape.shape_pt_lon),
      shape_pt_sequence: Number(shape.shape_pt_sequence || 0),
    })),
  };
}

function parseCsv(text) {
  const input = text.replace(/^\uFEFF/, "");
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let index = 0; index < input.length; index += 1) {
    const char = input[index];
    const next = input[index + 1];
    if (char === '"' && quoted && next === '"') {
      field += '"';
      index += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      row.push(field);
      field = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") index += 1;
      row.push(field);
      if (row.some((value) => value !== "")) rows.push(row);
      row = [];
      field = "";
    } else {
      field += char;
    }
  }
  if (field || row.length) {
    row.push(field);
    rows.push(row);
  }
  const headers = rows.shift()?.map((header) => header.trim()) || [];
  return rows.map((values) => headers.reduce((result, header, index) => {
    result[header] = values[index] ?? "";
    return result;
  }, {}));
}

async function fetchAllRows(endpoint, apiKey) {
  const rows = [];
  let skip = 0;
  let total = Infinity;
  const base = endpoint.replace(/[?&]$/, "");

  while (rows.length < total && skip < 20000) {
    const url = new URL(base);
    url.searchParams.set("$top", PAGE_SIZE);
    url.searchParams.set("$skip", skip);
    if (apiKey) url.searchParams.set("api_key", apiKey);
    const response = await fetchWithTimeout(url, { headers: { Accept: "application/json" } }, 12000);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const json = await response.json();
    const page = unwrapRows(json);
    if (!page.length) break;
    rows.push(...page);
    total = Number(json.Count ?? json.count ?? json.total ?? total);
    skip += page.length;
    if (page.length < PAGE_SIZE) break;
  }
  return rows;
}

async function fetchWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timeout);
  }
}

function unwrapRows(payload) {
  if (Array.isArray(payload)) return payload;
  if (Array.isArray(payload?.rows)) return payload.rows;
  if (Array.isArray(payload?.Rows)) return payload.Rows;
  if (Array.isArray(payload?.data)) return payload.data;
  if (Array.isArray(payload?.Data)) return payload.Data;
  if (Array.isArray(payload?.features)) {
    return payload.features.map((feature) => {
      const attributes = feature.attributes || feature.properties || {};
      return {
      ...(attributes.attributes || attributes),
      geometry: feature.geometry,
      };
    });
  }
  return [];
}

function buildGtfsFeed(records) {
  const normalized = records.map(normalizeRecord).filter((record) => record.routeId && record.points.length);
  const tramRecords = normalized.filter((record) => isTramRecord(record, normalized));
  const hasTransportType = normalized.some((record) => record.hasTransportType);
  const recordsToUse = hasTransportType ? tramRecords : (tramRecords.length ? tramRecords : normalized);
  const grouped = new Map();

  recordsToUse.forEach((record) => {
    const key = `${record.routeId}|${record.direction || "0"}`;
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(record);
  });

  const routes = [];
  const stops = [];
  const trips = [];
  const stopTimes = [];
  const shapes = [];
  let routeNumber = 0;

  for (const [groupKey, recordsInGroup] of grouped) {
    const first = recordsInGroup[0];
    const routeId = slug(first.routeId);
    const direction = first.direction || "0";
    const routeKey = `${routeId}|${direction}`;
    const ordered = recordsInGroup
      .sort((a, b) => (a.stopSequence ?? 9999) - (b.stopSequence ?? 9999))
      .flatMap((record) => record.points.map((point, index) => ({
        ...point,
        order: (record.stopSequence ?? 0) * 100 + index,
      })))
      .sort((a, b) => a.order - b.order);
    const deduped = dedupePoints(ordered);
    if (deduped.length < 2) continue;

    const id = `route_${routeNumber++}`;
    const tripId = `trip_${routeNumber}`;
    const shapeId = `shape_${routeNumber}`;
    routes.push({
      route_id: id,
      route_short_name: first.routeId,
      route_long_name: first.routeName || `Трамвай ${first.routeId}`,
      route_type: 0,
      route_color: routeColorFor(first.routeId, routeNumber),
      route_text_color: "FFFFFF",
      direction,
    });
    trips.push({ route_id: id, service_id: "MOSCOW", trip_id: tripId, trip_headsign: first.headsign || "", direction_id: direction === "0" ? 0 : 1, shape_id: shapeId });

    const namedStops = first.stopNames.length ? placeStopsAlongLine(deduped, first.stopNames) : deduped;
    namedStops.forEach((point, index) => {
      const stop = {
        stop_id: `stop_${id}_${index + 1}`,
        stop_code: point.id || "",
        stop_name: point.name || `Остановка ${index + 1}`,
        stop_lat: point.lat,
        stop_lon: point.lon,
      };
      stops.push(stop);
      const time = formatTime(5 * index);
      stopTimes.push({ trip_id: tripId, arrival_time: time, departure_time: time, stop_id: stop.stop_id, stop_sequence: index + 1 });
    });
    deduped.forEach((point, index) => shapes.push({ shape_id: shapeId, shape_pt_lat: point.lat, shape_pt_lon: point.lon, shape_pt_sequence: index + 1 }));
  }

  return makeFeed({ routes, stops, trips, stopTimes, shapes });
}

function normalizeRecord(raw) {
  const source = raw?.Cells || raw?.cells || raw?.attributes || raw?.properties || raw || {};
  const entries = Object.entries(source);
  const get = (patterns) => {
    const match = entries.find(([key]) => patterns.some((pattern) => pattern.test(normalizeKey(key))));
    return match?.[1];
  };
  const routeId = clean(get([/route.?short.?name/, /route.?number/, /route.?id/, /^route$/, /номер.*маршрут/, /маршрут/, /номер/, /линия/]));
  const routeName = clean(get([/route.?long.?name/, /route.?name/, /наименование/, /название/, /^name$/]));
  const direction = clean(get([/direction/, /направлен/])) || "0";
  const stopSequence = number(get([/stop.?sequence/, /sequence/, /порядок/, /номер.*останов/]));
  const stopName = clean(get([/stop.?name/, /название.*останов/, /остановка/]));
  const stopId = clean(get([/^stop.?id$/, /^id.*останов/, /код.*останов/]));
  const type = clean(get([/route.?type/, /transport.?type/, /type.?of.?transport/, /typeoftransport/, /type.?object/, /vehicle/, /транспорт/, /вид.*маршрут/, /тип.*транспорт/, /вид/]));
  const track = clean(get([/^trackoffollowing$/, /трассаследования/]));
  const searchText = entries.map(([, value]) => clean(value)).join(" ");
  const points = extractPoints({ ...source, geometry: raw?.geometry || source.geometry }, stopName, stopId);
  return {
    routeId, routeName, direction, stopSequence, stopName, stopId, type, searchText,
    stopNames: parseRouteStops(track),
    hasTransportType: Boolean(type), points, headsign: clean(get([/headsign/, /конечн/])),
  };
}

function parseRouteStops(forward) {
  const stops = [];
  tokenizeTrack(forward).forEach((token) => {
    const previous = stops[stops.length - 1];
    if (previous && isStopNameContinuation(previous, token)) {
      stops[stops.length - 1] = `${previous} - ${token}`;
      return;
    }
    if (previous === token) return;
    stops.push(token);
  });
  return stops;
}

function isStopNameContinuation(previous, token) {
  if (/^[а-яё]/.test(token)) return true;
  if (previous === "Покровское" && /^(Глебово|Стрешнево)$/.test(token)) return true;
  return previous === "Свято" && token.startsWith("Данилов");
}

function tokenizeTrack(value) {
  return clean(value).split(/\s+-\s+/).map((part) => part.trim()).filter(Boolean);
}

function placeStopsAlongLine(points, names) {
  if (names.length === 1) return [{ ...points[0], name: names[0], id: "" }];
  const distances = [0];
  for (let index = 1; index < points.length; index += 1) {
    distances.push(distances[index - 1] + pointDistance(points[index - 1], points[index]));
  }
  const total = distances[distances.length - 1] || 0;
  return names.map((name, nameIndex) => {
    const target = total * (nameIndex / (names.length - 1));
    const segment = Math.max(1, distances.findIndex((distance) => distance >= target));
    const previous = distances[segment - 1];
    const span = distances[segment] - previous || 1;
    const ratio = Math.min(1, Math.max(0, (target - previous) / span));
    const start = points[segment - 1];
    const finish = points[Math.min(segment, points.length - 1)];
    return {
      lat: start.lat + (finish.lat - start.lat) * ratio,
      lon: start.lon + (finish.lon - start.lon) * ratio,
      name,
      id: "",
    };
  });
}

function pointDistance(a, b) {
  return haversine(
    { stop_lat: a.lat, stop_lon: a.lon },
    { stop_lat: b.lat, stop_lon: b.lon },
  );
}

function isTramRecord(record, allRecords) {
  if (record.hasTransportType) {
    return /трам|tram/i.test(record.type || "") || /^0$/.test(record.type || "");
  }
  if (/трам|tram/i.test(record.searchText || "")) return true;
  return allRecords.every((item) => !item.hasTransportType);
}

function extractPoints(source, fallbackName, fallbackId) {
  const points = [];
  const geometryEntry = Object.entries(source).find(([key]) => /geo|coord|гео|координат/i.test(key));
  const geometry = source.geometry || source.geoData || source.geo_data || source.coordinates || geometryEntry?.[1];
  if (geometry) points.push(...parseGeometry(geometry, fallbackName, fallbackId));
  if (points.length) return points;

  const entries = Object.entries(source);
  const lat = findNumber(entries, [/^lat/, /широт/, /latitude/]);
  const lon = findNumber(entries, [/^lon/, /долгот/, /longitude/]);
  if (lat !== null && lon !== null && inMoscow(lat, lon)) {
    return [{ lat, lon, name: fallbackName, id: fallbackId }];
  }
  return [];
}

function parseGeometry(value, name, id) {
  if (typeof value === "object" && value.coordinates) return parseCoordinates(value.coordinates, name, id);
  if (typeof value === "object" && value.geometry) return parseGeometry(value.geometry, name, id);
  if (typeof value !== "string") return [];
  try {
    const parsed = JSON.parse(value);
    return parseGeometry(parsed, name, id);
  } catch {
    const numbers = value.match(/-?\d+(?:\.\d+)?/g)?.map(Number) || [];
    if (numbers.length >= 2) {
      const [a, b] = numbers;
      const lat = Math.abs(a) > 50 ? a : b;
      const lon = Math.abs(a) > 50 ? b : a;
      return inMoscow(lat, lon) ? [{ lat, lon, name, id }] : [];
    }
  }
  return [];
}

function parseCoordinates(coordinates, name, id) {
  if (!Array.isArray(coordinates)) return [];
  if (typeof coordinates[0] === "number") {
    const [lon, lat] = coordinates;
    return inMoscow(lat, lon) ? [{ lat, lon, name, id }] : [];
  }
  return coordinates.flatMap((item) => parseCoordinates(item, name, id));
}

function findNumber(entries, patterns) {
  const pair = entries.find(([key]) => patterns.some((pattern) => pattern.test(normalizeKey(key))));
  const value = Number(String(pair?.[1] ?? "").replace(",", "."));
  return Number.isFinite(value) ? value : null;
}

function makeFeed(data) {
  return {
    agency: [{ agency_name: "Мосгортранс", agency_url: "https://mosgortrans.ru", agency_timezone: "Europe/Moscow", agency_lang: "ru" }],
    routes: data.routes,
    stops: data.stops,
    trips: data.trips,
    stopTimes: data.stopTimes,
    shapes: data.shapes,
    calendar: [{ service_id: "MOSCOW", monday: 1, tuesday: 1, wednesday: 1, thursday: 1, friday: 1, saturday: 1, sunday: 1, start_date: "20260101", end_date: "20261231" }],
  };
}

function renderFeed(feed, usingDemo) {
  clearStationCards();
  state.feed = feed;
  state.routes = feed.routes;
  state.usingDemo = usingDemo;
  state.selectedId = null;
  $("#route-details").classList.add("hidden");
  renderMap();
  renderRouteList();
  updateStats();
  fitMap();
  selectFromHash();
  applySavedMapState();
  if (!usingDemo) frameInitialMap();
  else if (!state.mapFramed) frameMap();
}

function renderMap() {
  window.clearTimeout(state.hoverTimer);
  state.hoveredId = null;
  state.layers.forEach((layer) => layer.remove());
  state.layers.clear();
  const shapeGroups = groupBy(state.feed.shapes, "shape_id");
  const routes = [...state.routes].sort((left, right) => Number(left.route_id === state.selectedId) - Number(right.route_id === state.selectedId));
    routes.forEach((route) => {
      const trip = state.feed.trips.find((item) => item.route_id === route.route_id);
      const points = (shapeGroups[trip?.shape_id] || []).sort((a, b) => a.shape_pt_sequence - b.shape_pt_sequence).map((point) => [point.shape_pt_lat, point.shape_pt_lon]);
      if (points.length < 2) return;
      
      const selected = route.route_id === state.selectedId;
      const dimmed = state.selectedId && !selected;
      
      // ЛОГИКА РАСКРАСКИ ПО ЗАГРУЖЕННОСТИ
      let color = `#${route.route_color || "e33d4d"}`;
      if (state.showMapLoad && window.TramForecast?.state.loaded) {
        const view = {
          route: route.route_short_name,
          horizon: "day",
          date: state.forecast.date,
          hourFrom: state.forecast.selectedHour,
          hourTo: state.forecast.selectedHour,
          scenario: state.forecast.scenario
        };
        const board = TramForecast.present(view);
        const row = board.rows[0];
        if (row && row.scenarioValue != null) {
          const tone = TramForecast.tone(row.scenarioValue, row.usual);
          color = tone === "hot" ? "#e23b4b" : tone === "warm" ? "#e2a322" : "#2faf67";
        }
      }

      const weights = routeWeights(selected, false);
      const casing = L.polyline(points, { color: "#ffffff", weight: weights.casing, opacity: dimmed ? .1 : .9, lineCap: "round", lineJoin: "round", interactive: false });
      const line = L.polyline(points, { color, weight: weights.line, opacity: dimmed ? .1 : 1, lineCap: "round", lineJoin: "round", interactive: false });
      // ... остальной код создания hit и group без изменений ...
      const hit = L.polyline(points, { color, weight: weights.hit, opacity: 0, lineCap: "round", lineJoin: "round", interactive: true })
        .on("mouseover", () => holdRoute(route.route_id, route)) // <-- Передаем route
        .on("mouseout", () => releaseRoute(route.route_id))
        .on("click", () => selectRoute(route.route_id));
    const group = L.layerGroup([casing, line, hit]).addTo(state.map);
    group._casing = casing;
    group._line = line;
    group._hit = hit;
    state.layers.set(route.route_id, group);
  });
  renderStops();
  state.layers.forEach((group) => group._hit.bringToFront());
  raiseStops();
}

function holdRoute(routeId, routeObj) {
  window.clearTimeout(state.hoverTimer);
  if (state.hoveredId && state.hoveredId !== routeId) paintRouteHover(state.hoveredId, false);
  state.hoveredId = routeId;
  paintRouteHover(routeId, true);
  
  // Формируем текст подсказки с загруженностью
  let tooltipText = `${routeObj.route_short_name} · ${routeObj.route_long_name}`;
  if (window.TramForecast?.state.loaded) {
    const view = {
      route: routeObj.route_short_name,
      horizon: "day",
      date: state.forecast.date,
      hourFrom: state.forecast.selectedHour,
      hourTo: state.forecast.selectedHour,
      scenario: state.forecast.scenario
    };
    const board = TramForecast.present(view);
    const row = board.rows[0];
    if (row && row.scenarioValue != null) {
      const loadText = TramForecast.formatCount(row.scenarioValue);
      const usualText = row.usual != null ? ` (обычно ${TramForecast.formatCount(row.usual)})` : "";
      tooltipText += `<br><strong>Посадки:</strong> ${loadText}${usualText}`;
    }
  }
  
  state.layers.get(routeId)?._hit.setTooltipContent(tooltipText).openTooltip();
}

function releaseRoute(routeId) {
  window.clearTimeout(state.hoverTimer);
  state.hoverTimer = window.setTimeout(() => {
    if (state.hoveredId !== routeId) return;
    paintRouteHover(routeId, false);
    state.layers.get(routeId)?._hit.closeTooltip();
    state.hoveredId = null;
  }, LINE_HOVER_LINGER_MS);
}

function paintRouteHover(routeId, active) {
  const group = state.layers.get(routeId);
  if (!group) return;
  const selected = routeId === state.selectedId;
  const dimmed = Boolean(state.selectedId) && !selected;
  group._line.setStyle({
    weight: routeWeights(selected, active).line,
    opacity: active || !dimmed ? 1 : .1,
  });
  group._casing.setStyle({
    weight: routeWeights(selected, active).casing,
    opacity: active ? 1 : (dimmed ? .1 : .9),
  });
  if (active) {
    group._casing.bringToFront();
    group._line.bringToFront();
    group._hit.bringToFront();
  }
  raiseStops();
}

function raiseStops() {
  state.stopsLayer?.eachLayer((layer) => layer.bringToFront());
}

function routeWeights(selected, hovered) {
  const scale = state.lineScale;
  return {
    line: ((selected ? 8 : 6) + (hovered ? 3 : 0)) * scale,
    casing: ((selected ? 14 : 11) + (hovered ? 4 : 0)) * scale,
    hit: Math.max(16, LINE_HIT_WEIGHT * scale),
  };
}

function applyLineScale() {
  state.layers.forEach((group, routeId) => {
    const selected = routeId === state.selectedId;
    const hovered = routeId === state.hoveredId;
    const weights = routeWeights(selected, hovered);
    group._line.setStyle({ weight: weights.line });
    group._casing.setStyle({ weight: weights.casing });
    group._hit.setStyle({ weight: weights.hit });
  });
  raiseStops();
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function bindEdgeResize(handle, options) {
  let drag = null;
  handle.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    drag = { x: event.clientX, width: options.read() };
    options.target.classList.add("is-resizing");
    document.body.classList.add("is-resizing");
    handle.setPointerCapture(event.pointerId);
  });
  handle.addEventListener("pointermove", (event) => {
    if (!drag) return;
    options.write(Math.round(options.next(drag.width, event.clientX - drag.x)));
  });
  const end = () => {
    if (!drag) return;
    drag = null;
    options.target.classList.remove("is-resizing");
    document.body.classList.remove("is-resizing");
  };
  handle.addEventListener("pointerup", end);
  handle.addEventListener("pointercancel", end);
}

const LOAD_FILL = { ok: "#2faf67", warm: "#e2a322", hot: "#e23b4b" };

function stationLoad(route, index, total) {
  const base = TramDemand.demandFor(route.route_short_name).index;
  const position = total <= 1 ? 0.5 : index / (total - 1);
  const wave = Math.sin(position * Math.PI);
  return base * (0.62 + 0.72 * wave);
}

function renderStops() {
  state.stopsLayer.clearLayers();
  if (!state.showStops || !state.feed) return;
  const selectedStops = state.selectedId ? stopsForRoute(state.selectedId) : [];
  const selectedIndex = new Map(selectedStops.map((stop, index) => [stop.stop_id, index]));
  const stopColors = buildStopColorIndex();
  const selectedRoute = state.routes.find((route) => route.route_id === state.selectedId);
  const stops = [...state.feed.stops].sort((left, right) => {
    const rank = (stop) => (stop.stop_id === state.stationStopId ? 3 : 0) + (state.stationCards.has(stop.stop_id) ? 2 : 0) + (selectedIndex.has(stop.stop_id) ? 1 : 0);
    return rank(left) - rank(right);
  });
  stops.forEach((stop) => {
    const onSelectedLine = selectedIndex.has(stop.stop_id);
    const colors = stopColors.get(stop.stop_id) || [];
    let fill = `#${onSelectedLine && selectedRoute ? selectedRoute.route_color : (colors[0] || "e33d4d")}`;
    let tooltip = colors.length > 1 ? `${stop.stop_name} · ${colors.length} линии` : stop.stop_name;
    if (onSelectedLine && selectedRoute) {
      const load = stationLoad(selectedRoute, selectedIndex.get(stop.stop_id), selectedStops.length);
      const tone = TramDemand.loadTone(load);
      fill = LOAD_FILL[tone];
    }
    const open = state.stationCards.has(stop.stop_id);
    const focused = stop.stop_id === state.stationStopId;
    const visible = !state.selectedId || onSelectedLine || open;
    L.circleMarker([stop.stop_lat, stop.stop_lon], {
      radius: focused ? 10 : (open ? 8 : (onSelectedLine ? 5.5 : 3.5)),
      weight: focused ? 2.5 : (open ? 2 : 1.5),
      color: "#fff",
      opacity: visible ? 1 : .1,
      fillColor: fill,
      fillOpacity: visible ? 1 : .1,
    }).bindTooltip(tooltip, { direction: "top", offset: [0, -4], className: "tram-tooltip" }).addTo(state.stopsLayer);
  });
}

function buildStopColorIndex() {
  const colors = new Map();
  const trips = new Map(state.feed.trips.map((trip) => [trip.trip_id, trip]));
  const routes = new Map(state.feed.routes.map((route) => [route.route_id, route]));
  state.feed.stopTimes.forEach((stopTime) => {
    const route = routes.get(trips.get(stopTime.trip_id)?.route_id);
    if (!route?.route_color) return;
    if (!colors.has(stopTime.stop_id)) colors.set(stopTime.stop_id, []);
    const stopColors = colors.get(stopTime.stop_id);
    if (!stopColors.includes(route.route_color)) stopColors.push(route.route_color);
  });
  return colors;
}

function renderRouteList(query = "") {
  const normalizedQuery = query.trim().toLowerCase();
  const list = $("#route-list");
  const routes = state.routes.filter((route) => {
    if (!normalizedQuery) return true;
    return `${route.route_short_name} ${route.route_long_name} ${stopsForRoute(route.route_id).map((stop) => stop.stop_name).join(" ")}`.toLowerCase().includes(normalizedQuery);
  });
  list.innerHTML = routes.length ? routes.map((route) => {
    const stops = stopsForRoute(route.route_id);
    return `<button class="route-item ${route.route_id === state.selectedId ? "selected" : ""}" data-route-id="${route.route_id}">
      <span class="route-badge" style="background:#${route.route_color}">${escapeHtml(route.route_short_name)}</span>
      <span class="route-item-text"><span class="route-item-name">${escapeHtml(route.route_long_name)}</span><span class="route-item-meta">${stops.length} остановок · ${route.direction === "1" ? "обратное" : "прямое"}</span></span>
      ${boardingPill(route.route_short_name)}
    </button>`;
  }).join("") : '<div class="empty-state">Ничего не найдено.<br />Попробуйте номер маршрута или название остановки.</div>';
  list.querySelectorAll("[data-route-id]").forEach((item) => item.addEventListener("click", () => selectRoute(item.dataset.routeId)));
}

function selectRoute(routeId) {
  const route = state.routes.find((item) => item.route_id === routeId);
  if (!route) return;
  state.selectedId = routeId;
  renderMap();
  renderRouteList($("#route-search").value);
  const stops = stopsForRoute(routeId);
  $("#detail-number").textContent = route.route_short_name;
  $("#detail-number").style.backgroundColor = `#${route.route_color}`;
  $("#detail-name").textContent = route.route_long_name;
  $("#detail-stops").textContent = stops.length;
  $("#detail-length").textContent = `${routeDistance(stops).toFixed(1)} км`;
  renderDemand(route);
  renderStations(stops);
  $("#route-details").classList.remove("hidden");
}

function renderStations(stops) {
  $("#stations-list").innerHTML = stops.map((stop, index) => `<li><button type="button" class="station-item ${state.stationCards.has(stop.stop_id) ? "selected" : ""}" data-stop-id="${escapeHtml(stop.stop_id)}"><span>${index + 1}</span>${escapeHtml(stop.stop_name)}</button></li>`).join("");
  $("#stations-list").querySelectorAll(".station-item").forEach((button) => {
    button.addEventListener("click", () => openStationCard(button.dataset.stopId));
  });
}

function openStationCard(stopId, placement) {
  const route = state.routes.find((item) => item.route_id === state.selectedId);
  const stop = stopsForRoute(state.selectedId).find((item) => item.stop_id === stopId);
  if (!stop || !route) return;
  if (state.stationCards.has(stopId)) {
    focusStationCard(stopId);
    return;
  }
  const station = window.TramFacts && TramFacts.state.loaded
    ? TramFacts.stationFor(stop.stop_name, route.route_short_name, stop.stop_lat, stop.stop_lon)
    : null;
  const card = document.getElementById("station-card-template").content.firstElementChild.cloneNode(true);
  card.dataset.stopId = stopId;
  card.querySelector(".station-card-name").textContent = stop.stop_name;
  card.querySelector(".station-card-name").title = stop.stop_name;
  card.setAttribute("aria-label", stop.stop_name);
  document.body.appendChild(card);
  fillStationCard(card, station, stop);
  state.stationCards.set(stopId, card);
  if (placement?.left && placement?.top) {
    card.style.left = placement.left;
    card.style.top = placement.top;
    if (placement.width) card.style.width = placement.width;
    if (placement.height) card.style.height = placement.height;
  } else {
    placeStationCard(card, state.stationCards.size - 1);
  }
  bindStationCard(card, stopId);
  focusStationCard(stopId);
}

function fillStationCard(card, station, stop) {
  const weatherValue = card.querySelector(".station-weather-value");
  const trafficValue = card.querySelector(".station-traffic-value");
  const pois = card.querySelector(".station-pois");
  weatherValue.textContent = station?.temperature == null ? "—" : `${station.temperature > 0 ? "+" : ""}${station.temperature}°`;
  trafficValue.textContent = station ? String(station.roads || 0) : "—";
  const nearby = window.TramFacts && stop ? TramFacts.poisNear(stop.stop_lat, stop.stop_lon, 5, 500) : [];
  pois.innerHTML = nearby.length
    ? nearby.map((place) => `<li>${escapeHtml(place.name)}${place.label ? ` · ${escapeHtml(place.label)}` : ""}</li>`).join("")
    : "<li>—</li>";
}

function placeStationCard(card, index) {
  const width = card.offsetWidth || 320;
  const height = card.offsetHeight || 280;
  const shift = (index % 6) * 32;
  card.style.left = `${Math.max(8, Math.round((window.innerWidth - width) / 2 + shift - 64))}px`;
  card.style.top = `${Math.max(8, Math.round((window.innerHeight - height) / 3 + shift))}px`;
}

function focusStationCard(stopId) {
  const card = state.stationCards.get(stopId);
  if (!card) return;
  const changed = state.stationStopId !== stopId;
  state.stationCards.delete(stopId);
  state.stationCards.set(stopId, card);
  state.stationStopId = stopId;
  state.stationCardZ += 1;
  card.style.zIndex = String(state.stationCardZ);
  document.querySelectorAll(".station-card").forEach((item) => item.classList.toggle("is-active", item.dataset.stopId === stopId));
  document.querySelectorAll(".station-item").forEach((button) => button.classList.toggle("selected", state.stationCards.has(button.dataset.stopId)));
  syncForecastStopNote();
  if (!changed) return;
  renderStops();
  raiseStops();
}

function clearStationCards() {
  state.stationCards.forEach((card) => card.remove());
  state.stationCards.clear();
  state.stationStopId = null;
}

function closeStationCard(stopId) {
  const card = state.stationCards.get(stopId);
  if (!card) return;
  card.remove();
  state.stationCards.delete(stopId);
  if (state.stationStopId === stopId) {
    const remaining = [...state.stationCards.keys()];
    state.stationStopId = remaining.length ? remaining[remaining.length - 1] : null;
  }
  document.querySelectorAll(".station-card").forEach((item) => item.classList.toggle("is-active", item.dataset.stopId === state.stationStopId));
  document.querySelectorAll(".station-item").forEach((button) => button.classList.toggle("selected", state.stationCards.has(button.dataset.stopId)));
  syncForecastStopNote();
  renderStops();
  raiseStops();
}

function bindStationCard(card, stopId) {
  const handle = card.querySelector(".station-card-head");
  let drag = null;
  card.addEventListener("pointerdown", () => focusStationCard(stopId));
  handle.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || event.target.closest("button")) return;
    const rect = card.getBoundingClientRect();
    drag = { dx: event.clientX - rect.left, dy: event.clientY - rect.top };
    handle.setPointerCapture(event.pointerId);
  });
  handle.addEventListener("pointermove", (event) => {
    if (!drag) return;
    const width = card.offsetWidth;
    const left = Math.min(window.innerWidth - 48, Math.max(48 - width, event.clientX - drag.dx));
    const top = Math.min(window.innerHeight - 36, Math.max(0, event.clientY - drag.dy));
    card.style.left = `${left}px`;
    card.style.top = `${top}px`;
  });
  const endDrag = () => { drag = null; };
  handle.addEventListener("pointerup", endDrag);
  handle.addEventListener("pointercancel", endDrag);
  card.querySelector(".station-card-close").addEventListener("click", () => closeStationCard(stopId));
  bindStationResize(card);
}

function setStationsOpen(open) {
  const panel = $("#stations-panel");
  panel.classList.toggle("open", open);
  $("#stations-toggle").setAttribute("aria-expanded", open ? "true" : "false");
}

function bindStationResize(card) {
  const handle = card.querySelector(".station-card-resize");
  let resizing = null;
  handle.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    focusStationCard(card.dataset.stopId);
    const rect = card.getBoundingClientRect();
    resizing = { x: event.clientX, y: event.clientY, width: rect.width, height: rect.height };
    handle.setPointerCapture(event.pointerId);
  });
  handle.addEventListener("pointermove", (event) => {
    if (!resizing) return;
    const width = Math.min(window.innerWidth - 16, Math.max(240, resizing.width + event.clientX - resizing.x));
    const height = Math.min(window.innerHeight - 16, Math.max(180, resizing.height + event.clientY - resizing.y));
    card.style.width = `${Math.round(width)}px`;
    card.style.height = `${Math.round(height)}px`;
  });
  const endResize = () => { resizing = null; };
  handle.addEventListener("pointerup", endResize);
  handle.addEventListener("pointercancel", endResize);
}

function renderDemand(route) {
  const vehicles = window.TramFacts && TramFacts.state.loaded ? TramFacts.fleetFor(route.route_short_name) : null;
  $("#fleet-count").textContent = vehicles == null ? "—" : String(vehicles);
  const factors = (window.TramFacts && TramFacts.state.loaded && TramFacts.contextFor(route.route_short_name)) || TramDemand.contextFor(route.route_short_name);
  applyRoutePois(factors, route);
  $("#demand-context").innerHTML = factors.map((item) => `<article class="factor-card ${item.off ? "off" : ""}"><span class="factor-icon">${item.icon}</span><span class="factor-label">${escapeHtml(item.label)}</span><strong>${escapeHtml(item.value)}</strong></article>`).join("");
  syncForecastControls();
  if (!window.TramForecast || (!TramForecast.state.loaded && !TramForecast.state.failed)) {
    $("#demand-now").textContent = "…";
    return;
  }
  if (TramForecast.state.failed) {
    renderMockDemand(route);
    return;
  }
  const view = forecastView(route);
  const board = TramForecast.present(view);
  const shownTotal = board.totalScenario;
  $("#forecast-total-label").textContent = view.horizon === "day" ? "Посадки за выбранные часы" : view.horizon === "month" ? "Посадки за месяц" : "Посадки за год";
  $("#demand-now").textContent = TramForecast.formatCount(shownTotal);
  const load = $("#demand-load");
  const forecastTone = board.rows.some((row) => row.status === "forecast" || row.status === "partial-forecast");
  load.textContent = statusWithMultiplier(board);
  load.className = `load-pill ${board.totalScenario == null ? "missing" : board.scenario.apply || forecastTone ? "warm" : "ok"}`;
  $("#forecast-caption").textContent = view.horizon === "day" ? "Посадки по часам" : view.horizon === "month" ? "Посадки по дням" : "Посадки по месяцам";
  $("#demand-today").innerHTML = forecastChart(view, board.rows);
  $("#demand-today").querySelectorAll("[data-period]").forEach((button) => {
    button.addEventListener("click", () => selectForecastPeriod(button.dataset.period, button.dataset.hour));
  });
  $("#forecast-table").innerHTML = forecastTable(board);
  $("#forecast-table").querySelectorAll("[data-period]").forEach((button) => {
    button.addEventListener("click", () => selectForecastPeriod(button.dataset.period, button.dataset.hour));
  });
  $("#forecast-risk").innerHTML = riskText(view, board.rows);
  syncHourChip();
}

function renderMockDemand(route) {
  const demand = TramDemand.demandFor(route.route_short_name);
  $("#demand-now").textContent = TramDemand.formatPassengers(demand.current.passengers);
  $("#demand-load").textContent = "макет";
  $("#demand-load").className = "load-pill missing";
  $("#demand-today").innerHTML = hourColumns(demand.upcoming);
  $("#forecast-table").innerHTML = "";
  $("#forecast-risk").innerHTML = "";
}

function isOpenForecastDate(value) {
  return /^\d{4}-\d{2}-\d{2}$/.test(value || "") && value >= "2025-01-01";
}

function statusWithMultiplier(board) {
  if (!board.scenario.apply || !Number.isFinite(board.scenario.coefficient)) return board.statusLabel;
  const coeff = board.scenario.coefficient.toLocaleString("ru-RU", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  return `${board.statusLabel} × ${coeff}`;
}

function forecastView(route) {
  return {
    route: route.route_short_name,
    horizon: state.forecast.horizon,
    date: state.forecast.date,
    hourFrom: state.forecast.hourFrom,
    hourTo: state.forecast.hourTo,
    scenario: {
      weather: state.forecast.scenario.weather,
      eventOn: String(state.forecast.scenario.eventCoeff || "").trim() !== "",
      eventCoeff: state.forecast.scenario.eventCoeff,
      seasonMode: state.forecast.scenario.seasonMode,
      seasonCoeff: state.forecast.scenario.seasonCoeff,
    },
  };
}

function boardingPill(routeName) {
  if (!window.TramForecast || (!TramForecast.state.loaded && !TramForecast.state.failed)) {
    return '<span class="load-pill missing">…</span>';
  }
  if (TramForecast.state.failed) {
    const demand = TramDemand.demandFor(routeName);
    return `<span class="load-pill missing">${TramDemand.formatLoad(demand.index)}</span>`;
  }
  const board = TramForecast.present({
    route: routeName,
    horizon: "day",
    date: state.forecast.date,
    hourFrom: state.forecast.selectedHour,
    hourTo: state.forecast.selectedHour,
    scenario: {
      weather: state.forecast.scenario.weather,
      eventOn: String(state.forecast.scenario.eventCoeff || "").trim() !== "",
      eventCoeff: state.forecast.scenario.eventCoeff,
      seasonMode: state.forecast.scenario.seasonMode,
      seasonCoeff: state.forecast.scenario.seasonCoeff,
    },
  });
  const row = board.rows[0];
  if (!row || row.scenarioValue == null) return '<span class="load-pill missing">—</span>';
  const tone = TramForecast.tone(row.scenarioValue, row.usual);
  return `<span class="load-pill ${tone}">${TramForecast.formatCount(row.scenarioValue)}</span>`;
}

function syncForecastControls() {
  document.querySelectorAll("[data-horizon]").forEach((button) => button.classList.toggle("active", button.dataset.horizon === state.forecast.horizon));
  document.querySelectorAll("[data-trips]").forEach((button) => button.classList.toggle("active", Number(button.dataset.trips) === state.forecast.trips));
  setControlValue("#forecast-date", state.forecast.date);
  setControlValue("#hour-from", String(state.forecast.hourFrom));
  setControlValue("#hour-to", String(state.forecast.hourTo));
  setControlValue("#scenario-weather", state.forecast.scenario.weather);
  setControlValue("#scenario-season", state.forecast.scenario.seasonMode);
  setControlValue("#scenario-event-coeff", state.forecast.scenario.eventCoeff);
  setControlValue("#scenario-season-coeff", state.forecast.scenario.seasonCoeff);
  $("#scenario-season-field").classList.toggle("hidden", state.forecast.scenario.seasonMode !== "custom");
}

function setControlValue(selector, value) {
  const element = $(selector);
  if (!element || document.activeElement === element) return;
  element.value = value;
}

function syncHourChip() {
  const hours = window.TramForecast ? TramForecast.hourRange(state.forecast.hourFrom, state.forecast.hourTo) : [state.forecast.selectedHour];
  if (!hours.includes(state.forecast.selectedHour)) state.forecast.selectedHour = hours[0];
  const slider = $("#hour-slider");
  slider.min = String(hours[0]);
  slider.max = String(hours[hours.length - 1]);
  slider.value = String(state.forecast.selectedHour);
  $("#hour-readout").textContent = `${String(state.forecast.selectedHour).padStart(2, "0")}:00`;
  $("#hour-play").textContent = state.forecast.playing ? "❚❚" : "▶";
  $("#hour-play").setAttribute("aria-pressed", state.forecast.playing ? "true" : "false");
}

function syncForecastStopNote() {
  const note = $("#forecast-stop-note");
  if (!note) return;
  const card = state.stationStopId ? state.stationCards.get(state.stationStopId) : null;
  const name = card?.querySelector(".station-card-name")?.textContent;
  note.textContent = name ? `Остановка «${name}»: прогноз доступен только для маршрута.` : "";
}

function forecastChart(view, rows) {
  const max = Math.max(...rows.map((row) => row.scenarioValue || 0), 1);
  return rows.map((row) => {
    const selected = view.horizon === "day" ? row.hour === state.forecast.selectedHour : row.period === state.forecast.date || row.period === state.forecast.date.slice(0, 7);
    const label = view.horizon === "day" ? String(row.hour).padStart(2, "0") : row.period.slice(-2);
    const height = row.scenarioValue == null ? 8 : Math.max(8, (row.scenarioValue / max) * 100);
    const title = `${row.period} · база ${TramForecast.formatCount(row.baseValue)} · сценарий ${TramForecast.formatCount(row.scenarioValue)} · обычный ${TramForecast.formatCount(row.usual)} · ${row.statusLabel}`;
    return `<button type="button" class="hour-col ${selected ? "is-selected" : ""}" data-period="${escapeHtml(row.period)}" data-hour="${row.hour ?? ""}" title="${escapeHtml(title)}"><div class="hour-bar ${row.tone}" style="height:${height}%"></div><span>${escapeHtml(label)}</span></button>`;
  }).join("");
}

function forecastTable(board) {
  const rows = board.rows;
  const body = rows.map((row) => `<tr class="${row.hour === state.forecast.selectedHour || row.period === state.forecast.date ? "is-selected" : ""}"><td><button type="button" data-period="${escapeHtml(row.period)}" data-hour="${row.hour ?? ""}">${escapeHtml(row.period)}</button></td><td>${TramForecast.formatCount(row.baseValue)}</td><td>${TramForecast.formatCount(row.scenarioValue)}</td><td>${TramForecast.formatCount(row.usual)}</td><td>${escapeHtml(row.statusLabel)}</td></tr>`).join("");
  return `<thead><tr><th>Период</th><th>База</th><th>Сценарий</th><th>Обычный</th><th>Статус</th></tr></thead><tbody>${body}</tbody><tfoot><tr><th>Итого</th><td>${TramForecast.formatCount(board.totalBase)}</td><td>${TramForecast.formatCount(board.totalScenario)}</td><td></td><td>${escapeHtml(board.statusLabel)}</td></tr></tfoot>`;
}

function riskText(view, rows) {
  if (view.horizon !== "day") return "";
  const risks = rows.filter((row) => row.baseValue != null && row.usual != null && (row.tone === "hot" || row.tone === "warm"));
  return risks.map((row) => `<article class="risk-row"><strong>${escapeHtml(row.period)}</strong> ${TramForecast.formatCount(row.baseValue)} / ${TramForecast.formatCount(row.usual)} <button type="button" class="text-button" data-period="${escapeHtml(row.period)}" data-hour="${row.hour}">Проверить выпуск и расписание</button></article>`).join("");
}

function selectForecastPeriod(period, hour) {
  if (hour !== "" && hour != null) state.forecast.selectedHour = Number(hour);
  else if (/^\d{4}-\d{2}-\d{2}$/.test(period)) state.forecast.date = period;
  else if (/^\d{4}-\d{2}$/.test(period)) state.forecast.date = `${period}-01`;
  refreshForecast();
}

function syncForecastTimeDisplay() {
  const dateDisplay = $("#display-date");
  const hourDisplay = $("#display-hour");
  if (dateDisplay && state.forecast.date) {
    const [y, m, d] = state.forecast.date.split("-");
    dateDisplay.textContent = `${d}.${m}.${y}`;
  }
  if (hourDisplay) {
    hourDisplay.textContent = `${String(state.forecast.selectedHour).padStart(2, "0")}:00`;
  }
}

function refreshForecast() {
  syncForecastTimeDisplay(); // <-- Заменяем ручной код на вызов функции

  renderRouteList($("#route-search").value);
  const route = state.routes.find((item) => item.route_id === state.selectedId);
  if (route) renderDemand(route);
  else syncHourChip();
  
  if (state.showMapLoad) renderMap();
}

function bindForecastControls() {
  const hours = Array.from({ length: 24 }, (_, hour) => `<option value="${hour}">${String(hour).padStart(2, "0")}:00</option>`).join("");
  $("#hour-from").innerHTML = hours;
  $("#hour-to").innerHTML = hours;
  $("#hour-from").value = "0";
  $("#hour-to").value = "23";
  document.querySelectorAll("[data-horizon]").forEach((button) => {
    button.addEventListener("click", () => {
      state.forecast.horizon = button.dataset.horizon;
      refreshForecast();
    });
  });
  $("#forecast-date").addEventListener("change", (event) => {
    const value = event.target.value;
    if (!isOpenForecastDate(value)) {
      event.target.value = state.forecast.date;
      return;
    }
    state.forecast.date = value;
    refreshForecast();
  });
  $("#hour-from").addEventListener("change", (event) => {
    state.forecast.hourFrom = Number(event.target.value);
    refreshForecast();
  });
  $("#hour-to").addEventListener("change", (event) => {
    state.forecast.hourTo = Number(event.target.value);
    refreshForecast();
  });
  $("#hour-slider").addEventListener("input", (event) => {
    state.forecast.selectedHour = Number(event.target.value);
    refreshForecast();
  });
  $("#hour-play").addEventListener("click", toggleHourPlay);
  $("#scenario-weather").addEventListener("change", (event) => {
    state.forecast.scenario.weather = event.target.value;
    refreshForecast();
  });
  $("#scenario-season").addEventListener("change", (event) => {
    state.forecast.scenario.seasonMode = event.target.value;
    refreshForecast();
  });
  $("#scenario-event-coeff").addEventListener("input", (event) => {
    state.forecast.scenario.eventCoeff = event.target.value;
    state.forecast.scenario.eventOn = event.target.value.trim() !== "";
    refreshForecast();
  });
  $("#scenario-season-coeff").addEventListener("input", (event) => {
    state.forecast.scenario.seasonCoeff = event.target.value;
    refreshForecast();
  });
  
  $("#scenario-reset").addEventListener("click", () => {
    state.forecast.scenario = { weather: "base", eventOn: false, eventPlace: "", eventTime: "", eventCoeff: "", seasonMode: "base", seasonCoeff: "" };
    refreshForecast();
  });
  // ... (оставьте обработчики scenario-reset и т.д. как были) ...

  // 1. Сворачивание панели (НОВАЯ ЛОГИКА со стрелкой)
  const sidebar = $(".sidebar");
  const toggleBtn = $("#sidebar-collapse-btn");
  if (toggleBtn && sidebar) {
    toggleBtn.addEventListener("click", () => {
      const isCollapsed = sidebar.classList.toggle("collapsed");
      toggleBtn.textContent = isCollapsed ? "▶" : "◀";
      toggleBtn.title = isCollapsed ? "Развернуть панель" : "Свернуть панель";
      setTimeout(() => state.map.invalidateSize({ pan: false }), 320);
    });
  }

  // 2. Переключатель остановок
  const stopsToggle = $("#toggle-stops");
  if (stopsToggle) {
    stopsToggle.checked = state.showStops;
    stopsToggle.addEventListener("change", (event) => {
      state.showStops = event.target.checked;
      renderStops();
    });
  }

  // 3. Кнопка загруженности на карте
  const mapLoadBtn = $("#toggle-map-load");
  if (mapLoadBtn) {
    mapLoadBtn.addEventListener("click", () => {
      state.showMapLoad = !state.showMapLoad;
      mapLoadBtn.classList.toggle("active", state.showMapLoad);
      mapLoadBtn.textContent = state.showMapLoad ? "Скрыть загруженность" : "Загруженность";
      renderMap();
    });
  }

  // 4. Кнопки trips
  document.querySelectorAll("[data-trips]").forEach((button) => {
    button.addEventListener("click", () => {
      state.forecast.trips = Number(button.dataset.trips);
      refreshForecast();
    });
  });

  // 5. Остальные кнопки (CSV, Risk)
  $("#forecast-csv").addEventListener("click", downloadForecastCsv);
  $("#forecast-risk").addEventListener("click", (event) => {
    const button = event.target.closest("[data-period]");
    if (!button) return;
    selectForecastPeriod(button.dataset.period, button.dataset.hour);
  });
} // <-- Конец bindForecastControls
  // 3. Остальной код функции
  $("#forecast-csv").addEventListener("click", downloadForecastCsv);
  $("#forecast-risk").addEventListener("click", (event) => {
    const button = event.target.closest("[data-period]");
    if (!button) return;
    selectForecastPeriod(button.dataset.period, button.dataset.hour);
  });
 // <-- Эта скобка закрывает саму функцию bindForecastControls

function toggleHourPlay() {
  state.forecast.playing = !state.forecast.playing;
  clearInterval(state.forecast.playTimer);
  syncHourChip();
  if (!state.forecast.playing) return;
  state.forecast.playTimer = window.setInterval(() => {
    const hours = TramForecast.hourRange(state.forecast.hourFrom, state.forecast.hourTo);
    const index = Math.max(0, hours.indexOf(state.forecast.selectedHour));
    state.forecast.selectedHour = hours[(index + 1) % hours.length];
    refreshForecast();
  }, 800);
}

function downloadForecastCsv() {
  const route = state.routes.find((item) => item.route_id === state.selectedId);
  if (!route || !window.TramForecast?.state.loaded) return;
  const board = TramForecast.present(forecastView(route));
  const rows = board.rows.map((row) => ({
    период: row.period,
    маршрут: route.route_short_name,
    точка: "маршрут",
    базовые_посадки: row.baseValue ?? "",
    сценарные_посадки: row.scenarioValue ?? "",
    единицы: "посадки",
    статус: row.baseValue == null ? "нет данных" : board.scenario.apply ? board.statusLabel : row.statusLabel,
    коэффициент: board.scenario.coefficient,
    примечание: "прогноз маршрута",
  }));
  rows.push({
    период: "итого",
    маршрут: route.route_short_name,
    точка: "маршрут",
    базовые_посадки: board.totalBase ?? "",
    сценарные_посадки: board.totalScenario ?? "",
    единицы: "посадки",
    статус: board.statusLabel,
    коэффициент: board.scenario.coefficient,
    примечание: "прогноз маршрута",
  });
  const headers = Object.keys(rows[0] || { период: "" });
  const quote = (value) => `"${String(value ?? "").replaceAll('"', '""')}"`;
  const body = [headers.join(";"), ...rows.map((row) => headers.map((header) => quote(row[header])).join(";"))].join("\r\n");
  const blob = new Blob([`\uFEFF${body}`], { type: "text/csv;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  const [from, to] = [state.forecast.hourFrom, state.forecast.hourTo].sort((left, right) => left - right);
  link.download = `tram-${route.route_short_name}-${state.forecast.horizon}-${state.forecast.date}-${String(from).padStart(2, "0")}-${String(to).padStart(2, "0")}.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
}

function weekColumns(days) {
  const max = Math.max(...days.map((day) => day.peak.passengers), 1);
  return days.map((day) => {
    const peak = day.peak;
    return `<div class="hour-col" title="${escapeHtml(day.label)} · пик ${TramDemand.formatPassengers(peak.passengers)} пасс. в ${TramDemand.formatHour(peak.hour)} · ${TramDemand.formatLoad(peak.load)}"><span class="week-value">${TramDemand.formatPassengers(peak.passengers)}</span><div class="week-bar-slot"><div class="hour-bar ${TramDemand.loadTone(peak.load)}" style="height:${Math.max(12, (peak.passengers / max) * 100)}%"></div></div><span>${escapeHtml(day.shortLabel)}</span></div>`;
  }).join("");
}

function hourColumns(hours) {
  const max = Math.max(...hours.map((item) => item.passengers), 1);
  return hours.map((item) => `<div class="hour-col" title="${TramDemand.formatHour(item.hour)} · ${TramDemand.formatPassengers(item.passengers)} пасс. · ${TramDemand.formatLoad(item.load)}"><div class="hour-bar ${TramDemand.loadTone(item.load)}" style="height:${Math.max(8, (item.passengers / max) * 100)}%"></div><span>${String(item.hour).padStart(2, "0")}</span></div>`).join("");
}

function readMapState() {
  try {
    return JSON.parse(sessionStorage.getItem(MAP_STATE_KEY)) || null;
  } catch {
    return null;
  }
}

function saveMapState() {
  const route = state.routes.find((item) => item.route_id === state.selectedId);
  const cards = [...state.stationCards.entries()].map(([stopId, card]) => {
    const stop = stopsForRoute(state.selectedId).find((item) => item.stop_id === stopId);
    const name = stop?.stop_name || card.querySelector(".station-card-name")?.textContent || "";
    return {
      name,
      left: card.style.left,
      top: card.style.top,
      width: card.style.width,
      height: card.style.height,
      active: stopId === state.stationStopId,
    };
  }).filter((card) => card.name);
  const snapshot = {
    route: route?.route_short_name || "",
    search: $("#route-search")?.value || "",
    stationsOpen: $("#stations-panel")?.classList.contains("open") ?? true,
    lineScale: state.lineScale,
    sidebarWidth: $(".sidebar")?.style.width || "",
    routeCardWidth: $("#route-details")?.style.getPropertyValue("--route-card-width") || "",
    forecast: {
      horizon: state.forecast.horizon,
      date: state.forecast.date,
      hourFrom: state.forecast.hourFrom,
      hourTo: state.forecast.hourTo,
      selectedHour: state.forecast.selectedHour,
      segment: state.forecast.segment,
      trips: state.forecast.trips,
      scenario: { ...state.forecast.scenario },
    },
    cards,
  };
  sessionStorage.setItem(MAP_STATE_KEY, JSON.stringify(snapshot));
}

function hasExplicitMapLink() {
  const params = new URLSearchParams(location.search);
  const routeName = decodeURIComponent(location.hash.replace(/^#/, "").split("&")[0]);
  return Boolean(routeName || params.get("date") || params.get("hour"));
}

function applySavedLayout(saved) {
  const scale = Number(saved.lineScale);
  if (scale >= 0.5 && scale <= 2.2) {
    state.lineScale = scale;
    $("#line-size").value = String(scale);
  }
  const sidebarWidth = parseFloat(saved.sidebarWidth);
  if (sidebarWidth >= 220 && sidebarWidth <= 560) {
    const sidebar = $(".sidebar");
    sidebar.style.width = `${sidebarWidth}px`;
    sidebar.style.flexBasis = `${sidebarWidth}px`;
    state.map.invalidateSize({ pan: false });
  }
  const cardWidth = parseFloat(saved.routeCardWidth);
  if (cardWidth >= 280) {
    const card = $("#route-details");
    card.classList.add("is-sized");
    card.style.setProperty("--route-card-width", `${cardWidth}px`);
    $(".map-area").style.setProperty("--route-card-width", `${cardWidth}px`);
  }
  if (typeof saved.stationsOpen === "boolean") setStationsOpen(saved.stationsOpen);
}

function restoreForecast(forecast) {
  if (!forecast || typeof forecast !== "object") return;
  if (forecast.horizon === "day" || forecast.horizon === "month" || forecast.horizon === "year") state.forecast.horizon = forecast.horizon;
  if (isOpenForecastDate(forecast.date)) state.forecast.date = forecast.date;
  if (Number.isInteger(forecast.hourFrom) && forecast.hourFrom >= 0 && forecast.hourFrom <= 23) state.forecast.hourFrom = forecast.hourFrom;
  if (Number.isInteger(forecast.hourTo) && forecast.hourTo >= 0 && forecast.hourTo <= 23) state.forecast.hourTo = forecast.hourTo;
  if (Number.isInteger(forecast.selectedHour) && forecast.selectedHour >= 0 && forecast.selectedHour <= 23) state.forecast.selectedHour = forecast.selectedHour;
  if (typeof forecast.segment === "string") state.forecast.segment = forecast.segment;
  if (forecast.trips === 0 || forecast.trips === 1 || forecast.trips === 2) state.forecast.trips = forecast.trips;
  const scenario = forecast.scenario || {};
  if (scenario.weather === "base" || scenario.weather === "rain" || scenario.weather === "heavy" || scenario.weather === "snow") state.forecast.scenario.weather = scenario.weather;
  if (typeof scenario.eventOn === "boolean") state.forecast.scenario.eventOn = scenario.eventOn;
  if (typeof scenario.eventPlace === "string") state.forecast.scenario.eventPlace = scenario.eventPlace;
  if (typeof scenario.eventTime === "string") state.forecast.scenario.eventTime = scenario.eventTime;
  if (typeof scenario.eventCoeff === "string") state.forecast.scenario.eventCoeff = scenario.eventCoeff;
  if (scenario.seasonMode === "base" || scenario.seasonMode === "custom") state.forecast.scenario.seasonMode = scenario.seasonMode;
  if (typeof scenario.seasonCoeff === "string") state.forecast.scenario.seasonCoeff = scenario.seasonCoeff;
  state.forecast.playing = false;
}

function restoreStationCards(cards) {
  if (!Array.isArray(cards) || !state.selectedId) return;
  const stops = stopsForRoute(state.selectedId);
  cards.forEach((item) => {
    const stop = stops.find((entry) => entry.stop_name === item.name);
    if (stop) openStationCard(stop.stop_id, item);
  });
  const active = [...cards].reverse().find((item) => item.active) || cards[cards.length - 1];
  const focused = active && stops.find((entry) => entry.stop_name === active.name);
  if (focused) focusStationCard(focused.stop_id);
}

function applySavedMapState() {
  const saved = readMapState();
  if (!saved || state.usingDemo) return;
  applySavedLayout(saved);
  if (hasExplicitMapLink()) return;
  restoreForecast(saved.forecast);
  if (typeof saved.search === "string") $("#route-search").value = saved.search;
  const route = state.routes.find((item) => item.route_short_name === String(saved.route || ""));
  if (!route) {
    renderRouteList($("#route-search").value);
    syncHourChip();
    return;
  }
  selectRoute(route.route_id);
  restoreStationCards(saved.cards);
}

function selectFromHash() {
  const params = new URLSearchParams(location.search);
  const date = params.get("date");
  const hour = params.get("hour");
  if (isOpenForecastDate(date)) {
    state.forecast.date = date;
    state.forecast.horizon = "day";
  }
  if (hour != null && hour !== "" && Number(hour) >= 0 && Number(hour) <= 23) {
    state.forecast.selectedHour = Number(hour);
    state.forecast.horizon = "day";
  }
  const routeName = decodeURIComponent(location.hash.replace(/^#/, "").split("&")[0]);
  if (!routeName) return;
  const route = state.routes.find((item) => item.route_short_name === routeName);
  if (route) selectRoute(route.route_id);
}

function stopsForRoute(routeId) {
  const trip = state.feed?.trips.find((item) => item.route_id === routeId);
  if (!trip) return [];
  return state.feed.stopTimes.filter((item) => item.trip_id === trip.trip_id).sort((a, b) => a.stop_sequence - b.stop_sequence)
    .map((item) => state.feed.stops.find((stop) => stop.stop_id === item.stop_id)).filter(Boolean);
}

function updateStats() {
  const allStops = state.feed.stops.length;
  const distance = state.routes.reduce((total, route) => total + routeDistance(stopsForRoute(route.route_id)), 0);
  $("#stat-routes").textContent = state.routes.length;
  $("#stat-stops").textContent = allStops;
  $("#stat-distance").textContent = Math.round(distance);
}

function fitMap() {
  state.map.invalidateSize({ pan: false });
  setTimeout(() => state.map.invalidateSize({ pan: false }), 120);
}

async function downloadGtfs() {
  if (!state.feed || !window.JSZip) return;
  const zip = new JSZip();
  const files = {
    "agency.txt": csv(state.feed.agency),
    "routes.txt": csv(state.feed.routes.map(({ direction, ...route }) => route)),
    "stops.txt": csv(state.feed.stops),
    "trips.txt": csv(state.feed.trips),
    "stop_times.txt": csv(state.feed.stopTimes),
    "calendar.txt": csv(state.feed.calendar),
    "shapes.txt": csv(state.feed.shapes),
  };
  Object.entries(files).forEach(([name, content]) => zip.file(name, content));
  const blob = await zip.generateAsync({ type: "blob", compression: "DEFLATE" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "moscow-tram-gtfs.zip";
  link.click();
  URL.revokeObjectURL(link.href);
  showToast("GTFS ZIP сформирован и скачан");
}

function csv(rows) {
  if (!rows.length) return "";
  const headers = Object.keys(rows[0]);
  const quote = (value) => `"${String(value ?? "").replaceAll('"', '""')}"`;
  return [headers.join(","), ...rows.map((row) => headers.map((header) => quote(row[header])).join(","))].join("\r\n");
}

function makeDemoFeed() {
  const basePaths = [
    ["3", "Метро «Чистые пруды» — улица Академика Янгеля", [[55.764, 37.638], [55.752, 37.635], [55.737, 37.625], [55.721, 37.608], [55.704, 37.592], [55.682, 37.588]]],
    ["6", "Станция «Сокол» — Братцево", [[55.810, 37.516], [55.800, 37.535], [55.790, 37.556], [55.783, 37.579], [55.794, 37.608], [55.811, 37.624], [55.836, 37.625]]],
    ["17", "Останкино — Медведково", [[55.821, 37.619], [55.821, 37.647], [55.830, 37.674], [55.844, 37.686], [55.866, 37.677], [55.884, 37.660]]],
    ["39", "Метро «Университет» — Черёмушки", [[55.693, 37.526], [55.710, 37.542], [55.725, 37.558], [55.735, 37.579], [55.728, 37.606], [55.713, 37.625]]],
    ["43", "Метро «Пролетарская» — Автозаводская", [[55.731, 37.667], [55.720, 37.664], [55.706, 37.657], [55.696, 37.646], [55.694, 37.627], [55.700, 37.608]]],
    ["46", "Станция «Каланчёвская» — Сокольники", [[55.776, 37.652], [55.787, 37.650], [55.800, 37.656], [55.817, 37.662], [55.831, 37.669]]],
    ["47", "Новогиреево — Курский вокзал", [[55.752, 37.815], [55.760, 37.788], [55.764, 37.752], [55.765, 37.718], [55.760, 37.684], [55.757, 37.648]]],
    ["50", "Трамвайное депо — Тушино", [[55.858, 37.438], [55.846, 37.464], [55.834, 37.486], [55.826, 37.512], [55.819, 37.540], [55.810, 37.561]]],
  ];
  const routeNumbers = ["1", "7", "11", "12", "17", "25", "26", "28", "50"];
  const namedRoutes = {
    Т1: "Метрогородок — метро «Университет»",
    Т2: "Чертановская — МЦД «Новогиреево»",
  };
  const routes = [], stops = [], trips = [], stopTimes = [], shapes = [];
  routeNumbers.forEach((number, routeIndex) => {
    const [, , baseCoordinates] = basePaths[routeIndex % basePaths.length];
    const driftLat = ((routeIndex % 5) - 2) * 0.0018;
    const driftLon = ((Math.floor(routeIndex / 5) % 5) - 2) * 0.0018;
    const coordinates = baseCoordinates.map(([lat, lon]) => [lat + driftLat, lon + driftLon]);
    const name = namedRoutes[number] || `${number} · трамвайная линия`;
    const routeId = `route_${routeIndex + 1}`, tripId = `trip_${routeIndex + 1}`, shapeId = `shape_${routeIndex + 1}`;
    routes.push({ route_id: routeId, route_short_name: number, route_long_name: name, route_type: 0, route_color: routeColorFor(number, routeIndex), route_text_color: "FFFFFF", direction: "0" });
    trips.push({ route_id: routeId, service_id: "MOSCOW", trip_id: tripId, trip_headsign: name.split(" — ")[1] || name, direction_id: 0, shape_id: shapeId });
    coordinates.forEach(([lat, lon], index) => {
      const stopId = `stop_${routeIndex + 1}_${index + 1}`;
      stops.push({ stop_id: stopId, stop_code: "", stop_name: `${number} · остановка ${index + 1}`, stop_lat: lat, stop_lon: lon });
      stopTimes.push({ trip_id: tripId, arrival_time: formatTime(index * 5), departure_time: formatTime(index * 5), stop_id: stopId, stop_sequence: index + 1 });
      shapes.push({ shape_id: shapeId, shape_pt_lat: lat, shape_pt_lon: lon, shape_pt_sequence: index + 1 });
    });
  });
  return makeFeed({ routes, stops, trips, stopTimes, shapes });
}

function routeDistance(stops) {
  return stops.slice(1).reduce((sum, stop, index) => sum + haversine(stops[index], stop), 0);
}
function haversine(a, b) {
  const rad = Math.PI / 180, dLat = (b.stop_lat - a.stop_lat) * rad, dLon = (b.stop_lon - a.stop_lon) * rad;
  const x = Math.sin(dLat / 2) ** 2 + Math.cos(a.stop_lat * rad) * Math.cos(b.stop_lat * rad) * Math.sin(dLon / 2) ** 2;
  return 6371 * 2 * Math.atan2(Math.sqrt(x), Math.sqrt(1 - x));
}
function dedupePoints(points) { return points.filter((point, index) => index === 0 || point.lat !== points[index - 1].lat || point.lon !== points[index - 1].lon); }
function groupBy(items, key) { return items.reduce((result, item) => ((result[item[key]] ||= []).push(item), result), {}); }
function normalizeKey(key) { return String(key).toLowerCase().replaceAll(/[«»"']/g, "").replaceAll(/[_\s-]+/g, ""); }
function clean(value) { return value === null || value === undefined ? "" : String(value).trim(); }
function number(value) { const parsed = Number(String(value ?? "").replace(",", ".")); return Number.isFinite(parsed) ? parsed : null; }
function inMoscow(lat, lon) { return lat > 54.2 && lat < 56.6 && lon > 35.2 && lon < 40.5; }
function slug(value) { return clean(value).toLowerCase().replaceAll(/[^a-zа-яё0-9]+/gi, "_"); }
function formatTime(minutes) { return `${String(Math.floor(minutes / 60)).padStart(2, "0")}:${String(minutes % 60).padStart(2, "0")}:00`; }
function routeColorFor(routeId, index) {
  if (String(routeId) === "50") return "8a5a2b";
  return routeColor(index);
}

function applyRoutePois(factors, route) {
  if (!Array.isArray(factors) || !window.TramFacts || !TramFacts.state.pois?.length) return;
  const factor = factors.find((item) => item.label === "Места");
  if (!factor) return;
  const summary = TramFacts.poiSummary(route.route_id, stopsForRoute(route.route_id));
  factor.value = String(summary.count);
  factor.off = summary.count === 0;
  factor.title = summary.example ? `${summary.example}. В 300 м от остановок.` : "В 300 м от остановок точек нет";
}

function routeColor(index) {
  const hue = (index * 137.508) % 360;
  const saturation = 84 / 100;
  const lightness = 50 / 100;
  const channel = (n) => {
    const k = (n + hue / 30) % 12;
    return lightness - saturation * Math.min(lightness, 1 - lightness) * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  return [channel(0), channel(8), channel(4)].map((value) => Math.round(value * 255).toString(16).padStart(2, "0")).join("");
}
function escapeHtml(value) { return clean(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;"); }
function setStatus(text) { $("#data-status").textContent = text; }
function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("visible");
  clearTimeout(showToast.timeout);
  showToast.timeout = setTimeout(() => toast.classList.remove("visible"), 4500);
}
