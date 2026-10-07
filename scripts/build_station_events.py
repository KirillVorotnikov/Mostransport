"""Привязка открытых событий к трамвайным остановкам.

Координаты остановки берутся с трассы маршрута и усредняются, если одно и то же
имя стоит на нескольких линиях рядом. Событие (ДТП, ремонт, точка интереса)
относится к ближайшей остановке. В сводный CSV попадают средние по станции.
"""

from __future__ import annotations

import csv
import json
import math
import re
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTES_PATH = ROOT / "data" / "tram_routes_3221.geojson"
CACHE_DIR = ROOT / "data" / "enrichment" / "cache"
OUT_DIR = ROOT / "data" / "enrichment"
ROUTE_SUMMARY = OUT_DIR / "routes_enriched.csv"

NEAR_M = 150
POI_NEAR_M = 700
SAME_STOP_M = 1600
SAMPLE_STEP_M = 40
YEARS = {"2024", "2025", "2026"}
STREET_MARKS = ("улица", "переулок", "проезд", "проспект", "шоссе", "тупик", "набережная", "бульвар", "аллея")
CELL = 0.0015


def haversine_m(lat1, lon1, lat2, lon2):
    rad = math.radians
    dlat = rad(lat2 - lat1)
    dlon = rad(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rad(lat1)) * math.cos(rad(lat2)) * math.sin(dlon / 2) ** 2
    return 6371000 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def cell_key(lat, lon):
    return (int(lat / CELL), int(lon / CELL))


def continuation(previous, token):
    if re.match(r"^[а-яё]", token):
        return True
    if previous == "Покровское" and re.match(r"^(Глебово|Стрешнево)$", token):
        return True
    return previous == "Свято" and token.startswith("Данилов")


def parse_stops(track):
    stops = []
    for token in re.split(r"\s+-\s+", track or ""):
        token = token.strip()
        if not token:
            continue
        previous = stops[-1] if stops else None
        if previous and continuation(previous, token):
            stops[-1] = f"{previous} - {token}"
            continue
        if previous == token:
            continue
        stops.append(token)
    return stops


def flatten_coords(geometry):
    points = []

    def walk(node):
        if isinstance(node, list) and node and isinstance(node[0], (int, float)):
            lon, lat = node[0], node[1]
            points.append((lat, lon))
            return
        if isinstance(node, list):
            for item in node:
                walk(item)

    walk((geometry or {}).get("coordinates") or [])
    return points


def place_stops(points, names):
    if not points or not names:
        return []
    if len(names) == 1:
        return [{"name": names[0], "lat": points[0][0], "lon": points[0][1]}]
    distances = [0.0]
    for index in range(1, len(points)):
        distances.append(distances[-1] + haversine_m(*points[index - 1], *points[index]))
    total = distances[-1] or 0.0
    placed = []
    for name_index, name in enumerate(names):
        target = total * (name_index / (len(names) - 1))
        segment = max(1, next((index for index, distance in enumerate(distances) if distance >= target), len(distances) - 1))
        previous = distances[segment - 1]
        span = distances[segment] - previous or 1.0
        ratio = min(1.0, max(0.0, (target - previous) / span))
        start = points[segment - 1]
        finish = points[min(segment, len(points) - 1)]
        placed.append({
            "name": name,
            "lat": start[0] + (finish[0] - start[0]) * ratio,
            "lon": start[1] + (finish[1] - start[1]) * ratio,
        })
    return placed


def sample_line(points, step_m):
    samples = []
    if not points:
        return samples
    samples.append(points[0])
    carry = 0.0
    for index in range(1, len(points)):
        lat1, lon1 = points[index - 1]
        lat2, lon2 = points[index]
        dist = haversine_m(lat1, lon1, lat2, lon2)
        if dist == 0:
            continue
        walked = step_m - carry
        while walked <= dist:
            share = walked / dist
            samples.append((lat1 + (lat2 - lat1) * share, lon1 + (lon2 - lon1) * share))
            walked += step_m
        carry = dist - (walked - step_m)
    samples.append(points[-1])
    return samples


def index_points(points):
    buckets = defaultdict(list)
    for lat, lon in points:
        buckets[cell_key(lat, lon)].append((lat, lon))
    return buckets


def near_line(lat, lon, buckets, radius_m):
    span = max(1, math.ceil(radius_m / (CELL * 111_000)))
    base_i, base_j = cell_key(lat, lon)
    for di in range(-span, span + 1):
        for dj in range(-span, span + 1):
            for sample_lat, sample_lon in buckets.get((base_i + di, base_j + dj), ()):
                if haversine_m(lat, lon, sample_lat, sample_lon) <= radius_m:
                    return True
    return False


def cluster_same_name(placements):
    groups = defaultdict(list)
    for item in placements:
        groups[item["name"]].append(item)
    stations = []
    for name, items in groups.items():
        pending = list(items)
        while pending:
            seed = pending.pop(0)
            group = [seed]
            changed = True
            while changed:
                changed = False
                rest = []
                for point in pending:
                    if any(haversine_m(point["lat"], point["lon"], member["lat"], member["lon"]) <= SAME_STOP_M for member in group):
                        group.append(point)
                        changed = True
                    else:
                        rest.append(point)
                pending = rest
            stations.append({
                "name": name,
                "lat": sum(point["lat"] for point in group) / len(group),
                "lon": sum(point["lon"] for point in group) / len(group),
                "routes": sorted({point["route"] for point in group}, key=lambda value: (len(value), value)),
            })
    stations.sort(key=lambda item: (item["name"], item["lat"], item["lon"]))
    return stations


def nearest_station(lat, lon, stations):
    best = None
    best_distance = None
    for station in stations:
        distance = haversine_m(lat, lon, station["lat"], station["lon"])
        if best_distance is None or distance < best_distance:
            best = station
            best_distance = distance
    return best, best_distance


def as_lat_lon(first, second):
    if abs(first) > 50:
        return first, second
    return second, first


def load_accidents():
    archive = CACHE_DIR / "moskva.geojson.zip"
    with zipfile.ZipFile(archive) as bundle:
        name = next(item for item in bundle.namelist() if item.endswith(".geojson") or item.endswith(".json"))
        raw = json.loads(bundle.read(name))
    records = raw["features"] if isinstance(raw, dict) and raw.get("type") == "FeatureCollection" else raw
    found = {}
    for record in records:
        props = record.get("properties") or record
        when = str(props.get("datetime") or props.get("DATE_TIME") or props.get("date") or "")
        if when[:4] not in YEARS:
            continue
        pair = (props.get("POINT") or {}).get("coordinates") or (record.get("geometry") or {}).get("coordinates")
        if pair and len(pair) >= 2:
            lat, lon = as_lat_lon(pair[0], pair[1])
        elif props.get("LAT") is not None and props.get("LNG") is not None:
            lat, lon = float(props["LAT"]), float(props["LNG"])
        else:
            continue
        if not (55.4 <= lat <= 56.1 and 37.1 <= lon <= 38.0):
            continue
        text_bits = [props.get("category"), props.get("address"), props.get("tags"), props.get("EM_TYPE"), props.get("STREET")]
        for vehicle in props.get("vehicles") or []:
            text_bits.extend((vehicle.get("category"), vehicle.get("brand")))
        blob = " ".join(str(bit or "") for bit in text_bits).lower()
        event_id = str(props.get("id") or props.get("gibdd_number") or props.get("EM_NUMBER") or f"{when}|{lat:.5f}|{lon:.5f}")
        found[event_id] = {
            "event_type": "авария",
            "event_id": event_id,
            "title": str(props.get("category") or props.get("address") or ""),
            "lat": lat,
            "lon": lon,
            "when": when[:19],
            "injured": number_or_blank(props.get("injured_count")),
            "dead": number_or_blank(props.get("dead_count")),
            "participants": number_or_blank(props.get("participants_count")),
            "tram": "трамв" in blob,
        }
    return list(found.values())


def number_or_blank(value):
    if value is None or value == "":
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def load_overpass():
    repairs = {}
    roads = {}
    for path in sorted(CACHE_DIR.glob("overpass_*.json")):
        if path.stat().st_size <= 2:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for element in payload.get("elements", []):
            center = element.get("center") or {}
            lat = element.get("lat", center.get("lat"))
            lon = element.get("lon", center.get("lon"))
            if lat is None or lon is None:
                continue
            tags = element.get("tags") or {}
            highway = tags.get("highway") or ""
            event_id = f"{element.get('type')}/{element.get('id')}"
            item = {
                "event_id": event_id,
                "title": tags.get("name") or tags.get("construction") or highway,
                "lat": lat,
                "lon": lon,
                "when": "",
                "injured": "",
                "dead": "",
                "participants": "",
                "tram": False,
            }
            if highway in {"motorway", "trunk", "primary"}:
                item["event_type"] = "магистраль"
                roads[event_id] = item
            else:
                item["event_type"] = "ремонт"
                repairs[event_id] = item
    return list(repairs.values()), list(roads.values())


def load_pois():
    found = {}
    for path in CACHE_DIR.glob("wiki_*.json"):
        if path.stat().st_size <= 2:
            continue
        try:
            articles = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for article in articles:
            title = article.get("title") or ""
            if any(mark in title.lower() for mark in STREET_MARKS):
                continue
            if article.get("lat") is None or article.get("lon") is None:
                continue
            found[article.get("pageid") or title] = {
                "event_type": "точка интереса",
                "event_id": str(article.get("pageid") or title),
                "title": title,
                "lat": article["lat"],
                "lon": article["lon"],
                "when": "",
                "injured": "",
                "dead": "",
                "participants": "",
                "tram": False,
            }
    return list(found.values())


def load_route_weather():
    weather = {}
    with ROUTE_SUMMARY.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            weather[row["route_number"]] = row
    return weather


def mean(values):
    numbers = [value for value in values if value != ""]
    if not numbers:
        return ""
    return round(sum(numbers) / len(numbers), 2)


def main():
    payload = json.loads(ROUTES_PATH.read_text(encoding="utf-8"))
    placements = []
    line_samples = []
    for feature in payload["features"]:
        props = feature["properties"]
        route = str(props.get("RouteNumber") or "")
        points = flatten_coords(feature.get("geometry"))
        line_samples.extend(sample_line(points, SAMPLE_STEP_M))
        for stop in place_stops(points, parse_stops(props.get("TrackOfFollowing") or "")):
            placements.append({**stop, "route": route})
    stations = cluster_same_name(placements)
    line_index = index_points(line_samples)

    accidents = [item for item in load_accidents() if near_line(item["lat"], item["lon"], line_index, NEAR_M)]
    repairs, roads = load_overpass()
    repairs = [item for item in repairs if near_line(item["lat"], item["lon"], line_index, NEAR_M)]
    roads = [item for item in roads if near_line(item["lat"], item["lon"], line_index, NEAR_M)]
    pois = [item for item in load_pois() if near_line(item["lat"], item["lon"], line_index, POI_NEAR_M)]

    events = accidents + repairs + pois
    for event in events + roads:
        station, distance = nearest_station(event["lat"], event["lon"], stations)
        event["station"] = station["name"]
        event["routes"] = station["routes"]
        event["station_lat"] = station["lat"]
        event["station_lon"] = station["lon"]
        event["distance_m"] = round(distance, 1)

    by_station = defaultdict(list)
    roads_by_station = defaultdict(int)
    for event in events:
        by_station[(event["station"], round(event["station_lat"], 5), round(event["station_lon"], 5))].append(event)
    for road in roads:
        roads_by_station[(road["station"], round(road["station_lat"], 5), round(road["station_lon"], 5))] += 1

    weather = load_route_weather()
    station_rows = []
    for station in stations:
        key = (station["name"], round(station["lat"], 5), round(station["lon"], 5))
        assigned = by_station.get(key, [])
        accidents_here = [item for item in assigned if item["event_type"] == "авария"]
        repairs_here = [item for item in assigned if item["event_type"] == "ремонт"]
        pois_here = [item for item in assigned if item["event_type"] == "точка интереса"]
        route_weather = [weather[route] for route in station["routes"] if route in weather]
        texts = [row["weather"] for row in route_weather if row.get("weather")]
        weather_text = max(set(texts), key=texts.count) if texts else ""
        poi_titles = sorted({item["title"] for item in pois_here})[:8]
        station_rows.append({
            "станция": station["name"],
            "широта": round(station["lat"], 6),
            "долгота": round(station["lon"], 6),
            "маршруты": ";".join(station["routes"]),
            "аварий": len(accidents_here),
            "аварий_с_трамваем": sum(1 for item in accidents_here if item["tram"]),
            "раненых_в_среднем": mean([item["injured"] for item in accidents_here]),
            "погибших_в_среднем": mean([item["dead"] for item in accidents_here]),
            "участников_в_среднем": mean([item["participants"] for item in accidents_here]),
            "ремонтов": len(repairs_here),
            "магистралей": roads_by_station.get(key, 0),
            "точек_интереса": len(pois_here),
            "примеры_точек": "; ".join(poi_titles),
            "температура": mean([float(row["temperature_c"]) for row in route_weather]),
            "ветер_км_ч": mean([float(row["wind_kmh"]) for row in route_weather]),
            "осадки_мм": mean([float(row["precipitation_mm"]) for row in route_weather]),
            "погода": weather_text,
            "средняя_дистанция_м": mean([item["distance_m"] for item in assigned]),
        })

    station_path = OUT_DIR / "stations_averaged.csv"
    event_path = OUT_DIR / "events_nearest_station.csv"
    write_csv(station_path, station_rows)
    event_rows = [{
        "тип": event["event_type"],
        "ид": event["event_id"],
        "название": event["title"],
        "широта": round(event["lat"], 6),
        "долгота": round(event["lon"], 6),
        "дата": event["when"],
        "станция": event["station"],
        "маршруты": ";".join(event["routes"]),
        "широта_станции": round(event["station_lat"], 6),
        "долгота_станции": round(event["station_lon"], 6),
        "дистанция_м": event["distance_m"],
        "раненых": event["injured"],
        "погибших": event["dead"],
        "трамвай": int(bool(event["tram"])),
    } for event in events]
    event_rows.sort(key=lambda row: (row["станция"], row["тип"], row["дата"], row["название"]))
    write_csv(event_path, event_rows)

    with_accidents = sum(1 for row in station_rows if row["аварий"])
    print(f"остановок {len(station_rows)}")
    print(f"событий {len(events)}: аварии {len(accidents)}, ремонты {len(repairs)}, точки {len(pois)}")
    print(f"остановок с авариями {with_accidents}")
    print(station_path)
    print(event_path)


def write_csv(path, rows):
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
