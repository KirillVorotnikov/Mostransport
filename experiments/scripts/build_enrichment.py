"""Внешние признаки для почасового прогноза посадок.

Погода — архив Open-Meteo в центроиде остановок маршрута (Москва, если
остановок нет). Праздники — производственный календарь РФ на 2025 год.
Офис — остановка маршрута; офис с аварией — остановка, у которой к этой
дате уже было ДТП.
"""

from __future__ import annotations

import csv
import json
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "kaggle_enrichment"
STATIONS = ROOT / "data" / "enrichment" / "stations_averaged.csv"
EVENTS = ROOT / "data" / "enrichment" / "events_nearest_station.csv"

ROUTES = [1, 5, 7, 11, 12, 17, 25, 26, 28, 50]
MOSCOW = (55.751244, 37.618423)
START = date(2025, 1, 1)
END = date(2025, 12, 31)

NAMED = {
    date(2025, 1, 1): "newyear",
    date(2025, 1, 2): "newyear",
    date(2025, 1, 3): "newyear",
    date(2025, 1, 4): "newyear",
    date(2025, 1, 5): "newyear",
    date(2025, 1, 6): "newyear",
    date(2025, 1, 7): "christmas",
    date(2025, 1, 8): "newyear",
    date(2025, 2, 23): "defender",
    date(2025, 3, 8): "womens",
    date(2025, 5, 1): "labour",
    date(2025, 5, 9): "victory",
    date(2025, 6, 12): "russia",
    date(2025, 11, 4): "unity",
}
TRANSFERRED = {
    date(2025, 5, 2): "transfer",
    date(2025, 5, 8): "transfer",
    date(2025, 6, 13): "transfer",
    date(2025, 11, 3): "transfer",
    date(2025, 12, 31): "transfer",
}
SHORT_DAYS = {
    date(2025, 3, 7),
    date(2025, 4, 30),
    date(2025, 6, 11),
    date(2025, 11, 1),
}


def parse_routes(value):
    routes = []
    for part in str(value or "").replace(",", ";").split(";"):
        part = part.strip()
        if part.isdigit():
            routes.append(int(part))
    return routes


def load_offices():
    offices = []
    with STATIONS.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            for route in parse_routes(row["маршруты"]):
                if route not in ROUTES:
                    continue
                offices.append({
                    "route": route,
                    "station": row["станция"],
                    "lat": float(row["широта"]),
                    "lon": float(row["долгота"]),
                })
    return offices


def load_accidents():
    accidents = []
    with EVENTS.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["тип"] != "авария" or not row["дата"]:
                continue
            moment = datetime.strptime(row["дата"], "%Y-%m-%d %H:%M:%S")
            for route in parse_routes(row["маршруты"]):
                if route not in ROUTES:
                    continue
                accidents.append({
                    "route": route,
                    "date": moment.date().isoformat(),
                    "station": row["станция"],
                    "tram": int(float(row["трамвай"] or 0)),
                })
    return accidents


def centroids(offices):
    buckets = defaultdict(list)
    for office in offices:
        buckets[office["route"]].append((office["lat"], office["lon"]))
    points = {}
    for route in ROUTES:
        samples = buckets.get(route)
        if not samples:
            points[route] = MOSCOW
            continue
        lat = sum(item[0] for item in samples) / len(samples)
        lon = sum(item[1] for item in samples) / len(samples)
        points[route] = (lat, lon)
    return points


def fetch_weather(route, lat, lon):
    query = urllib.parse.urlencode({
        "latitude": f"{lat:.5f}",
        "longitude": f"{lon:.5f}",
        "start_date": START.isoformat(),
        "end_date": END.isoformat(),
        "hourly": "temperature_2m,precipitation,wind_speed_10m,weather_code,snowfall",
        "timezone": "Europe/Moscow",
    })
    request = urllib.request.Request(
        "https://archive-api.open-meteo.com/v1/archive?" + query,
        headers={"User-Agent": "mostransport-bert-enrichment/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    hourly = payload["hourly"]
    rows = []
    for stamp, temp, precip, wind, code, snow in zip(
        hourly["time"],
        hourly["temperature_2m"],
        hourly["precipitation"],
        hourly["wind_speed_10m"],
        hourly["weather_code"],
        hourly["snowfall"],
    ):
        day, hour = stamp.split("T")
        rows.append({
            "route": route,
            "date": day,
            "hour": int(hour[:2]),
            "temp": temp if temp is not None else "",
            "precip": precip if precip is not None else "",
            "wind": wind if wind is not None else "",
            "code": code if code is not None else "",
            "snowfall": snow if snow is not None else "",
        })
    return rows


def fetch_day_off():
    request = urllib.request.Request(
        "https://isdayoff.ru/api/getdata?year=2025&cc=ru",
        headers={"User-Agent": "mostransport-bert-enrichment/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read().decode("ascii").strip()
    if len(raw) != 365 or any(char not in "01" for char in raw):
        raise RuntimeError(f"unexpected production calendar: {len(raw)} chars")
    return raw


def route_day_rows(offices, accidents):
    by_route_offices = defaultdict(set)
    for office in offices:
        by_route_offices[office["route"]].add(office["station"])
    by_route_events = defaultdict(list)
    for event in accidents:
        by_route_events[event["route"]].append((
            date.fromisoformat(event["date"]),
            event["station"],
            event["tram"],
        ))
    rows = []
    for route in ROUTES:
        stations = by_route_offices.get(route, set())
        events = sorted(by_route_events.get(route, []))
        day = START
        while day <= END:
            acc7 = acc30 = tram7 = 0
            hit = set()
            for event_day, station, tram in events:
                if event_day >= day:
                    break
                if station in stations:
                    hit.add(station)
                age = (day - event_day).days
                if age <= 30:
                    acc30 += 1
                    if age <= 7:
                        acc7 += 1
                        tram7 += tram
            rows.append({
                "route": route,
                "date": day.isoformat(),
                "offices": len(stations),
                "offices_hit": len(hit),
                "acc7": acc7,
                "acc30": acc30,
                "tram7": tram7,
            })
            day += timedelta(days=1)
    return rows


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    offices = load_offices()
    accidents = load_accidents()
    write_csv(OUT / "route_offices.csv", offices, ["route", "station", "lat", "lon"])
    write_csv(
        OUT / "office_accidents.csv",
        accidents,
        ["route", "date", "station", "tram"],
    )

    day_off = fetch_day_off()
    holiday_rows = []
    day = START
    index = 0
    while day <= END:
        name = NAMED.get(day) or TRANSFERRED.get(day) or "none"
        tomorrow = day + timedelta(days=1)
        tomorrow_name = NAMED.get(tomorrow) or TRANSFERRED.get(tomorrow) or "none"
        holiday_rows.append({
            "date": day.isoformat(),
            "day_off": int(day_off[index]),
            "weekend": int(day.weekday() >= 5),
            "short_day": int(day in SHORT_DAYS),
            "holiday": name,
            "preholiday": int(tomorrow_name != "none"),
        })
        day += timedelta(days=1)
        index += 1
    write_csv(
        OUT / "holidays_2025.csv",
        holiday_rows,
        ["date", "day_off", "weekend", "short_day", "holiday", "preholiday"],
    )

    weather = []
    for route, (lat, lon) in centroids(offices).items():
        part = fetch_weather(route, lat, lon)
        if len(part) != 365 * 24:
            raise RuntimeError(f"route {route}: expected {365 * 24} hours, got {len(part)}")
        weather.extend(part)
        print(f"weather route {route}: {len(part)} hours at {lat:.4f},{lon:.4f}")
        time.sleep(0.3)
    write_csv(
        OUT / "weather_hourly.csv",
        weather,
        ["route", "date", "hour", "temp", "precip", "wind", "code", "snowfall"],
    )
    days = route_day_rows(offices, accidents)
    write_csv(
        OUT / "route_day.csv",
        days,
        ["route", "date", "offices", "offices_hit", "acc7", "acc30", "tram7"],
    )
    print(f"offices {len(offices)} accidents {len(accidents)} weather {len(weather)} route-days {len(days)}")


if __name__ == "__main__":
    main()
