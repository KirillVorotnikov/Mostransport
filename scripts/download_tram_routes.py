#!/usr/bin/env python3
"""Download and filter Moscow tram routes from data.mos.ru dataset 3221.

The script uses only Python's standard library and does not store the API key.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen


DEFAULT_ENDPOINT = "https://apidata.mos.ru/v1/datasets/3221/features"
DEFAULT_OUTPUT = Path("data/tram_routes_3221.geojson")
TYPE_KEY_HINTS = (
    "routetype",
    "transporttype",
    "typeoftransport",
    "vehicletype",
    "typeobject",
    "transport",
    "видтранспорта",
    "типтранспорта",
    "видтс",
    "типтс",
    "видмаршрута",
)
TRAM_RE = re.compile(r"\btram(?:way)?\b|трамва", re.IGNORECASE)


def main() -> int:
    args = parse_args()
    api_key = args.api_key or os.getenv("MOS_API_KEY")
    if not api_key:
        print(
            "Не найден API-ключ. Передайте --api-key или задайте переменную MOS_API_KEY.",
            file=sys.stderr,
        )
        return 2

    try:
        features = download_features(
            endpoint=args.endpoint,
            api_key=api_key,
            page_size=args.page_size,
            timeout=args.timeout,
        )
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as error:
        print(f"Ошибка загрузки data.mos.ru: {error}", file=sys.stderr)
        return 1

    tram_features = [feature for feature in features if is_tram(feature)]
    if not tram_features:
        print(
            f"Получено записей: {len(features)}, трамвайных записей не найдено. "
            "Проверьте поля типа транспорта в ответе API.",
            file=sys.stderr,
        )
        return 1

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    save_geojson(output, tram_features)
    csv_output = output.with_suffix(".csv")
    save_csv(csv_output, tram_features)
    route_ids = {
        first_value(feature.get("properties") or {}, ("route_id", "route number", "RouteNumber", "номер маршрута", "маршрут"))
        for feature in tram_features
    }
    route_ids.discard("")

    print(f"Получено GeoJSON-записей: {len(features)}")
    print(f"Оставлено трамвайных записей: {len(tram_features)}")
    print(f"Уникальных номеров маршрутов: {len(route_ids)}")
    print(f"GeoJSON: {output}")
    print(f"CSV:     {csv_output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Получить только трамвайные маршруты Москвы из data.mos.ru."
    )
    parser.add_argument(
        "--api-key",
        help="API-ключ data.mos.ru; безопаснее использовать переменную MOS_API_KEY.",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"URL GeoJSON API (по умолчанию: {DEFAULT_ENDPOINT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Файл GeoJSON (по умолчанию: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=1000,
        help="Количество записей на страницу (по умолчанию: 1000).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=45,
        help="Таймаут одного запроса в секундах (по умолчанию: 45).",
    )
    return parser.parse_args()


def download_features(
    endpoint: str,
    api_key: str,
    page_size: int,
    timeout: int,
) -> list[dict[str, Any]]:
    if page_size < 1:
        raise ValueError("--page-size должен быть больше нуля")

    features: list[dict[str, Any]] = []
    skip = 0

    while True:
        url = add_query(
            endpoint,
            {"$top": page_size, "$skip": skip, "api_key": api_key},
        )
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "moscow-tram-gtfs/1.0",
            },
        )
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)

        page = normalise_payload(payload)
        if not page:
            break
        features.extend(page)
        skip += len(page)
        print(f"\rЗагружено записей: {len(features)}", end="", flush=True)

        total = payload.get("Count") or payload.get("count") if isinstance(payload, dict) else None
        if total and skip >= int(total):
            break
        if len(page) < page_size:
            break

    print()
    return features


def add_query(url: str, values: dict[str, Any]) -> str:
    parsed = urlsplit(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update({key: str(value) for key, value in values.items()})
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment)
    )


def normalise_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("features"), list):
            return [normalise_feature(item) for item in payload["features"]]
        for key in ("rows", "Rows", "data", "Data"):
            if isinstance(payload.get(key), list):
                return [normalise_feature(item) for item in payload[key]]
        return []
    if isinstance(payload, list):
        return [normalise_feature(item) for item in payload]
    return []


def normalise_feature(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {"type": "Feature", "geometry": None, "properties": {"value": item}}

    attributes = (
        item.get("attributes")
        or item.get("properties")
        or item.get("Cells")
        or item.get("cells")
        or {}
    )
    if isinstance(attributes, dict) and isinstance(attributes.get("attributes"), dict):
        attributes = attributes["attributes"]
    geometry = item.get("geometry") or item.get("geoData") or item.get("geo_data")
    if not isinstance(attributes, dict):
        attributes = {"value": attributes}

    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": attributes,
    }


def is_tram(feature: dict[str, Any]) -> bool:
    properties = feature.get("properties") or {}
    if not isinstance(properties, dict):
        return False

    explicit_type_values: list[str] = []
    all_values: list[str] = []
    for key, value in properties.items():
        normalised_key = normalise_key(key)
        text = str(value or "").strip()
        all_values.append(text)
        if any(hint in normalised_key for hint in TYPE_KEY_HINTS):
            explicit_type_values.append(text)

    if explicit_type_values:
        return any(
            TRAM_RE.search(value) or value.strip().lower() == "0"
            for value in explicit_type_values
        )
    return bool(TRAM_RE.search(" ".join(all_values)))


def normalise_key(value: Any) -> str:
    return re.sub(r"[\s_«»\"'()-]+", "", str(value).lower())


def save_geojson(path: Path, features: list[dict[str, Any]]) -> None:
    document = {
        "type": "FeatureCollection",
        "name": "Moscow tram routes",
        "source": "https://data.mos.ru/opendata/3221",
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "features": features,
    }
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_csv(path: Path, features: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["route_id", "route_name", "transport_type", "geometry_type"])
        for index, feature in enumerate(features, start=1):
            properties = feature.get("properties") or {}
            writer.writerow(
                [
                    first_value(properties, ("route_id", "route number", "RouteNumber", "номер маршрута", "маршрут"))
                    or index,
                    first_value(properties, ("route_name", "route name", "RouteName", "наименование маршрута", "название")),
                    first_value(properties, ("route_type", "transport_type", "TypeOfTransport", "вид транспорта", "тип транспорта")),
                    (feature.get("geometry") or {}).get("type", ""),
                ]
            )


def first_value(properties: dict[str, Any], names: tuple[str, ...]) -> str:
    wanted = {normalise_key(name) for name in names}
    for key, value in properties.items():
        if normalise_key(key) in wanted:
            return str(value or "").strip()
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
