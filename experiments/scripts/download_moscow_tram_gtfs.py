#!/usr/bin/env python3
"""Download Moscow GTFS and export dated tram stop departures for 2025."""

import argparse
import csv
import io
import os
import zipfile
from datetime import date, timedelta
from urllib.request import urlopen


START = date(2025, 1, 1)
END = date(2025, 12, 31)


def read_csv(zf, name):
    with zf.open(name) as stream:
        return list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig")))


def dates_for_service(calendar, calendar_dates):
    result = {}
    for row in calendar:
        start = date.fromisoformat(row["start_date"])
        end = date.fromisoformat(row["end_date"])
        day = start
        while day <= end:
            weekday = day.strftime("%A").lower()
            if START <= day <= END and row.get(weekday) == "1":
                result.setdefault(row["service_id"], set()).add(day)
            day += timedelta(days=1)

    for row in calendar_dates:
        day = date.fromisoformat(row["date"])
        if START <= day <= END:
            days = result.setdefault(row["service_id"], set())
            if row["exception_type"] == "1":
                days.add(day)
            elif row["exception_type"] == "2":
                days.discard(day)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="URL of a GTFS ZIP archive")
    parser.add_argument("-o", "--output", default="moscow_tram_schedule_2025.csv")
    args = parser.parse_args()

    if os.path.isfile(args.url):
        with open(args.url, "rb") as source:
            payload = source.read()
    else:
        with urlopen(args.url, timeout=120) as response:
            payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        routes = {
            row["route_id"]: row
            for row in read_csv(zf, "routes.txt")
            if row.get("route_type") == "0"
        }
        stops = {
            row["stop_id"]: row
            for row in read_csv(zf, "stops.txt")
        }
        trips = [
            row for row in read_csv(zf, "trips.txt")
            if row["route_id"] in routes
        ]
        service_dates = dates_for_service(
            read_csv(zf, "calendar.txt") if "calendar.txt" in zf.namelist() else [],
            read_csv(zf, "calendar_dates.txt") if "calendar_dates.txt" in zf.namelist() else [],
        )
        trip_by_id = {row["trip_id"]: row for row in trips}
        stop_times = read_csv(zf, "stop_times.txt")

    fields = [
        "service_date", "route_id", "route_short_name", "route_long_name",
        "trip_id", "direction_id", "stop_sequence", "stop_id", "stop_name",
        "stop_lat", "stop_lon", "arrival_time", "departure_time",
        "trip_headsign",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in stop_times:
            trip = trip_by_id.get(row["trip_id"])
            if not trip:
                continue
            route = routes[trip["route_id"]]
            stop = stops.get(row["stop_id"], {})
            for service_date in sorted(service_dates.get(trip["service_id"], ())):
                writer.writerow({
                    "service_date": service_date.isoformat(),
                    "route_id": route["route_id"],
                    "route_short_name": route.get("route_short_name", ""),
                    "route_long_name": route.get("route_long_name", ""),
                    "trip_id": trip["trip_id"],
                    "direction_id": trip.get("direction_id", ""),
                    "stop_sequence": row["stop_sequence"],
                    "stop_id": row["stop_id"],
                    "stop_name": stop.get("stop_name", ""),
                    "stop_lat": stop.get("stop_lat", ""),
                    "stop_lon": stop.get("stop_lon", ""),
                    "arrival_time": row.get("arrival_time", ""),
                    "departure_time": row.get("departure_time", ""),
                    "trip_headsign": trip.get("trip_headsign", ""),
                })


if __name__ == "__main__":
    main()
