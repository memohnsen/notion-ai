#!/usr/bin/env python3
"""Fetch Notion database rows, analyze with OpenRouter, and send a Slack summary."""
# Local run (venv): python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python scripts/notion_instagram_report.py

import csv
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

import requests
from dotenv import load_dotenv

NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_API_VERSION = os.getenv("NOTION_API_VERSION", "2025-09-03")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

DEFAULT_MODEL = "google/gemini-3-flash-preview"

def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def optional_env(name: str) -> str:
    value = os.getenv(name)
    return value.strip() if value else ""


def normalize_database_id(database_id: str) -> str:
    return database_id.strip().replace("-", "")


def notion_headers(notion_api_key: str, version: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {notion_api_key}",
        "Notion-Version": version,
        "Content-Type": "application/json",
    }


def extract_property_value(prop: Dict[str, Any]) -> str:
    prop_type = prop.get("type")

    if prop_type == "title":
        return "".join(text.get("plain_text", "") for text in prop.get("title", []))
    if prop_type == "rich_text":
        return "".join(text.get("plain_text", "") for text in prop.get("rich_text", []))
    if prop_type == "number":
        number = prop.get("number")
        return "" if number is None else str(number)
    if prop_type == "select":
        select = prop.get("select")
        return "" if not select else select.get("name", "")
    if prop_type == "multi_select":
        return ", ".join(item.get("name", "") for item in prop.get("multi_select", []))
    if prop_type == "date":
        date = prop.get("date")
        if not date:
            return ""
        start = date.get("start", "")
        end = date.get("end")
        return f"{start} -> {end}" if end else start
    if prop_type == "checkbox":
        return str(prop.get("checkbox", False))
    if prop_type == "url":
        return prop.get("url", "") or ""
    if prop_type == "email":
        return prop.get("email", "") or ""
    if prop_type == "phone_number":
        return prop.get("phone_number", "") or ""
    if prop_type == "people":
        return ", ".join(person.get("name", "") or person.get("id", "") for person in prop.get("people", []))
    if prop_type == "files":
        return ", ".join(file.get("name", "") or file.get("file", {}).get("url", "") for file in prop.get("files", []))
    if prop_type == "relation":
        return ", ".join(item.get("id", "") for item in prop.get("relation", []))
    if prop_type == "formula":
        formula = prop.get("formula", {})
        formula_type = formula.get("type")
        return "" if formula_type is None else str(formula.get(formula_type, ""))
    if prop_type == "status":
        status = prop.get("status")
        return "" if not status else status.get("name", "")
    if prop_type == "created_time":
        return prop.get("created_time", "") or ""
    if prop_type == "last_edited_time":
        return prop.get("last_edited_time", "") or ""
    if prop_type == "created_by":
        created_by = prop.get("created_by")
        return "" if not created_by else created_by.get("name", "") or created_by.get("id", "")
    if prop_type == "last_edited_by":
        last_edited_by = prop.get("last_edited_by")
        return "" if not last_edited_by else last_edited_by.get("name", "") or last_edited_by.get("id", "")
    if prop_type == "rollup":
        rollup = prop.get("rollup", {})
        rollup_type = rollup.get("type")
        if rollup_type == "number":
            number = rollup.get("number")
            return "" if number is None else str(number)
        if rollup_type == "date":
            date = rollup.get("date")
            if not date:
                return ""
            start = date.get("start", "")
            end = date.get("end")
            return f"{start} -> {end}" if end else start
        if rollup_type == "array":
            return ", ".join(extract_property_value(item) for item in rollup.get("array", []))
        return ""

    return str(prop.get(prop_type, ""))


def fetch_notion_pages(notion_api_key: str, database_id: str) -> List[Dict[str, Any]]:
    url = f"{NOTION_API_BASE}/databases/{database_id}/query"
    version = NOTION_API_VERSION
    headers = notion_headers(notion_api_key, version)

    pages: List[Dict[str, Any]] = []
    payload: Dict[str, Any] = {}
    tried_fallback = False

    while True:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text.strip()
            if not tried_fallback:
                try:
                    error_payload = response.json()
                except ValueError:
                    error_payload = {}
                if error_payload.get("code") == "invalid_request_url" and version != "2022-06-28":
                    version = "2022-06-28"
                    headers = notion_headers(notion_api_key, version)
                    tried_fallback = True
                    continue
            raise RuntimeError(f"Notion API error: {detail}") from exc
        data = response.json()
        pages.extend(data.get("results", []))

        if not data.get("has_more"):
            break
        payload["start_cursor"] = data.get("next_cursor")

    return pages


def build_csv_rows(pages: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    for page in pages:
        props = page.get("properties", {})
        row: Dict[str, str] = {}
        for name, prop in props.items():
            row[name] = extract_property_value(prop)
        rows.append(row)

    return rows


def write_csv(rows: List[Dict[str, str]], filepath: str) -> str:
    if not rows:
        with open(filepath, "w", encoding="utf-8") as file:
            file.write("")
        return ""

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(filepath, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()


def build_prompt(csv_text: str, row_count: int) -> str:
    guidance = (
        """
        GUIDANCE:
        You are an expert social media analyst. Review the Instagram content data in the CSV below. 
        Your goal is to analyze the data and provide insights on content performance, trends, audience resonance, posting cadence, and experiments to try next. 

        DATA INSIGHTS:
        - Followers are not a good proxy for performance as I've been following more people.
        - This page is a mix of posts for the various jobs I do. 
        - I work as an Olympic Weightlifting Coach, so most posts revolve around this.
        - I also work as a software developer for mobile apps directed at Olympic Weightlifting Coaches and Athletes.  
             - App 1: MeetCal - MeetCal is an iOS and Android app that takes all the PDFs for USAW meets and puts them in one place. Start lists, records, standards, schedules, etc.
             - App 2: Forge Performance Journal - Forge is an iOS app that acts as a mental journal for weightlifting and powerlifting. It includes daily check-ins, post-workout reflections, post-competition reflections, mental exercises like breathing and visualization. As well as trend tracking for all the forms and the ability to connect that data with Oura and Whoop wearable health data.
        - Instagram has a new feature called Trial Reels that only show posts to non-followers. I have been posting more humor type content to this and if it does well then sharing to the main feed.

        CONTENT GOAL:
        1. Gain more athletes who are paying me for coaching.
        2. Gain more installs and subscriptions to my mobile apps.

        POSTING CADENCE:
        - Thusday: App Advertisement
        - Sunday: Athlete Highlight
        - All other days are a mix of weightlifting tips and humor

        RESPONSE:
        - Include a short executive summary and 3-5 actionable recommendations. 
        - Include a list of 10 hooks that could do well, half for the apps and half for the coaching.
        - Include any new things I should try, for example talking reels do poorly, but that's more so a result of me, not of the format.
        - Keep the response under 2500 characters.
        - Use plain text formatting for the response.
        """
    )

    return (
        f"{guidance}\n\n"
        f"Rows: {row_count}\n"
        "CSV:\n"
        f"{csv_text}"
    )


def analyze_with_openrouter(api_key: str, model: str, prompt: str) -> str:
    url = f"{OPENROUTER_API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise analyst."},
            {"role": "user", "content": prompt},
        ],
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("OpenRouter returned no choices.")

    message = choices[0].get("message", {})
    content = message.get("content")
    if not content:
        raise RuntimeError("OpenRouter response missing content.")

    return content.strip()


def post_to_slack(webhook_url: str, message: str) -> None:
    payload = {"text": message}
    response = requests.post(webhook_url, json=payload, timeout=30)
    response.raise_for_status()


def main() -> None:
    load_dotenv()

    notion_api_key = require_env("NOTION_API_KEY")
    database_id = normalize_database_id(require_env("NOTION_DATABASE_ID"))
    openrouter_api_key = require_env("OPENROUTER_API_KEY")
    slack_webhook_url = require_env("SLACK_WEBHOOK_URL")

    openrouter_model = optional_env("OPENROUTER_MODEL") or DEFAULT_MODEL

    pages = fetch_notion_pages(notion_api_key, database_id)
    rows = build_csv_rows(pages)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = f"/tmp/notion_db_{timestamp}.csv"
    csv_text = write_csv(rows, csv_path)

    if not csv_text:
        analysis = "No data found in the Notion database to analyze."
    else:
        truncated_csv = csv_text
        truncated_notice = ""

        prompt = build_prompt(truncated_csv, len(rows))
        analysis = analyze_with_openrouter(openrouter_api_key, openrouter_model, prompt)
        analysis = f"{analysis}{truncated_notice}"

    slack_message = (
        "Instagram Content Analysis (Notion DB)\n"
        f"Database rows: {len(rows)}\n"
        f"Generated: {timestamp}\n\n"
        f"{analysis}"
    )

    post_to_slack(slack_webhook_url, slack_message)
    print("Analysis sent to Slack.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
