"""Serve the local LLM-SHIELD demonstration UI and its JSON API."""

from __future__ import annotations

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from app.web_demo import DemoPipelineError, get_demo_config, run_receiver, run_sender

logger = logging.getLogger("app.web_server")
WEB_ROOT = Path(__file__).resolve().parent / "web"
MAX_REQUEST_BYTES = 2 * 1024 * 1024


class DemoRequestHandler(BaseHTTPRequestHandler):
    server_version = "LLMShieldDemo/1.0"

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/config":
            self._send_json(200, get_demo_config())
            return

        static_files = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/static/styles.css": ("styles.css", "text/css; charset=utf-8"),
            "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
        }
        file_entry = static_files.get(path)
        if file_entry is None:
            self._send_json(404, {"error": "Not found."})
            return

        filename, content_type = file_entry
        body = (WEB_ROOT / filename).read_bytes()
        self.send_response(200)
        self._send_headers(content_type, len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path not in {"/api/sender", "/api/sender/stream", "/api/receiver"}:
            self._send_json(404, {"error": "Not found."})
            return

        try:
            payload = self._read_json()
            if path == "/api/sender/stream":
                self._stream_sender(payload)
                return
            if path == "/api/sender":
                response = run_sender(
                    message=payload.get("message"),
                    password=payload.get("password"),
                    topic=payload.get("topic"),
                    temperature=payload.get("temperature", 0.7),
                    top_k=payload.get("top_k", 20),
                )
            else:
                response = run_receiver(
                    carrier_text=payload.get("carrier_text"),
                    password=payload.get("password"),
                    salt_hex=payload.get("salt_hex"),
                    nonce_hex=payload.get("nonce_hex"),
                    mapped_length=payload.get("mapped_length"),
                )
        except DemoPipelineError as error:
            self._send_json(
                400,
                {"error": str(error), "stage": error.stage},
            )
            return
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            self._send_json(400, {"error": str(error), "stage": "input"})
            return
        except Exception as error:
            logger.error("Demo request failed (%s).", type(error).__name__)
            self._send_json(
                500,
                {"error": "The demo pipeline failed unexpectedly. See the server output for details."},
            )
            return

        self._send_json(200, response)

    def _stream_sender(self, payload: dict[str, object]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def send_event(event: dict[str, object]) -> None:
            body = json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n"
            self.wfile.write(body)
            self.wfile.flush()

        try:
            response = run_sender(
                message=payload.get("message"),
                password=payload.get("password"),
                topic=payload.get("topic"),
                temperature=payload.get("temperature", 0.7),
                top_k=payload.get("top_k", 20),
                on_stage=lambda key, data: send_event(
                    {"type": "stage", "key": key, "data": data}
                ),
            )
            send_event({"type": "complete", "data": response})
        except DemoPipelineError as error:
            send_event({"type": "error", "error": str(error), "stage": error.stage})
        except Exception as error:
            logger.error("Streaming demo request failed (%s).", type(error).__name__)
            send_event(
                {
                    "type": "error",
                    "error": (
                        "The demo pipeline failed unexpectedly. "
                        "See the server output for details."
                    ),
                }
            )

    def _read_json(self) -> dict[str, object]:
        content_length = self.headers.get("Content-Length")
        try:
            body_length = int(content_length or "")
        except ValueError as error:
            raise ValueError("A valid Content-Length header is required.") from error
        if body_length < 0 or body_length > MAX_REQUEST_BYTES:
            raise ValueError(f"Request body must not exceed {MAX_REQUEST_BYTES} bytes.")
        raw_body = self.rfile.read(body_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except UnicodeDecodeError:
            raise ValueError("Request body must be valid UTF-8 JSON.") from None
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def _send_json(self, status: int, value: object) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._send_headers("application/json; charset=utf-8", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _send_headers(self, content_type: str, content_length: int) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self' data:; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'",
        )

    def log_message(self, format: str, *args: object) -> None:
        # Request bodies (including the password) are never written to access logs.
        logger.info("%s - %s", self.address_string(), format % args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (localhost by default)")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port")
    args = parser.parse_args()

    if args.host != "127.0.0.1":
        parser.error("The demo server must bind to 127.0.0.1.")
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535.")

    server = ThreadingHTTPServer((args.host, args.port), DemoRequestHandler)
    print(f"LLM-SHIELD demo UI: http://127.0.0.1:{args.port}")
    print("Localhost only. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping LLM-SHIELD demo UI.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
