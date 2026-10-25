import json
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

from app import web_server


def test_sender_stream_endpoint_sends_stage_events_before_completion(monkeypatch):
    def fake_run_sender(**kwargs):
        stage = {"status": "success", "output": "AB"}
        kwargs["on_stage"]("mapping", stage)
        kwargs["on_stage"](
            "embedding",
            {
                "status": "running",
                "message": "Generating candidate for character 1/36.",
                "completed_characters": 0,
                "total_characters": 36,
            },
        )
        time.sleep(0.1)
        kwargs["on_stage"]("carrier", {"status": "success", "text": "cover text"})
        return {"mode": "PSK", "session": {}, "stages": {"mapping": stage}}

    monkeypatch.setattr(web_server, "run_sender", fake_run_sender)
    server = ThreadingHTTPServer(("127.0.0.1", 0), web_server.DemoRequestHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/api/sender/stream",
            data=json.dumps({"message": "hello"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            first_event = json.loads(response.readline())
            second_event = json.loads(response.readline())
            third_event = json.loads(response.readline())
            remaining_events = [json.loads(line) for line in response.read().splitlines()]

        assert first_event["type"] == "stage"
        assert first_event["key"] == "mapping"
        assert first_event["data"]["output"] == "AB"
        assert second_event["type"] == "stage"
        assert second_event["key"] == "embedding"
        assert second_event["data"]["message"] == "Generating candidate for character 1/36."
        assert third_event["type"] == "stage"
        assert third_event["key"] == "carrier"
        assert remaining_events[0]["type"] == "complete"
        assert remaining_events[0]["data"]["stages"]["mapping"]["output"] == "AB"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_sender_stream_sends_error_without_completion(monkeypatch):
    def fake_run_sender(**kwargs):
        kwargs["on_stage"](
            "embedding",
            {
                "status": "running",
                "message": "Retry 1/12 for character 1/36.",
                "completed_characters": 0,
                "total_characters": 36,
            },
        )
        raise web_server.DemoPipelineError("embedding", "LLM generation failed.")

    monkeypatch.setattr(web_server, "run_sender", fake_run_sender)
    server = ThreadingHTTPServer(("127.0.0.1", 0), web_server.DemoRequestHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/api/sender/stream",
            data=json.dumps({"message": "hello"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            events = [json.loads(line) for line in response.read().splitlines()]

        assert [event["type"] for event in events] == ["stage", "error"]
        assert events[-1]["stage"] == "embedding"
        assert events[-1]["error"] == "LLM generation failed."
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_ecdhe_handshake_stream_reports_real_stages_without_key_bytes():
    server = ThreadingHTTPServer(("127.0.0.1", 0), web_server.DemoRequestHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/api/ecdhe/handshake/stream",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            events = [json.loads(line) for line in response.read().splitlines()]

        stages = [event for event in events if event["type"] == "stage"]
        assert [event["key"] for event in stages if event["data"]["status"] == "success"] == [
            "ecdhe_key_pair_generation",
            "ecdhe_public_key_exchange",
            "ecdhe_shared_secret",
            "ecdhe_key_derivation",
        ]
        assert events[-1]["type"] == "complete"
        assert events[-1]["data"]["mode"] == "ECDHE"
        assert events[-1]["data"]["session_id"]
        assert all(
            not any(
                key in event["data"]
                for key in ("private_key", "shared_secret", "dk1", "dk2")
            )
            for event in stages
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
