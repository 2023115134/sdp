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


def test_ecdhe_handshake_stream_forwards_carrier_events_in_order_without_key_bytes(
    monkeypatch,
):
    received_topics = []

    def fake_create_ecdhe_session(*, topic, on_stage):
        received_topics.append(topic)
        on_stage(
            "ecdhe_key_pair_generation",
            {"status": "success", "algorithm": "X25519"},
        )
        on_stage("ecdhe_public_key_exchange", {"status": "running"})
        for round_number in range(1, 11):
            carrier_data = {
                "round_number": round_number,
                "total_rounds": 10,
                "participant": "Sender" if round_number % 2 else "Receiver",
                "direction": "Sender → Receiver" if round_number % 2 else "Receiver → Sender",
                "purpose": "Public-key transmission",
            }
            on_stage("ecdhe_carrier", {**carrier_data, "status": "generating"})
            on_stage(
                "ecdhe_carrier_progress",
                {
                    **carrier_data,
                    "status": "generating",
                    "progress": {
                        "current_stage": "Generating candidate",
                        "current_character": 1,
                        "completed_characters": 0,
                        "total_characters": 28,
                        "target_position": 42,
                    },
                },
            )
            on_stage(
                "ecdhe_carrier",
                {
                    **carrier_data,
                    "status": "delivered",
                    "carrier_text": f"actual carrier {round_number}",
                },
            )
            on_stage("ecdhe_carrier", {**carrier_data, "status": "received"})
            on_stage("ecdhe_carrier", {**carrier_data, "status": "verified"})
        on_stage("ecdhe_key_confirmation", {"status": "success"})
        on_stage("ecdhe_key_derivation", {"status": "success"})
        return {"mode": "ECDHE", "session_id": "test-session"}

    monkeypatch.setattr(web_server, "create_ecdhe_session", fake_create_ecdhe_session)
    server = ThreadingHTTPServer(("127.0.0.1", 0), web_server.DemoRequestHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/api/ecdhe/handshake/stream",
            data=json.dumps({"topic": "selected carrier topic"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            events = [json.loads(line) for line in response.read().splitlines()]

        stages = [event for event in events if event["type"] == "stage"]
        carrier_stages = [event for event in stages if event["key"] == "ecdhe_carrier"]
        progress_stages = [
            event for event in stages if event["key"] == "ecdhe_carrier_progress"
        ]
        assert received_topics == ["selected carrier topic"]
        assert len(carrier_stages) == 40
        assert len(progress_stages) == 10
        assert progress_stages[0]["data"]["progress"]["target_position"] == 42
        streamed_carriers = [
            event["key"]
            for event in stages
            if event["key"] in {"ecdhe_carrier", "ecdhe_carrier_progress"}
        ]
        assert streamed_carriers[:5] == [
            "ecdhe_carrier",
            "ecdhe_carrier_progress",
            "ecdhe_carrier",
            "ecdhe_carrier",
            "ecdhe_carrier",
        ]
        assert [event["data"]["status"] for event in carrier_stages] == [
            status
            for _ in range(10)
            for status in ("generating", "delivered", "received", "verified")
        ]
        assert [event["data"]["round_number"] for event in carrier_stages[::4]] == list(range(1, 11))
        assert [event["data"]["carrier_text"] for event in carrier_stages if event["data"]["status"] == "delivered"] == [
            f"actual carrier {round_number}" for round_number in range(1, 11)
        ]
        assert events[-1]["type"] == "complete"
        assert events[-1]["data"]["mode"] == "ECDHE"
        assert events[-1]["data"]["session_id"]
        assert all(
            not any(
                key in event.get("data", {})
                for key in ("private_key", "shared_secret", "dk1", "dk2")
            )
            for event in events
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_ecdhe_handshake_stream_preserves_round_failure_details(monkeypatch):
    def fail_during_exchange(*, topic, on_stage):
        failure = {
            "round_number": 8,
            "total_rounds": 10,
            "participant": "Receiver",
            "direction": "Receiver → Sender",
            "purpose": "Public-key transmission",
            "status": "failed",
            "error": "Carrier verification failed at exchange 8/10 (ValueError).",
        }
        on_stage("ecdhe_carrier", failure)
        raise web_server.DemoPipelineError("key_establishment", failure["error"])

    monkeypatch.setattr(web_server, "create_ecdhe_session", fail_during_exchange)
    server = ThreadingHTTPServer(("127.0.0.1", 0), web_server.DemoRequestHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/api/ecdhe/handshake/stream",
            data=json.dumps({"topic": "test topic"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            events = [json.loads(line) for line in response.read().splitlines()]

        assert [event["type"] for event in events] == ["stage", "error"]
        assert events[0]["data"]["round_number"] == 8
        assert events[0]["data"]["status"] == "failed"
        assert events[-1]["stage"] == "key_establishment"
        assert events[-1]["error"] == events[0]["data"]["error"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
