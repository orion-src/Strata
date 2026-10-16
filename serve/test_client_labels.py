"""Client labels on the parallel slots: a request can name itself - the OpenAI body's `user` field
(chat-completions and responses) or an `X-Client-Label` header (any route) - and /metrics then carries that
label on the busy slot (slots_view), sanitized to one line of at most 64 characters; the Monitor's slot card
shows it ("Model state 0 - hermes-sub").  A request with no label leaves the slot payload exactly as it was.

The fake batch engine of serve/test_parallel runs behind the real StrataEngine and Service, so this is the
whole HTTP path without a GPU.

    python -m unittest serve.test_client_labels -v
"""
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import (CLIENT_LABEL_MAX, ByteTokenizer, Service, StrataEngine, sanitize_client_label,
                          serve)
from serve.test_parallel import FAKE_BATCH


class Sanitize(unittest.TestCase):
    def test_control_chars_become_spaces_and_collapse(self):
        self.assertEqual(sanitize_client_label("hermes\x00\t\nsub"), "hermes sub")
        self.assertEqual(sanitize_client_label("a\xa0b\u200bc"), "a b c")   # NBSP splits, a zero-width char goes

    def test_capped_at_64(self):
        self.assertEqual(sanitize_client_label("x" * 200), "x" * CLIENT_LABEL_MAX)
        self.assertEqual(sanitize_client_label("y " * 100), "y " * 31 + "y")   # capped, then the cut space goes

    def test_nothing_left_is_no_label(self):
        for raw in ("", "   ", "\x00\x01", None, 5, {"user": "x"}):
            self.assertIsNone(sanitize_client_label(raw))


class SlotLabels(unittest.TestCase):
    """The real Service over HTTP, the fake batch engine behind it: the label reaches /metrics slots from
    the body's `user` or the header, is sanitized, and its absence changes nothing."""

    def start(self, slots=2):
        import serve.server as server
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / "fake_strata.py"
        script.write_text(FAKE_BATCH, encoding="utf-8")
        self.log = Path(self.tmp.name) / "requests.log"
        real = server.subprocess.Popen
        extra = ["--batch", str(slots), "--log", str(self.log)]
        with mock.patch.object(server.subprocess, "Popen",
                               lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
            self.engine = StrataEngine("strata", extra)
        tok = ByteTokenizer()
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        for attr in ("httpd", "engine", "tmp"):
            obj = getattr(self, attr, None)
            if obj is None:
                continue
            if attr == "httpd":
                obj.shutdown()
                obj.server_close()
            elif attr == "engine":
                obj.unload()
            else:
                obj.cleanup()

    def start_long(self, body, headers=None):
        """POST a LONGREPLY chat request (the fake engine answers it with ~55 tokens, about a second of
        decode) on its own thread; return the thread."""
        body = {"messages": [{"role": "user", "content": "hi LONGREPLY"}], "max_tokens": 64,
                "reasoning_effort": "none", **body}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **(headers or {})})
        t = threading.Thread(target=lambda: urllib.request.urlopen(req, timeout=60).read(), daemon=True)
        t.start()
        return t

    def run_pair(self, body_b, headers_b=None):
        """A first (unlabeled) request takes the solo path while it is alone; the request under test arrives
        while it decodes, so it goes into a batch slot - the _two_at_once pattern of serve/test_parallel."""
        ta = self.start_long({})
        time.sleep(0.15)
        tb = self.start_long(body_b, headers_b)
        return ta, tb

    def get_slots(self):
        with urllib.request.urlopen(self.base + "/metrics", timeout=10) as r:
            return json.loads(r.read().decode())["live"].get("slots", [])

    def wait_slots(self, want, timeout=10):
        """Poll /metrics until `want(busy_slots)` holds; return the busy slots then."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            busy = [s for s in self.get_slots() if s["state"] != "idle"]
            if want(busy):
                return busy
            time.sleep(0.02)
        self.fail("the batch slots never showed what the test asked for")

    def test_user_field_names_the_slot(self):
        self.start(2)
        ta, tb = self.run_pair({"user": "hermes-sub"})
        try:
            labeled = self.wait_slots(lambda busy: any(s.get("label") for s in busy))
            self.assertEqual([s["label"] for s in labeled if s.get("label")], ["hermes-sub"])
        finally:
            ta.join(60); tb.join(60)
        self.assertFalse(ta.is_alive() or tb.is_alive())
        # the label belongs to the request: the idle slots after it say nothing
        time.sleep(0.3)
        for s in self.get_slots():
            self.assertNotIn("label", s)

    def test_header_names_the_slot(self):
        self.start(2)
        ta, tb = self.run_pair({}, headers_b={"X-Client-Label": "cli-a"})
        try:
            labeled = self.wait_slots(lambda busy: any(s.get("label") for s in busy))
            self.assertEqual([s["label"] for s in labeled if s.get("label")], ["cli-a"])
        finally:
            ta.join(60); tb.join(60)

    def test_the_header_wins_over_the_body_field(self):
        self.start(2)
        ta, tb = self.run_pair({"user": "from-body"}, headers_b={"X-Client-Label": "from-header"})
        try:
            labeled = self.wait_slots(lambda busy: any(s.get("label") for s in busy))
            self.assertEqual([s["label"] for s in labeled if s.get("label")], ["from-header"])
        finally:
            ta.join(60); tb.join(60)

    def test_a_labeled_slot_hands_on_clean_to_the_next_tenant(self):
        # the live dict is fresh per admission: a label never leaks onto the next occupant of the slot
        self.start(2)
        ta, tb = self.run_pair({"user": "first-tenant"})
        self.wait_slots(lambda busy: any(s.get("label") for s in busy))
        ta.join(60); tb.join(60)
        ta, tb = self.run_pair({})                       # unlabeled; the slots may well be reused
        try:
            busy = self.wait_slots(lambda b: len(b) >= 2)
            self.assertFalse([s for s in busy if "label" in s])
        finally:
            ta.join(60); tb.join(60)

    def test_oversized_and_control_chars_are_sanitized(self):
        self.start(2)
        messy = "hermes\x00\t\nsub " + "x" * 200
        ta, tb = self.run_pair({"user": messy})
        try:
            labeled = self.wait_slots(lambda busy: any(s.get("label") for s in busy))
            label = next(s["label"] for s in labeled if s.get("label"))
        finally:
            ta.join(60); tb.join(60)
        self.assertEqual(label, ("hermes sub " + "x" * 200)[:CLIENT_LABEL_MAX])
        self.assertLessEqual(len(label), CLIENT_LABEL_MAX)
        self.assertFalse(any(ord(ch) < 32 or ch == "\x7f" for ch in label))

    def test_no_label_leaves_the_payload_as_before(self):
        self.start(2)
        ta, tb = self.run_pair({})
        try:
            busy = self.wait_slots(lambda b: len(b) >= 2)     # both requests are in slots at once
            for s in busy:
                self.assertNotIn("label", s)
                # the busy-slot keys are exactly what slots_view sent before labels existed
                self.assertEqual(set(s), {"slot", "state", "prompt_tokens", "generated", "elapsed_s", "tok_s"})
        finally:
            ta.join(60); tb.join(60)
        time.sleep(0.3)
        for s in self.get_slots():
            self.assertEqual(s["state"], "idle")
            self.assertEqual(set(s), {"slot", "state", "held_tokens"})   # the idle payload, key for key


if __name__ == "__main__":
    unittest.main()
