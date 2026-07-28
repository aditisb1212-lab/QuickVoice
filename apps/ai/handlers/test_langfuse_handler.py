import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from handlers import langfuse_handler


class LangfuseHandlerDisabledTests(unittest.TestCase):
    """
    With no keys configured (the default in CI / a fresh clone), every
    public function must be a safe no-op: no exceptions, no network calls.
    """

    def setUp(self):
        langfuse_handler._client = None
        langfuse_handler._client_checked = False
        self._env_backup = {
            key: os.environ.pop(key, None)
            for key in ("LANGFUSE_ENABLED", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")
        }

    def tearDown(self):
        for key, value in self._env_backup.items():
            if value is not None:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)
        langfuse_handler._client = None
        langfuse_handler._client_checked = False

    def test_get_client_is_none_without_keys(self):
        self.assertIsNone(langfuse_handler.get_client())

    def test_log_call_trace_is_a_no_op_without_keys(self):
        # Should not raise even with a realistic-looking payload.
        langfuse_handler.log_call_trace(
            {
                "callId": "call_123",
                "agentId": "agent_1",
                "organizationId": "org_1",
                "userId": "user_1",
                "provider": "TWILIO",
                "direction": "inbound",
                "durationSeconds": 42,
                "transcripts": [
                    {"messageId": "m1", "role": "user", "message": "hi", "timestamp": "2026-01-01T00:00:00Z"},
                    {"messageId": "m2", "role": "agent", "message": "hello", "timestamp": "2026-01-01T00:00:01Z"},
                ],
            }
        )

    def test_log_retrieval_span_is_a_no_op_without_keys(self):
        langfuse_handler.log_retrieval_span(
            agent_id="agent_1",
            query="what are your hours?",
            status="hit",
            matches=3,
            latency_ms=120,
        )

    def test_explicit_disable_wins_even_with_keys_present(self):
        os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-test"
        os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-test"
        os.environ["LANGFUSE_ENABLED"] = "false"
        self.assertIsNone(langfuse_handler.get_client())


class LangfuseHandlerEvaluatorTests(unittest.TestCase):
    def test_has_transcript(self):
        self.assertEqual(langfuse_handler._score_has_transcript([]), 0.0)
        self.assertEqual(langfuse_handler._score_has_transcript([{"role": "agent"}]), 1.0)

    def test_agent_responded(self):
        self.assertEqual(langfuse_handler._score_agent_responded([{"role": "user"}]), 0.0)
        self.assertEqual(
            langfuse_handler._score_agent_responded([{"role": "user"}, {"role": "agent"}]),
            1.0,
        )

    def test_conversation_balance(self):
        self.assertEqual(langfuse_handler._score_conversation_balance([{"role": "agent"}]), 0.0)
        self.assertEqual(
            langfuse_handler._score_conversation_balance([{"role": "user"}, {"role": "agent"}]),
            1.0,
        )

    def test_first_and_last_message_helpers(self):
        transcript = [
            {"role": "user", "message": "first user turn"},
            {"role": "agent", "message": "first agent reply"},
            {"role": "user", "message": "second user turn"},
            {"role": "agent", "message": "final agent reply"},
        ]
        self.assertEqual(langfuse_handler._first_message(transcript, "user"), "first user turn")
        self.assertEqual(langfuse_handler._last_message(transcript, "agent"), "final agent reply")


if __name__ == "__main__":
    unittest.main()
