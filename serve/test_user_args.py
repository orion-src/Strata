"""user_args on the server side: the config's "user_args" are appended to the engine arguments at every
start, after anything setup chose.  No GPU needed: python -m unittest serve.test_user_args"""
import unittest

from serve.server import engine_args


class UserArgsServer(unittest.TestCase):
    def test_absent_equals_empty(self):
        base = {"exe": "strata", "args": ["--serve", "--pack", "pack/full"], "cwd": "."}
        self.assertEqual(engine_args(base), engine_args({**base, "user_args": []}))

    def test_appended_last_blanks_dropped(self):
        base = {"exe": "strata", "args": ["--serve", "--pack", "pack/full"], "cwd": "."}
        out = engine_args({**base, "user_args": ["--conversation-cache-mib", "8192", "", "  "]})
        self.assertEqual(out[:3], base["args"])
        self.assertEqual(out[-2:], ["--conversation-cache-mib", "8192"])

    def test_non_list_ignored(self):
        base = {"exe": "strata", "args": ["--serve"], "cwd": "."}
        for bad in ("not a list", 4, True, {"--x": 1}):
            self.assertEqual(engine_args({**base, "user_args": bad}), engine_args(base))

    def test_user_choice_wins_over_setup_flags(self):
        # the engine's parser assigns in a loop (last occurrence wins), and user_args come last
        base = {"exe": "strata", "args": ["--serve"], "cwd": ".", "parallel": 4}
        out = engine_args({**base, "user_args": ["--batch", "2"]})
        self.assertEqual(out[out.index("--batch") + 1], "4")
        self.assertEqual(out[-2:], ["--batch", "2"])


if __name__ == "__main__":
    unittest.main()
