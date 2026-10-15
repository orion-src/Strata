"""user_args: engine options of the user's own that survive setup (the conversation cache, for example).
Setup side only here (no imports from serve/, so this runs on a bare PC before pip - AGENTS.md);
the server side is serve/test_user_args.py.  No GPU or download needed: python tools/test_setup_user_args.py"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import setup as S  # noqa: E402


class UserArgs(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(S.user_args_list("--conversation-cache-mib 8192 --conversation-cache-slots 4"),
                         ["--conversation-cache-mib", "8192", "--conversation-cache-slots", "4"])
        self.assertEqual(S.user_args_list(""), [])

    def test_parse_keeps_windows_paths(self):
        # posix=False: cmd hands us "--native D:\pack" and the backslash must survive (posix mode eats it);
        # a path quoted inside the string keeps its spaces
        self.assertEqual(S.user_args_list(r'--native D:\pack'), ["--native", r"D:\pack"])
        self.assertEqual(S.user_args_list(r'--native "C:\Program Files\Strata" --conv D:\pack'),
                         ["--native", r"C:\Program Files\Strata", "--conv", r"D:\pack"])

    def test_key_survives_setup(self):
        # not a setup-written key, so carry_over keeps it (the #629 rule: what setup does not write, it keeps)
        self.assertNotIn("user_args", S.SETUP_KEYS)
        old = {"args": ["--serve"], "user_args": ["--conversation-cache-mib", "8192"]}
        cfg = {"args": ["--serve", "--batch", "4"]}
        kept = S.carry_over(old, cfg)
        self.assertEqual(cfg["user_args"], ["--conversation-cache-mib", "8192"])
        self.assertIn("user_args", kept)

    def test_removal_survives_later_runs(self):
        # --user-args "" writes []; carry_over skips keys already in the new cfg, so the removal sticks
        old = {"args": ["--serve"], "user_args": ["--conversation-cache-mib", "8192"]}
        cfg = {"args": ["--serve"], "user_args": []}
        S.carry_over(old, cfg)
        self.assertEqual(cfg["user_args"], [])


if __name__ == "__main__":
    unittest.main()
