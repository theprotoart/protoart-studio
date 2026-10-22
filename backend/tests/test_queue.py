"""End-to-end test of the backend job queue using the fake engine (no torch needed)."""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "bee_backend.py"


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        env = dict(os.environ, PROTOART_HOME=self.tmp.name, BEESTUDIO_FAKE_ENGINE="1",
                   HF_HOME=os.path.join(self.tmp.name, "hf"))
        env.pop("HF_HUB_CACHE", None)
        env.pop("HUGGINGFACE_HUB_CACHE", None)
        self.p = subprocess.Popen([sys.executable, str(BACKEND)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, env=env, cwd=str(BACKEND.parent))
        self.msgs = []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            self.msgs.append(json.loads(line))

    def tearDown(self):
        self.p.stdin.close()
        self.p.wait(timeout=10)
        self.p.stdout.close()
        self.tmp.cleanup()

    def send(self, mid, cmd, **args):
        self.p.stdin.write(json.dumps({"id": mid, "cmd": cmd, "args": args}) + "\n")
        self.p.stdin.flush()

    def wait_for(self, pred, timeout=10):
        end = time.time() + timeout
        while time.time() < end:
            for m in list(self.msgs):
                if pred(m):
                    return m
            time.sleep(0.02)
        self.fail("timed out; messages: " + json.dumps(self.msgs[-10:]))

    @staticmethod
    def mine(d):
        """Folder types, leaving out the app's own starter folders (made on first run)."""
        return {k: v for k, v in d["families"].items() if not k.startswith(d["default"].rstrip("/") + "/")}

    def outcome(self, mid):
        return self.wait_for(lambda m: m.get("id") == mid and m["type"] in ("result", "error"))

    def test_cancel_one_queued_job_keeps_the_others(self):
        for i, p in [(1, "first"), (2, "second"), (3, "third")]:
            self.send(i, "generate", prompt=p)
        q = self.wait_for(lambda m: m["type"] == "queue" and [j["state"] for j in m["jobs"]] == ["running", "queued", "queued"])
        self.assertEqual([j["state"] for j in q["jobs"]], ["running", "queued", "queued"])
        self.assertEqual(q["jobs"][1]["label"], "second")
        self.send(10, "cancel_job", job_id=2)
        self.assertEqual(self.outcome(10)["data"], "removed")
        self.assertTrue(self.outcome(2).get("cancelled"))
        self.assertEqual(self.outcome(1)["type"], "result")
        r3 = self.outcome(3)
        self.assertEqual(r3["type"], "result")
        self.assertEqual(r3["data"][0]["prompt"], "third")
        self.wait_for(lambda m: m["type"] == "queue" and m["jobs"] == [])

    def test_move_a_waiting_job_up(self):
        self.send(30, "pause", on=True)
        for i, p in [(31, "first"), (32, "second"), (33, "third")]:
            self.send(i, "generate", prompt=p)
        self.wait_for(lambda m: m["type"] == "queue" and len(m["jobs"]) == 3)
        self.send(40, "queue_move", job_id=33, delta=-1)
        self.assertEqual(self.outcome(40)["data"], 1)
        q = self.wait_for(lambda m: m["type"] == "queue" and [j["label"] for j in m["jobs"]] == ["first", "third", "second"])
        self.assertTrue(q)
        self.send(41, "queue_move", job_id=33, delta=-5)  # to the front, no further
        self.assertEqual(self.outcome(41)["data"], 0)
        self.send(42, "queue_move", job_id=999, delta=1)
        self.assertIsNone(self.outcome(42)["data"])
        self.send(43, "pause", on=False)
        first = self.outcome(33)  # runs before the others now
        self.assertEqual(first["type"], "result")
        self.send(44, "cancel_all")

    def test_pause_holds_the_running_image_and_the_queue(self):
        self.send(1, "generate", prompt="a")
        self.send(2, "generate", prompt="b")
        self.wait_for(lambda m: m["type"] == "progress" and m.get("job") == 1)
        self.send(10, "pause", on=True)
        self.assertTrue(self.outcome(10)["data"]["paused"])
        self.wait_for(lambda m: m["type"] == "queue" and m.get("paused") is True)
        time.sleep(0.3)
        steps = len([m for m in self.msgs if m["type"] == "progress"])
        time.sleep(0.6)  # paused: no more steps, job 1 not finished, job 2 not started
        self.assertEqual(len([m for m in self.msgs if m["type"] == "progress"]), steps)
        self.assertFalse(any(m.get("id") in (1, 2) and m["type"] == "result" for m in self.msgs))
        self.send(11, "pause", on=False)
        self.assertFalse(self.outcome(11)["data"]["paused"])
        self.assertEqual(self.outcome(1)["type"], "result")
        self.assertEqual(self.outcome(2)["type"], "result")

    def test_pause_waits_before_the_next_job_and_stop_all_clears_it(self):
        self.send(10, "pause", on=True)
        self.outcome(10)
        self.send(1, "generate", prompt="a")
        time.sleep(0.5)
        self.assertFalse(any(m["type"] == "progress" for m in self.msgs))  # held while paused
        self.send(11, "cancel_all")
        self.assertTrue(self.outcome(1).get("cancelled"))
        self.wait_for(lambda m: m["type"] == "queue" and m["jobs"] == [] and m.get("paused") is False)
        self.send(2, "generate", prompt="b")  # not paused any more
        self.assertEqual(self.outcome(2)["type"], "result")

    def test_stop_running_job_then_continue(self):
        self.send(1, "generate", prompt="a")
        self.send(2, "generate", prompt="b")
        self.wait_for(lambda m: m["type"] == "progress" and m.get("job") == 1)
        self.send(10, "cancel_job", job_id=1)
        self.assertEqual(self.outcome(10)["data"], "stopping")
        self.assertTrue(self.outcome(1).get("cancelled"))
        self.assertEqual(self.outcome(2)["type"], "result")  # next job not affected by the stop

    def test_preload_uses_idle_time_only(self):
        self.send(1, "generate", prompt="a")
        self.wait_for(lambda m: m["type"] == "progress" and m.get("job") == 1)
        self.send(2, "preload", model="sd15")           # busy: skipped straight away
        self.assertEqual(self.outcome(2)["data"], {"skipped": "busy"})
        self.assertEqual(self.outcome(1)["type"], "result")
        self.send(3, "preload", model="sd15")           # idle: runs
        self.assertEqual(self.outcome(3)["data"], {"ready": "sd15"})

    def test_generate_stops_a_preload_of_another_model(self):
        self.send(1, "preload", model="sdxl")
        self.wait_for(lambda m: m["type"] == "queue" and m["jobs"] and m["jobs"][0]["cmd"] == "preload")
        self.send(2, "generate", prompt="b", model="sd15")
        self.assertTrue(self.outcome(1).get("cancelled"))
        self.assertEqual(self.outcome(2)["type"], "result")

    def test_generate_waits_for_a_preload_of_the_same_model(self):
        self.send(1, "preload", model="sd15")
        self.wait_for(lambda m: m["type"] == "queue" and m["jobs"] and m["jobs"][0]["cmd"] == "preload")
        self.send(2, "generate", prompt="b", model="sd15")
        self.assertEqual(self.outcome(1)["data"], {"ready": "sd15"})
        self.assertEqual(self.outcome(2)["type"], "result")

    def test_relink_a_moved_folder(self):
        old = os.path.join(self.tmp.name, "AAA")
        new = os.path.join(self.tmp.name, "drive", "AAA")
        os.makedirs(os.path.join(old, "loras"))
        self.send(1, "lora_dirs_set", dirs=[os.path.join(old, "loras")])
        self.outcome(1)
        self.send(2, "lora_dir_family", dir=os.path.join(old, "loras"), family="sd15")
        self.outcome(2)
        self.send(3, "content_mark", kind="model", name="local:" + old + "/dream.safetensors", tag="people", value=True)
        self.outcome(3)
        self.send(4, "folders_get")
        self.assertEqual(self.outcome(4)["data"], [{"path": os.path.join(old, "loras"), "found": True}])
        os.makedirs(os.path.dirname(new))
        os.rename(old, new)  # the folder moves to another drive
        self.send(5, "folders_get")
        self.assertFalse(self.outcome(5)["data"][0]["found"])
        self.send(6, "folder_relink", old=old, new=new)
        r = self.outcome(6)
        self.assertEqual(r["type"], "result", r)
        self.assertEqual(r["data"]["folders"], [{"path": os.path.join(new, "loras"), "found": True}])
        self.send(7, "lora_dirs_get")
        d = self.outcome(7)["data"]
        self.assertEqual(d["dirs"], [os.path.join(new, "loras")])
        self.assertEqual(self.mine(d), {os.path.join(new, "loras"): "sd15"})
        with open(os.path.join(self.tmp.name, "content_marks.json")) as f:
            marks = json.load(f)
        self.assertEqual(marks["people_models"], {"local:" + new + "/dream.safetensors": True})

    def test_moved_folder_keeps_everything(self):
        old = os.path.join(self.tmp.name, "AAA")
        new = os.path.join(self.tmp.name, "NewDrive", "AAA")
        for base in (old, new):
            os.makedirs(os.path.join(base, "loras_xl"))
            Path(base, "dream.safetensors").write_bytes(b"x")
        self.send(1, "lora_dirs_set", dirs=[os.path.join(old, "loras_xl")])
        self.outcome(1)
        self.send(2, "lora_dir_family", dir=os.path.join(old, "loras_xl"), family="sdxl")
        self.outcome(2)
        self.send(3, "model_dirs_set", dirs=[old])
        self.outcome(3)
        self.send(4, "content_mark", kind="model", name="local:" + old + "/dream.safetensors", tag="people", value=True)
        self.outcome(4)
        # the wrong folder is refused
        wrong = os.path.join(self.tmp.name, "Elsewhere")
        os.makedirs(wrong)
        self.send(5, "folder_relink", old=old, new=wrong)
        self.assertEqual(self.outcome(5)["type"], "error")
        self.send(6, "folder_relink", old=old, new=new)
        r = self.outcome(6)["data"]
        self.assertGreaterEqual(r["changed"], 3)
        self.assertEqual(r["folders"], [{"path": new, "found": True}])
        self.send(7, "lora_dirs_get")
        d = self.outcome(7)["data"]
        self.assertEqual(d["dirs"], [os.path.join(new, "loras_xl")])
        self.assertEqual(self.mine(d), {os.path.join(new, "loras_xl"): "sdxl"})
        marks = json.loads(Path(self.tmp.name, "content_marks.json").read_text())
        self.assertEqual(marks["people_models"], {"local:" + new + "/dream.safetensors": True})

    def test_reassigned_lora_moves_to_its_type_folder(self):
        base = Path(self.tmp.name, "L")
        (base / "sd15").mkdir(parents=True)
        (base / "mixed").mkdir()
        lora = base / "mixed" / "inky.safetensors"
        hdr = json.dumps({"lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight":
                          {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}).encode()
        lora.write_bytes(len(hdr).to_bytes(8, "little") + hdr + b"\0\0")
        (base / "mixed" / "inky.txt").write_text("inky style")
        self.send(1, "lora_dirs_set", dirs=[str(base / "mixed"), str(base / "sd15")])
        self.outcome(1)
        self.send(2, "lora_dir_family", dir=str(base / "sd15"), family="sd15")
        self.outcome(2)
        self.send(3, "lora_family", name="inky", family="sd15")
        r = self.outcome(3)["data"]
        self.assertEqual(r["moved"], {"to": str(base / "sd15"), "files": 2})
        self.assertTrue((base / "sd15" / "inky.safetensors").exists())
        self.assertTrue((base / "sd15" / "inky.txt").exists())
        self.assertFalse(lora.exists())
        self.send(4, "lora_family", name="inky", family="flux")  # no FLUX folder: relabelled only
        r = self.outcome(4)["data"]
        self.assertIn("skipped", r["moved"])
        self.assertEqual(r["families"], {"inky": "flux"})

    def test_sort_loras_into_type_folders(self):
        root = Path(self.tmp.name, "Mixed")
        (root / "Flux").mkdir(parents=True)  # an existing folder is reused

        def lora(name, keys):
            hdr = json.dumps({k: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for k in keys}).encode()
            (root / name).write_bytes(len(hdr).to_bytes(8, "little") + hdr + b"\0\0")
        lora("pencil.safetensors", ["lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight"])
        lora("xl.safetensors", ["lora_unet_input_blocks_4_1.lora_down.weight", "lora_te2_text_model.x"])
        lora("glow.safetensors", ["lora_unet_double_blocks_0_img_attn_proj.lora_down.weight"])
        lora("graffiti.safetensors", ["transformer.layers.0.attn.to_q.lora_A.weight"])
        lora("mystery.safetensors", ["foo.lora_down.weight"])
        (root / "pencil.txt").write_text("pencil sketch")
        self.send(1, "lora_dirs_set", dirs=[str(root)])
        self.outcome(1)
        self.send(2, "lora_sort", dir=str(root), dry_run=True)
        plan = self.outcome(2)["data"]
        self.assertEqual(plan["moves"], 4)
        self.assertEqual(plan["by_type"], {"sd15": 1, "sdxl": 1, "flux": 1, "ernie": 1})
        self.assertEqual(plan["unknown"], ["mystery"])
        self.assertTrue(plan["folders"]["flux"]["exists"])
        self.assertTrue(root.joinpath("pencil.safetensors").exists())  # a plan moves nothing
        self.send(3, "lora_sort", dir=str(root))
        r = self.outcome(3)["data"]
        self.assertEqual((r["moved"], r["errors"]), (4, []))
        for path in ("SD 1.5/pencil.safetensors", "SD 1.5/pencil.txt", "SDXL/xl.safetensors",
                     "Flux/glow.safetensors", "ERNIE/graffiti.safetensors", "mystery.safetensors"):
            self.assertTrue(root.joinpath(path).exists(), path)
        self.send(4, "lora_dirs_get")
        fams = self.outcome(4)["data"]["families"]
        self.assertEqual(fams[str(root / "Flux")], "flux")
        self.assertEqual(fams[str(root / "SD 1.5")], "sd15")
        self.send(5, "list_loras")
        rows = {l["name"]: l["family"] for l in self.outcome(5)["data"]}
        self.assertEqual(rows["pencil"], "sd15")
        self.assertEqual(rows["glow"], "flux")

    def test_first_run_makes_type_folders(self):
        self.send(1, "lora_dirs_get")
        d = self.outcome(1)["data"]
        self.assertEqual(sorted(d["families"].values()), ["ernie", "flux", "flux2", "sd15", "sdxl"])
        for name in ("SD 1.5", "SDXL", "FLUX", "FLUX.2", "ERNIE"):
            self.assertTrue(Path(d["default"], name).is_dir())

    def test_downloads_folder_and_move(self):
        self.send(1, "downloads_dir_get")
        d = self.outcome(1)["data"]
        self.assertFalse(d["custom"])
        self.assertTrue(d["path"].startswith(self.tmp.name))
        old = Path(self.tmp.name) / "old_cache"
        (old / "models--a--b" / "blobs").mkdir(parents=True)
        (old / "models--a--b" / "blobs" / "x").write_bytes(b"0" * 1000)
        (old / "something-else").mkdir()
        new = Path(self.tmp.name) / "ext" / "Downloaded models"
        self.send(2, "downloads_dir_set", path=str(new))
        self.assertTrue(self.outcome(2)["data"]["restart"])
        self.assertTrue((Path(self.tmp.name) / "downloads_location.json").exists())
        self.send(3, "downloads_move", **{"from": str(old), "to": str(new)})
        r = self.outcome(3)
        self.assertEqual(r["type"], "result", r)
        self.assertEqual(r["data"]["moved"], 1)
        self.assertTrue((new / "models--a--b" / "blobs" / "x").exists())
        self.assertFalse((old / "models--a--b").exists())
        self.assertTrue((old / "something-else").exists())  # only model downloads are moved
        self.send(4, "downloads_dir_set", path=None)
        self.assertEqual(self.outcome(4)["type"], "result")

    def test_cancel_all(self):
        for i in (1, 2, 3):
            self.send(i, "generate", prompt=str(i))
        self.wait_for(lambda m: m["type"] == "progress")
        self.send(10, "cancel_all")
        for i in (1, 2, 3):
            self.assertTrue(self.outcome(i).get("cancelled"))

    def test_quick_commands_answer_while_generating(self):
        self.send(1, "generate", prompt="slow")
        self.wait_for(lambda m: m["type"] == "progress")
        self.send(2, "presets_list")
        r = self.outcome(2)
        self.assertEqual(r["type"], "result")
        self.assertNotIn("result", [m["type"] for m in self.msgs if m.get("id") == 1])  # still running



class DownloadProblemTests(unittest.TestCase):
    def test_offline_download_explained(self):
        import bee_backend
        try:
            try:
                raise ConnectionError("curl: (6) Could not resolve host: huggingface.co")
            except ConnectionError as inner:
                raise OSError("lllyasviel/control_v11f1p_sd15_depth does not appear to have a file named config.json.") from inner
        except OSError as e:
            msg = bee_backend.download_problem(e)
        self.assertIn("couldn't reach huggingface.co", msg)
        self.assertIsNone(bee_backend.download_problem(ValueError("something else")))

    def test_missing_upstream_file_is_not_called_offline(self):
        import bee_backend
        try:
            try:
                raise RuntimeError("Client error '404 Not Found' for url 'https://huggingface.co/x/resolve/abc/model.safetensors'")
            except RuntimeError as inner:
                raise OSError("x does not appear to have a file named diffusion_pytorch_model.safetensors.") from inner
        except OSError as e:
            msg = bee_backend.download_problem(e)
        self.assertIn("wasn't found on Hugging Face", msg)
        self.assertNotIn("couldn't reach", msg)


if __name__ == "__main__":
    unittest.main()
