"""Image library and presets, stored in a single SQLite file.

Pure Python (no torch), so it can be unit-tested anywhere.
"""
import json
import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path

import nsfw as content

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    created_at REAL NOT NULL,
    prompt TEXT NOT NULL DEFAULT '',
    negative_prompt TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    sampler TEXT NOT NULL DEFAULT '',
    steps INTEGER,
    cfg REAL,
    seed INTEGER,
    width INTEGER,
    height INTEGER,
    settings_json TEXT NOT NULL DEFAULT '{}',
    starred INTEGER NOT NULL DEFAULT 0,
    parent_id INTEGER,
    kind TEXT NOT NULL DEFAULT 'generated'
);
CREATE TABLE IF NOT EXISTS image_loras (
    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    lora TEXT NOT NULL,
    weight REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_image_loras_lora ON image_loras(lora);
CREATE INDEX IF NOT EXISTS idx_images_starred ON images(starred);
CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS presets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL,
    settings_json TEXT NOT NULL
);
"""

# Settings that make sense to store in a preset (prompt is optional).
PRESET_KEYS = (
    "model", "sampler", "steps", "cfg", "width", "height", "loras",
    "negative_prompt", "prompt", "use_pe", "num_images", "seed", "restore_faces", "face_strength", "face_model", "face_fidelity",
    "enhance_prompt", "strength",
)


class Library:
    def __init__(self, db_path):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(SCHEMA)
        self._migrate()
        self._db.commit()

    def _migrate(self):
        cols = {r["name"] for r in self._db.execute("PRAGMA table_info(images)")}
        if "parent_id" not in cols:
            self._db.execute("ALTER TABLE images ADD COLUMN parent_id INTEGER")
        if "kind" not in cols:
            self._db.execute("ALTER TABLE images ADD COLUMN kind TEXT NOT NULL DEFAULT 'generated'")
        # content filter: checker score (NULL = not checked yet), prompt words, and your own mark (NULL = auto)
        if "nsfw_score" not in cols:
            self._db.execute("ALTER TABLE images ADD COLUMN nsfw_score REAL")
        if "nsfw_prompt" not in cols:
            self._db.execute("ALTER TABLE images ADD COLUMN nsfw_prompt INTEGER NOT NULL DEFAULT 0")
            rows = self._db.execute("SELECT id, prompt FROM images").fetchall()
            self._db.executemany("UPDATE images SET nsfw_prompt=1 WHERE id=?",
                                 [(r["id"],) for r in rows if content.prompt_flag(r["prompt"])])
        if "nsfw_mark" not in cols:
            self._db.execute("ALTER TABLE images ADD COLUMN nsfw_mark INTEGER")
        # Hide People: the prompt describes a person, and faces found in the picture (NULL = not checked yet)
        if "people_prompt" not in cols:
            self._db.execute("ALTER TABLE images ADD COLUMN people_prompt INTEGER NOT NULL DEFAULT 0")
            rows = self._db.execute("SELECT id, prompt FROM images").fetchall()
            self._db.executemany("UPDATE images SET people_prompt=1 WHERE id=?",
                                 [(r["id"],) for r in rows if content.people_flag(r["prompt"])])
        if "faces" not in cols:
            self._db.execute("ALTER TABLE images ADD COLUMN faces INTEGER")

    # ---------- images ----------
    def add_image(self, path, settings, parent_id=None, kind="generated"):
        """kind: 'generated', 'faces' (RestoreFormer) or 'enhanced' (Finegrain upscale)."""
        loras = settings.get("loras") or []
        with self._lock:
            cur = self._db.execute(
                """INSERT INTO images (path, created_at, prompt, negative_prompt, model, sampler,
                   steps, cfg, seed, width, height, settings_json, parent_id, kind, nsfw_prompt, people_prompt)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    str(path), time.time(), settings.get("prompt", ""),
                    settings.get("negative_prompt", ""), settings.get("model", ""),
                    settings.get("sampler", ""), settings.get("steps"), settings.get("cfg"),
                    settings.get("seed"), settings.get("width"), settings.get("height"),
                    json.dumps(settings), parent_id, kind,
                    1 if (content.prompt_flag(settings.get("prompt"))
                          or content.prompt_flag(settings.get("original_prompt"))) else 0,
                    1 if (content.people_flag(settings.get("prompt"))
                          or content.people_flag(settings.get("original_prompt"))) else 0,
                ),
            )
            image_id = cur.lastrowid
            self._db.executemany(
                "INSERT INTO image_loras (image_id, lora, weight) VALUES (?,?,?)",
                [(image_id, l["name"], float(l.get("weight", 1.0))) for l in loras],
            )
            self._db.commit()
        return self.get_image(image_id)

    def get_image(self, image_id):
        with self._lock:
            row = self._db.execute(f"SELECT i.*, {self.NSFW_SQL} AS nsfw FROM images i WHERE i.id=?",
                                   (image_id,)).fetchone()
            if row is None:
                return None
            return self._row_to_dict(row)

    def set_starred(self, image_id, starred):
        with self._lock:
            self._db.execute("UPDATE images SET starred=? WHERE id=?", (1 if starred else 0, image_id))
            self._db.commit()
        return self.get_image(image_id)

    # SQL for "this picture counts as NSFW": your mark wins, else the checker score or the prompt words.
    NSFW_SQL = ("COALESCE(i.nsfw_mark, CASE WHEN i.nsfw_prompt = 1 OR i.nsfw_score >= %s THEN 1 ELSE 0 END)"
                % content.THRESHOLD)

    def set_nsfw_mark(self, ids, mark):
        """mark: True (NSFW), False (fine) or None (let the app decide)."""
        ids = [int(i) for i in ids]
        if not ids:
            return []
        q = ",".join("?" * len(ids))
        with self._lock:
            self._db.execute(f"UPDATE images SET nsfw_mark=? WHERE id IN ({q})",
                             [None if mark is None else (1 if mark else 0)] + ids)
            self._db.commit()
        return [self.get_image(i) for i in ids]

    def set_nsfw_score(self, image_id, score):
        with self._lock:
            self._db.execute("UPDATE images SET nsfw_score=? WHERE id=?", (score, int(image_id)))
            self._db.commit()

    PEOPLE_SQL = "(COALESCE(i.people_prompt, 0) = 1 OR COALESCE(i.faces, 0) > 0)"

    def set_faces(self, image_id, n):
        with self._lock:
            self._db.execute("UPDATE images SET faces=? WHERE id=?", (int(n), int(image_id)))
            self._db.commit()

    def faces_unchecked(self, limit=100000):
        """Pictures not checked for faces yet, whose prompt doesn't already say there's a person: [(id, path)]."""
        with self._lock:
            rows = self._db.execute("SELECT id, path FROM images WHERE faces IS NULL AND COALESCE(people_prompt, 0) = 0 "
                                    "ORDER BY created_at DESC LIMIT ?", (int(limit),)).fetchall()
            return [(r["id"], r["path"]) for r in rows]

    def unchecked(self, limit=100000):
        """Pictures the checker hasn't looked at yet: [(id, path)], newest first."""
        with self._lock:
            rows = self._db.execute("SELECT id, path FROM images WHERE nsfw_score IS NULL "
                                    "ORDER BY created_at DESC LIMIT ?", (int(limit),)).fetchall()
            return [(r["id"], r["path"]) for r in rows]

    def nsfw_counts(self):
        with self._lock:
            r = self._db.execute(f"SELECT COUNT(*), SUM({self.NSFW_SQL}), SUM(i.nsfw_score IS NULL) "
                                 "FROM images i").fetchone()
            return {"total": r[0] or 0, "nsfw": r[1] or 0, "unchecked": r[2] or 0}

    def query(self, text=None, loras=None, starred_only=False, model=None, kind=None, limit=200, offset=0, seed=None,
              hide_nsfw=False, hide_people=False):
        """Search images.

        loras: list of LoRA names; an image matches only if it used ALL of them.
        text: matched against prompt and negative prompt (case-insensitive substring); a plain number also
              matches images made with that seed.
        seed: only images made with exactly this seed.
        hide_nsfw: leave out pictures that count as NSFW (see NSFW_SQL).
        hide_people: leave out pictures of people (their prompt says so, or faces were found in them).
        """
        where, params = [], []
        if seed is not None and str(seed).strip() != "":
            try:
                seed = int(str(seed).strip())
            except ValueError:
                return []  # not a number: nothing can match
            where.append("i.seed = ?")
            params.append(seed)
        if starred_only:
            where.append("i.starred = 1")
        if hide_nsfw:
            where.append(self.NSFW_SQL + " = 0")
        if hide_people:
            where.append("NOT " + self.PEOPLE_SQL)
        if model:
            where.append("i.model = ?")
            params.append(model)
        if kind:
            where.append("i.kind = ?")
            params.append(kind)
        if text:
            pat = "%" + text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            if text.strip().lstrip("-").isdigit():
                where.append("(i.prompt LIKE ? ESCAPE '\\' OR i.negative_prompt LIKE ? ESCAPE '\\' OR i.seed = ?)")
                params += [pat, pat, int(text.strip())]
            else:
                where.append("(i.prompt LIKE ? ESCAPE '\\' OR i.negative_prompt LIKE ? ESCAPE '\\')")
                params += [pat, pat]
        for name in loras or []:
            where.append("EXISTS (SELECT 1 FROM image_loras l WHERE l.image_id = i.id AND l.lora = ?)")
            params.append(name)
        sql = f"SELECT i.*, {self.NSFW_SQL} AS nsfw FROM images i"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY i.created_at DESC, i.id DESC LIMIT ? OFFSET ?"
        params += [int(limit), int(offset)]
        with self._lock:
            rows = self._db.execute(sql, params).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def used_loras(self):
        """Every LoRA that appears in the library, with how many images used it."""
        with self._lock:
            rows = self._db.execute(
                "SELECT lora, COUNT(DISTINCT image_id) AS n FROM image_loras GROUP BY lora ORDER BY lora"
            ).fetchall()
            return [{"name": r["lora"], "count": r["n"]} for r in rows]

    def delete_images(self, ids, include_versions=False, delete_files=True, dry_run=False):
        """Remove images from the library and (by default) delete their files from disk permanently.

        include_versions: also delete face-restored / enhanced versions made from these images (recursively).
        dry_run: only report what would be deleted.
        Returns {"ids": [...], "paths": [...], "versions": n, "errors": [...]}.
        """
        ids = {int(i) for i in ids}
        with self._lock:
            n_versions = 0
            if include_versions:
                frontier = set(ids)
                while frontier:
                    q = ",".join("?" * len(frontier))
                    kids = {r[0] for r in self._db.execute(f"SELECT id FROM images WHERE parent_id IN ({q})", list(frontier))}
                    kids -= ids
                    n_versions += len(kids)
                    ids |= kids
                    frontier = kids
            else:
                q = ",".join("?" * len(ids)) or "NULL"
                n_versions = self._db.execute(
                    f"SELECT COUNT(*) FROM images WHERE parent_id IN ({q}) AND id NOT IN ({q})",
                    list(ids) * 2).fetchone()[0]
            q = ",".join("?" * len(ids)) or "NULL"
            rows = self._db.execute(f"SELECT id, path FROM images WHERE id IN ({q})", list(ids)).fetchall()
            result = {"ids": [r["id"] for r in rows], "paths": [r["path"] for r in rows],
                      "versions": n_versions, "errors": []}
            if dry_run or not rows:
                return result
            if delete_files:
                for r in rows:
                    try:
                        os.remove(r["path"])
                    except FileNotFoundError:
                        pass  # already gone: still remove it from the library
                    except OSError as e:
                        result["errors"].append(f"{r['path']}: {e.strerror}")
            q = ",".join("?" * len(result["ids"]))
            self._db.execute(f"DELETE FROM images WHERE id IN ({q})", result["ids"])
            # versions whose original was deleted (but kept) become standalone images
            self._db.execute(f"UPDATE images SET parent_id = NULL WHERE parent_id IN ({q})", result["ids"])
            self._db.commit()
            return result

    def relocate(self, old_dir, new_dir):
        """Move every library image stored under old_dir into new_dir (keeping date subfolders)."""
        old_dir, new_dir = Path(old_dir).resolve(), Path(new_dir).resolve()
        moved, failed = 0, []
        with self._lock:
            rows = self._db.execute("SELECT id, path FROM images").fetchall()
            for r in rows:
                src = Path(r["path"])
                try:
                    rel = src.resolve().relative_to(old_dir)
                except ValueError:
                    continue  # stored somewhere else (e.g. an older custom folder): leave it alone
                dest = new_dir / rel
                try:
                    if src.exists():
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        if dest.exists():
                            dest = dest.with_name(f"{dest.stem}_{r['id']}{dest.suffix}")
                        shutil.move(str(src), str(dest))
                    self._db.execute("UPDATE images SET path=? WHERE id=?", (str(dest), r["id"]))
                    moved += 1
                except OSError as e:
                    failed.append(f"{src}: {e.strerror}")
            self._db.commit()
        return {"moved": moved, "failed": failed}

    def rewrite_path_prefix(self, old_prefix, new_prefix):
        """Point stored image paths (and the saved image folder) at a moved data folder."""
        old_prefix, new_prefix = old_prefix.rstrip("/") + "/", new_prefix.rstrip("/") + "/"
        with self._lock:
            cur = self._db.execute(
                "UPDATE images SET path = ? || substr(path, ?) WHERE substr(path, 1, ?) = ?",
                (new_prefix, len(old_prefix) + 1, len(old_prefix), old_prefix))
            changed = cur.rowcount
            row = self._db.execute("SELECT value_json FROM app_settings WHERE key='images_dir'").fetchone()
            if row:
                d = json.loads(row[0])
                if isinstance(d, str) and (d + "/").startswith(old_prefix):
                    d = new_prefix + d[len(old_prefix):] if len(d) >= len(old_prefix) else new_prefix.rstrip("/")
                    self._db.execute("UPDATE app_settings SET value_json=? WHERE key='images_dir'",
                                     (json.dumps(d.rstrip("/")),))
            self._db.commit()
        return changed

    def relink_models(self, fix):
        """Rename the model each picture was made with, e.g. 'local:/old/drive/x.safetensors' after the file
        moved. fix(model_id) -> new id. Returns how many pictures changed."""
        n = 0
        with self._lock:
            rows = self._db.execute("SELECT id, model, settings_json FROM images WHERE model LIKE 'local:%'").fetchall()
            for r in rows:
                new = fix(r["model"])
                if new == r["model"]:
                    continue
                st = json.loads(r["settings_json"] or "{}")
                if st.get("model") == r["model"]:
                    st["model"] = new
                self._db.execute("UPDATE images SET model=?, settings_json=? WHERE id=?", (new, json.dumps(st), r["id"]))
                n += 1
            self._db.commit()
        return n

    def relink_preset_models(self, fix):
        """Same for presets that name a model inside a moved folder. Returns how many changed."""
        n = 0
        with self._lock:
            for r in self._db.execute("SELECT id, settings_json FROM presets").fetchall():
                st = json.loads(r["settings_json"] or "{}")
                new = fix(st.get("model"))
                if new != st.get("model"):
                    st["model"] = new
                    self._db.execute("UPDATE presets SET settings_json=? WHERE id=?", (json.dumps(st), r["id"]))
                    n += 1
            self._db.commit()
        return n

    # ---------- app settings ----------
    def get_setting(self, key, default=None):
        with self._lock:
            row = self._db.execute("SELECT value_json FROM app_settings WHERE key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else default

    def set_setting(self, key, value):
        with self._lock:
            self._db.execute(
                "INSERT INTO app_settings (key, value_json) VALUES (?,?) "
                "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json", (key, json.dumps(value)))
            self._db.commit()

    def _row_to_dict(self, row):
        d = dict(row)
        d["starred"] = bool(d["starred"])
        if "nsfw" in d:
            d["nsfw"] = bool(d["nsfw"])
        if d.get("nsfw_mark") is not None:
            d["nsfw_mark"] = bool(d["nsfw_mark"])
        d["settings"] = json.loads(d.pop("settings_json") or "{}")
        d["loras"] = [
            {"name": r["lora"], "weight": r["weight"]}
            for r in self._db.execute(
                "SELECT lora, weight FROM image_loras WHERE image_id=?", (d["id"],)
            ).fetchall()
        ]
        return d

    # ---------- presets ----------
    def save_preset(self, name, settings):
        name = (name or "").strip()
        if not name:
            raise ValueError("Preset name can't be empty")
        clean = {k: settings[k] for k in PRESET_KEYS if k in settings}
        with self._lock:
            self._db.execute(
                """INSERT INTO presets (name, created_at, settings_json) VALUES (?,?,?)
                   ON CONFLICT(name) DO UPDATE SET settings_json=excluded.settings_json""",
                (name, time.time(), json.dumps(clean)),
            )
            self._db.commit()
        return self.list_presets()

    def list_presets(self):
        with self._lock:
            rows = self._db.execute("SELECT * FROM presets ORDER BY name COLLATE NOCASE").fetchall()
            return [
                {"id": r["id"], "name": r["name"], "settings": json.loads(r["settings_json"])}
                for r in rows
            ]

    def delete_preset(self, name):
        with self._lock:
            self._db.execute("DELETE FROM presets WHERE name=?", (name,))
            self._db.commit()
        return self.list_presets()
