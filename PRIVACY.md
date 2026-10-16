# Privacy

Protoart Studio runs entirely on your Mac. Your prompts, pictures, LoRAs and settings never leave it.
There is no account, no telemetry, no analytics and no crash reporting.

## When the app goes online

Only when something you asked for isn't on your Mac yet, and never in Strict offline mode (Settings → Privacy → Strict offline):

| Where | When | What is sent |
| --- | --- | --- |
| huggingface.co (and its file storage) | First use of a model or tool, or adding a model from Hugging Face | Standard file requests, telemetry switched off. No account token, unless you saved a Hugging Face access token (Models tab → Add from Hugging Face, for gated models): then it's sent to huggingface.co only, never to the file servers it redirects to. A token saved by other apps on this Mac is never used. |
| civitai.com (and its file storage) | Only when you press Check or Download under Models tab → Add from Civitai | The model link you pasted; your Civitai API key if you saved one (sent only to civitai.com). |
| github.com | First use of face restore or the face detector | File requests |
| pypi.org | Installing the fast ERNIE engine (once) | Package requests |

Links you click (for example "Get PhotoCraft") open in your own browser.

## What is stored on your Mac

| Location | Contents |
| --- | --- |
| `~/.protoartstudio/library.sqlite` | Your picture library: prompts, settings, file paths, presets, saved trigger words |
| `~/.protoartstudio/images/` (or the folder you chose) | Your pictures |
| `~/.protoartstudio/logs/` | The backend log (up to about 10 MB). Settings → Backend log → Clear log empties it. |
| `~/.protoartstudio/` (other folders) | Thumbnails, LoRAs, wildcards, edit tracking, model configs, the fast engine |
| `~/Library/Application Support/Protoart Studio/` | Window size, the last form you filled in, your look / theme, ★ favourite models and type locks |
| `~/.cache/huggingface/` | Downloaded models (shared with other apps) |
| `~/.protoartstudio/library.sqlite` → Civitai API key | Only if you save one; never shown again in the app, never logged |
| `~/.protoartstudio/library.sqlite` → Hugging Face access token | Only if you save one; never shown again in the app, never logged, not written to `~/.cache/huggingface` |
| The folder you export to (Library → Batch edit & export) | Copies you exported. By default they carry no prompt, settings or location; you can choose to keep the prompt and settings (file paths are never included) |

Delete these folders to remove everything the app stored.

## Details inside your pictures

By default each picture carries its prompt and settings (model, LoRAs, seed…), like other Stable Diffusion apps, so
you can reuse them later. Anyone you share the picture with can read them. File paths on your Mac are never included,
only file names. Settings → Details inside pictures → "Embed nothing" turns this off.
