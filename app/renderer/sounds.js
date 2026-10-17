// Notification sounds: a short chime when pictures are finished, a two-note chime when training is done, and a
// low tone when a job fails. Made on the fly with Web Audio (no sound files), warm and soft like a studio bell.
// Settings → Look → "🔔 Sounds" switches them off (stored per Mac in protoart.sounds).
(function () {
  "use strict";
  let ctx = null;
  const enabled = () => {
    try { return localStorage.getItem("protoart.sounds") !== "off"; } catch (e) { return true; }
  };

  function audio() {
    if (!ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return null;
      ctx = new AC();
    }
    if (ctx.state === "suspended") ctx.resume();
    return ctx;
  }

  // one bell-like note: a sine plus a soft overtone, quick attack, long gentle decay
  function note(freq, at, dur, gain) {
    const a = audio();
    if (!a) return;
    const t = a.currentTime + at;
    const out = a.createGain();
    out.gain.setValueAtTime(0.0001, t);
    out.gain.exponentialRampToValueAtTime(gain, t + 0.012);
    out.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    const lp = a.createBiquadFilter();
    lp.type = "lowpass";
    lp.frequency.value = 3200;
    out.connect(lp).connect(a.destination);
    for (const [mult, g] of [[1, 1], [2.01, 0.28], [3.02, 0.08]]) {
      const o = a.createOscillator();
      const og = a.createGain();
      o.type = "sine";
      o.frequency.value = freq * mult;
      og.gain.value = g;
      o.connect(og).connect(out);
      o.start(t);
      o.stop(t + dur + 0.05);
    }
  }

  const SOUNDS = {
    done: () => { note(880, 0, 0.9, 0.18); note(1318.5, 0.11, 1.1, 0.14); },             // A5 → E6
    trained: () => { note(659.3, 0, 0.8, 0.16); note(880, 0.16, 0.8, 0.15); note(1318.5, 0.34, 1.4, 0.15); },
    error: () => { note(220, 0, 0.6, 0.16); note(196, 0.18, 0.8, 0.14); },
  };

  window.ProtoSound = {
    play(kind) {
      if (!enabled()) return;
      try { (SOUNDS[kind] || SOUNDS.done)(); } catch (e) { /* no audio device: stay quiet */ }
    },
    setEnabled(on) {
      try { localStorage.setItem("protoart.sounds", on ? "on" : "off"); } catch (e) { /* ignore */ }
    },
    get enabled() { return enabled(); },
  };
})();
