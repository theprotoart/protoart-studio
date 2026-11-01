// ⚥ Swap gender: turns a prompt's women into men and men into women ("a woman in a brown outfit" ->
// "a man in a brown outfit", "1girl, solo" -> "1boy, solo", "her hair" -> "his hair"). Works on sentences and on
// Stable Diffusion tags alike; everything that isn't a gendered word is left exactly as it was.
(function () {
  const PAIRS = [
    ["woman", "man"], ["women", "men"], ["girl", "boy"], ["girls", "boys"], ["lady", "gentleman"], ["ladies", "gentlemen"],
    ["female", "male"], ["females", "males"], ["feminine", "masculine"], ["femininity", "masculinity"],
    ["she", "he"], ["herself", "himself"], ["queen", "king"], ["princess", "prince"], ["empress", "emperor"],
    ["duchess", "duke"], ["countess", "count"], ["baroness", "baron"], ["actress", "actor"], ["waitress", "waiter"],
    ["heroine", "hero"], ["mother", "father"], ["mom", "dad"], ["mommy", "daddy"], ["daughter", "son"],
    ["sister", "brother"], ["wife", "husband"], ["girlfriend", "boyfriend"], ["bride", "groom"], ["aunt", "uncle"],
    ["niece", "nephew"], ["grandmother", "grandfather"], ["grandma", "grandpa"], ["granddaughter", "grandson"],
    ["witch", "warlock"], ["sorceress", "sorcerer"], ["enchantress", "enchanter"], ["priestess", "priest"],
    ["goddess", "god"], ["nun", "monk"], ["madam", "sir"], ["mrs", "mr"], ["policewoman", "policeman"],
    ["businesswoman", "businessman"], ["spokeswoman", "spokesman"], ["sportswoman", "sportsman"],
    ["horsewoman", "horseman"], ["chairwoman", "chairman"], ["fisherwoman", "fisherman"], ["cowgirl", "cowboy"],
    ["schoolgirl", "schoolboy"], ["mermaid", "merman"], ["lass", "lad"], ["gal", "guy"], ["maid", "butler"],
    ["housewife", "househusband"], ["stepmother", "stepfather"], ["stepdaughter", "stepson"], ["stepsister", "stepbrother"],
    ["matriarch", "patriarch"], ["heiress", "heir"], ["hostess", "host"], ["stewardess", "steward"],
    ["seamstress", "tailor"], ["shepherdess", "shepherd"], ["huntress", "hunter"], ["tigress", "tiger"], ["lioness", "lion"],
    ["valkyrie", "viking"], ["ballerina", "ballet dancer"], ["damsel", "squire"], ["maiden", "young man"],
    ["1girl", "1boy"], ["2girls", "2boys"], ["3girls", "3boys"], ["4girls", "4boys"], ["5girls", "5boys"], ["6+girls", "6+boys"],
    ["multiple girls", "multiple boys"], ["female focus", "male focus"], ["female child", "male child"],
  ];
  // plurals of the simple pairs that weren't listed
  for (const [a, b] of PAIRS.slice()) {
    if (/\s|\d/.test(a + b) || /s$/.test(a) || /s$/.test(b)) continue;
    const pl = (w) => (/(ss|x|ch|sh)$/.test(w) ? w + "es" : /[^aeiou]y$/.test(w) ? w.slice(0, -1) + "ies" : w + "s");
    if (!PAIRS.some(([x, y]) => x === pl(a) || y === pl(b))) PAIRS.push([pl(a), pl(b)]);
  }
  const MAP = new Map();
  for (const [a, b] of PAIRS) {
    if (!MAP.has(a)) MAP.set(a, b);
    if (!MAP.has(b)) MAP.set(b, a);
  }
  // words after which "her" is the object ("looking at her", "behind her,"), not "her hair"
  const AFTER_OBJECT = new Set(("and or but with in on at to as while from by for of into onto over under up down off out " +
    "away back around about across behind beside near than then so because when where who which that the a an is was " +
    "are were be been being has have had will would can could should may might must do does did not again too also " +
    "very just").split(" "));

  function keepCase(src, out) {
    if (src === src.toUpperCase() && /[A-Z]/.test(src) && src.length > 1) return out.toUpperCase();
    if (src[0] === src[0].toUpperCase() && /[A-Z]/.test(src[0])) return out[0].toUpperCase() + out.slice(1);
    return out;
  }

  // multi-word entries ("multiple girls", "female focus") first, so they swap as one
  const MULTI = [...MAP.keys()].filter((k) => /\s/.test(k)).sort((a, b) => b.length - a.length);
  const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

  function swap(text) {
    if (!text) return { text: text || "", count: 0 };
    let count = 0;
    const holds = [];
    let t = text;
    for (const k of MULTI) {
      t = t.replace(new RegExp(`(^|[^\\w+])(${esc(k)})(?![\\w+])`, "gi"), (all, pre, w) => {
        count++;
        holds.push(keepCase(w, MAP.get(k)));
        return pre + `\u0000${holds.length - 1}\u0000`;
      });
    }
    // words, keeping everything between them as it is
    const parts = t.split(/([A-Za-z0-9+']+)/);
    for (let i = 1; i < parts.length; i += 2) {
      const w = parts[i], lw = w.toLowerCase();
      if (lw === "her") {
        // "her hair" -> "his hair"; "at her" / "her," / end -> "him"
        const gap = parts[i + 1] || "", next = (parts[i + 2] || "").toLowerCase();
        const obj = !next || /[.,;:!?)\]\n]/.test(gap) || AFTER_OBJECT.has(next);
        parts[i] = keepCase(w, obj ? "him" : "his"); count++;
      } else if (lw === "his") {
        const gap = parts[i + 1] || "", next = parts[i + 2] || "";
        parts[i] = keepCase(w, !next || /[.,;:!?)\]\n]/.test(gap) ? "hers" : "her"); count++;
      } else if (lw === "him") { parts[i] = keepCase(w, "her"); count++; }
      else if (lw === "hers") { parts[i] = keepCase(w, "his"); count++; }
      else if (MAP.has(lw)) { parts[i] = keepCase(w, MAP.get(lw)); count++; }
    }
    t = parts.join("").replace(/\u0000(\d+)\u0000/g, (_, n) => holds[Number(n)]);
    return { text: t, count };
  }

  const api = { swap };
  if (typeof window !== "undefined") window.GenderSwap = api;
  if (typeof module !== "undefined") module.exports = api;
})();
