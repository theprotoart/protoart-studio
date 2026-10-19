/* Natural-language random prompt generator, for models that read sentences (FLUX, FLUX.2, ERNIE-Image).
 * Same options as the Danbooru one (subject, detail, style), but it writes a short scene description:
 * who, what they look like and wear, what they're doing, where, when, the light, the shot and the look.
 * Safe-for-work wording only.
 */
(function () {
  const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
  const maybe = (p) => Math.random() < p;
  const an = (w) => (/^[aeiou]/i.test(w) ? "an " : "a ") + w;
  const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

  const P = {
    woman: ["young woman", "woman in her thirties", "elderly woman", "woman", "teenage girl", "female astronaut",
      "woman with freckles", "ballet dancer", "street musician", "female knight", "botanist", "jazz singer"],
    man: ["young man", "man in his forties", "old man with a white beard", "man", "teenage boy", "fisherman",
      "chef", "boxer", "male astronaut", "wandering samurai", "mechanic", "violinist"],
    hair: ["short black hair", "long wavy red hair", "a silver buzz cut", "curly brown hair", "a messy blonde bun",
      "braided dark hair", "shoulder-length auburn hair", "bright pink hair", "slicked-back grey hair", "an afro",
      "long straight white hair", "a blue bob cut"],
    eyes: ["green eyes", "dark brown eyes", "pale blue eyes", "amber eyes", "grey eyes", "hazel eyes"],
    outfit: ["a worn leather jacket and jeans", "a flowing white linen dress", "a tailored navy suit", "a chunky knitted sweater",
      "a yellow raincoat", "ornate silver armour", "a vintage floral blouse", "a hooded travelling cloak",
      "a red kimono with gold embroidery", "overalls splattered with paint", "a sleek black turtleneck",
      "a spacesuit with scuffed patches", "a cosy oversized hoodie", "a velvet evening gown"],
    extra: ["round glasses", "a long scarf", "a wide-brimmed hat", "a backpack", "headphones around the neck",
      "a camera on a strap", "a small tattoo on the wrist", "gold earrings", "a flower tucked behind one ear"],
    action: ["reading a book", "laughing at something off-camera", "sipping coffee", "walking through the crowd",
      "looking straight at the camera", "playing an old guitar", "painting on a large canvas", "holding a lantern",
      "riding a bicycle", "feeding pigeons", "dancing alone", "gazing out of a window", "tuning a radio",
      "writing in a notebook", "carrying a stack of books", "standing in the rain with an umbrella"],
    pair: ["two friends sharing an umbrella", "a couple dancing", "two old men playing chess", "a mother and daughter baking",
      "two astronauts floating side by side", "a girl and her grandfather fishing", "two musicians jamming together"],
    place: ["on a busy Tokyo street", "in a quiet library with tall shelves", "in a sunflower field", "in a rainy Paris café",
      "in a misty pine forest", "in a neon-lit alley", "in a rooftop garden", "in a desert canyon", "in a cosy cabin",
      "on a train platform", "in a Moroccan market", "in a snowy mountain village", "in an abandoned greenhouse",
      "by a lighthouse on a cliff", "in a recording studio", "in an old bookshop", "on a beach at low tide",
      "aboard a space station"],
    together: ["talking and laughing", "sharing a meal", "walking side by side", "looking at a map together",
      "dancing", "sitting back to back", "taking a selfie", "sharing an umbrella", "playing music together"],
    scene: ["a lighthouse on a rocky coast", "a floating island with waterfalls", "a narrow canal in Venice",
      "a mountain lake reflecting the peaks", "a sprawling cyberpunk city", "an ancient temple in the jungle",
      "a cherry blossom avenue", "a lone tree in a wheat field", "a cosy cottage in the woods", "a desert oasis",
      "an aurora over a frozen fjord", "a harbour full of fishing boats", "a castle on a hill above the clouds"],
    time: ["at golden hour", "at dawn", "at dusk", "at night", "on a foggy morning", "in the middle of a thunderstorm",
      "on a snowy afternoon", "under a full moon", "on a bright summer day", "in autumn"],
    light: ["soft window light", "warm golden sunlight", "dramatic side lighting", "neon reflections",
      "candlelight", "moody low-key lighting", "light streaming through the trees", "overcast, even light",
      "glowing lanterns", "cinematic rim light"],
    shot: ["Close-up portrait", "Full-body shot", "Wide shot", "Medium shot", "Low-angle shot", "Shot from above",
      "Over-the-shoulder view", "Candid street photo"],
    look: ["a 35mm film photo with soft grain", "a sharp, photorealistic photo", "a watercolour painting",
      "a detailed oil painting", "a vibrant anime illustration", "a moody cinematic film still",
      "a vintage 1970s photograph", "a clean digital illustration", "an ink and wash drawing",
      "a portrait photo with a shallow depth of field", "a stylised 3D animation still", "a pastel children's book illustration"],
    mood: ["The mood is calm and nostalgic.", "The atmosphere feels mysterious.", "It feels joyful and full of energy.",
      "There's a quiet, melancholic feeling.", "Everything feels warm and inviting.", "The scene feels epic and vast."],
  };

  function person(kind, avoidHair) {
    const who = pick(kind === "man" ? P.man : P.woman);
    const hair = pick(P.hair.filter((h) => h !== avoidHair));
    let s = `${an(who)} with ${hair}`;
    person.lastHair = hair;
    if (maybe(0.5)) s += ` and ${pick(P.eyes)}`;
    s += `, wearing ${pick(kind === "man" ? P.outfit.filter((o) => !/dress|gown|blouse/.test(o)) : P.outfit)}`;
    return s;
  }

  // animals (subject "animal"), by kind so the place and the details fit: [names, places, coat, doing]
  const ANIMAL_KINDS = {
    land: [["fox", "wolf", "red panda", "snow leopard", "tiger", "lion", "deer", "stag", "rabbit", "hedgehog", "elephant",
            "giraffe", "panda", "koala", "brown bear", "bison", "highland cow", "goat", "lamb", "squirrel", "raccoon", "horse",
            "white stallion"],
           ["in a misty forest", "in a sunny meadow full of wildflowers", "on a snowy mountain slope", "by a calm lake",
            "in tall golden grass", "in a bamboo grove", "on a mossy log", "in a desert at sunset", "in a city park"],
           ["fur", "coat", "eyes", "whiskers", "markings"],
           ["sitting quietly", "running", "looking straight at the viewer", "curled up asleep", "playing", "leaping",
            "drinking from a stream", "walking slowly", "stretching", "resting in the sun"]],
    pet: [["golden retriever", "French bulldog", "border collie", "Siamese cat", "tabby cat", "Maine Coon cat", "corgi",
           "dachshund", "Persian cat", "pug"],
          ["in a cozy living room", "on a windowsill", "in a garden", "on a sofa", "in a city park", "on a sunny porch",
           "on a beach", "in the snow"],
          ["fur", "eyes", "whiskers", "ears", "coat"],
          ["sitting quietly", "looking straight at the viewer", "curled up asleep", "playing with a ball", "yawning",
           "tilting its head", "running", "peeking out from behind a cushion"]],
    bird: [["owl", "barn owl", "raven", "hummingbird", "parrot", "peacock", "flamingo", "kingfisher", "robin", "eagle"],
           ["on a mossy branch", "in a misty forest", "by a calm lake", "among cherry blossoms", "on an old fence post",
            "in a tropical rainforest", "above a mountain valley"],
           ["feathers", "eyes", "wings", "beak"],
           ["perched on a branch", "spreading its wings", "in mid-flight", "looking straight at the viewer", "preening"]],
    sea: [["sea turtle", "whale", "dolphin", "octopus", "koi fish", "jellyfish", "seahorse", "manta ray"],
          ["under a coral reef", "in clear blue water", "in a deep ocean with light rays", "in a garden pond",
           "in a kelp forest"],
          ["scales", "skin", "fins", "tentacles", "patterns"],
          ["gliding slowly", "swimming toward the viewer", "drifting", "turning in the light"]],
  };

  /** opts: { subject: 'random'|'1girl'|'1boy'|'2girls'|'1girl 1boy'|'scenery'|'animal', detail: 1-3, style: true|false }
   *  returns { prompt } (no negative prompt: these models don't use one) */
  function generate(opts = {}) {
    const detail = Math.max(1, Math.min(3, opts.detail || 2));
    let subject = opts.subject || "random";
    if (subject === "random") subject = pick(["1girl", "1girl", "1boy", "1boy", "2girls", "1girl 1boy", "scenery", "animal"]);
    const out = [];
    if (subject === "animal") {
      const [names, places, coat, doing] = ANIMAL_KINDS[pick(["land", "land", "pet", "pet", "bird", "sea"])];
      const look = pick(["curious", "sleepy", "majestic", "tiny", "fluffy", "old", "young", "playful", "proud", "elegant", "alert"]);
      out.push(`${pick(["Close-up", "Medium shot", "Wide shot", "Low-angle view", "Portrait"])} of ${an(look + " " + pick(names))}, ${pick(doing)} ${pick(places)} ${pick(P.time)}.`);
      if (detail >= 2) out.push(`Its ${pick(coat)} ${pick(["catch the light", "are rendered in fine detail", "stand out against the background"])}.`);
    } else if (subject === "scenery") {
      out.push(`${pick(["Wide shot", "Aerial view", "Low-angle view", "Panoramic view", "Distant view"])} of ${pick(P.scene)} ${pick(P.time)}, lit by ${pick(P.light.filter((l) => !/window|candle|neon|lantern/.test(l)))}.`);
      if (detail >= 2) out.push(`No people, just the landscape${maybe(0.5) ? ", with a few birds in the sky" : ""}.`);
    } else if (subject === "2girls" || subject === "1girl 1boy") {
      const a = person("woman");
      const b = person(subject === "2girls" ? "woman" : "man", person.lastHair);
      if (maybe(0.4) && detail < 3) out.push(`${cap(pick(P.pair))} ${pick(P.place)} ${pick(P.time)}.`);
      else {
        out.push(`${cap(a)}, and ${b}, ${pick(P.place)} ${pick(P.time)}.`);
        if (detail >= 2) out.push(`They are ${pick(P.together)}.`);
      }
    } else {
      const p = person(subject === "1boy" ? "man" : "woman");
      out.push(`${pick(P.shot)} of ${p}, ${pick(P.action)} ${pick(P.place)} ${pick(P.time)}.`);
      if (detail >= 2 && maybe(0.7)) out.push(`${subject === "1boy" ? "He" : "She"} has ${pick(P.extra)}.`);
    }
    if (detail >= 2 && subject !== "scenery") out.push(`The scene is lit by ${pick(P.light.filter((l) => subject !== "animal" || !/candle|neon/.test(l)))}.`);
    if (opts.style !== false) out.push(`The picture looks like ${pick(subject === "scenery" ? P.look.filter((l) => !/portrait/.test(l)) : P.look)}.`);
    if (detail >= 3) out.push(pick(P.mood));
    return { prompt: out.join(" ") };
  }

  window.NaturalPrompt = { generate };
})();
