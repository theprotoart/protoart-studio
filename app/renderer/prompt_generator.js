/* Danbooru-style random prompt generator (safe-for-work tags only).
 * Builds prompts in the usual tag order: quality, subject count, character traits,
 * clothing, expression, pose, setting, lighting, composition, style.
 */
(function () {
  const T = {
    hair_color: ["blonde hair", "black hair", "brown hair", "white hair", "silver hair", "red hair", "pink hair",
      "blue hair", "purple hair", "green hair", "orange hair", "grey hair", "aqua hair", "multicolored hair",
      "gradient hair", "two-tone hair", "streaked hair"],
    hair_length: ["short hair", "medium hair", "long hair", "very long hair", "bob cut", "pixie cut"],
    hair_style: ["ponytail", "twintails", "side ponytail", "braid", "twin braids", "hair bun", "double bun",
      "messy hair", "wavy hair", "curly hair", "straight hair", "hime cut", "ahoge", "hair over one eye",
      "sidelocks", "blunt bangs", "swept bangs", "hair between eyes"],
    hair_acc: ["hair ribbon", "hair flower", "hairclip", "hairband", "hair bow", "headband", "beret", "witch hat",
      "sun hat", "baseball cap", "beanie", "hood up", "cat ears", "fox ears", "animal ears", "halo", "horns"],
    eyes: ["blue eyes", "red eyes", "green eyes", "brown eyes", "purple eyes", "yellow eyes", "golden eyes",
      "grey eyes", "aqua eyes", "pink eyes", "heterochromia", "orange eyes"],
    expression: ["smile", "light smile", "grin", "open mouth", "closed eyes", "laughing", "serious", "expressionless",
      "blush", "pout", "surprised", "smirk", "happy", "looking at viewer", "looking away", "looking back",
      "looking up", "half-closed eyes", "wink", ":d", "teeth"],
    top: ["white shirt", "collared shirt", "hoodie", "sweater", "turtleneck", "cardigan", "jacket", "leather jacket",
      "blazer", "sailor collar", "school uniform", "serafuku", "t-shirt", "crop top", "kimono", "yukata",
      "hanfu", "maid", "armor", "cape", "coat", "trench coat", "vest", "military uniform", "lab coat"],
    bottom: ["pleated skirt", "long skirt", "miniskirt", "shorts", "denim shorts", "jeans", "pants", "hakama"],
    dress: ["dress", "white dress", "black dress", "sundress", "gothic lolita", "frilled dress", "evening gown",
      "wedding dress", "china dress", "sleeveless dress"],
    legwear: ["thighhighs", "black thighhighs", "white thighhighs", "pantyhose", "kneehighs", "socks", "striped thighhighs"],
    shoes: ["boots", "sneakers", "loafers", "sandals", "high heels", "knee boots", "barefoot"],
    accessory: ["glasses", "scarf", "necktie", "bowtie", "choker", "earrings", "necklace", "gloves", "fingerless gloves",
      "wristwatch", "headphones", "bag", "backpack", "umbrella", "holding umbrella", "holding book", "holding cup",
      "holding flower", "holding sword", "holding staff", "holding phone", "bandaid on face", "eyepatch"],
    pose: ["standing", "sitting", "walking", "running", "jumping", "lying", "kneeling", "crossed arms",
      "hands on hips", "arms up", "arms behind back", "hand on own chin", "peace sign", "waving", "reaching",
      "leaning forward", "head tilt", "dancing", "stretching", "hugging own legs", "fighting stance"],
    setting: ["outdoors", "indoors", "city", "street", "rooftop", "alley", "train station", "classroom", "library",
      "cafe", "bedroom", "kitchen", "forest", "field", "flower field", "beach", "ocean", "lake", "river", "mountain",
      "desert", "snow", "shrine", "temple", "castle", "ruins", "garden", "park", "bridge", "space", "underwater",
      "cyberpunk", "fantasy", "night city", "sky", "cloud", "meadow", "cherry blossoms", "autumn leaves"],
    time_weather: ["day", "night", "sunset", "sunrise", "evening", "starry sky", "full moon", "rain", "snowing",
      "fog", "cloudy sky", "blue sky", "aurora", "lightning"],
    lighting: ["sunlight", "backlighting", "rim lighting", "dappled sunlight", "light rays", "sunbeam", "god rays",
      "neon lights", "lens flare", "bloom", "glowing", "soft lighting", "dramatic lighting", "cinematic lighting",
      "candlelight", "moonlight", "volumetric lighting"],
    composition: ["upper body", "cowboy shot", "full body", "portrait", "close-up", "from side", "from behind",
      "from above", "from below", "dutch angle", "wide shot", "dynamic angle", "depth of field", "blurry background",
      "bokeh", "fisheye", "panorama"],
    style: ["anime coloring", "watercolor (medium)", "oil painting (medium)", "sketch", "lineart", "flat color",
      "cel shading", "pastel colors", "monochrome", "greyscale", "limited palette", "retro artstyle", "1990s (style)",
      "chibi", "realistic", "painterly", "impressionism", "ukiyo-e", "pixel art", "colorful", "vibrant colors",
      "muted color", "traditional media", "concept art"],
    scenery_extra: ["no humans", "scenery", "building", "architecture", "tree", "water", "reflection", "horizon",
      "landscape", "cityscape", "lantern", "street lamp", "power lines", "vending machine", "window", "stairs",
      "boat", "torii", "waterfall", "cliff", "path", "grass", "flowers", "butterfly", "bird", "cat"],
    effects: ["petals", "falling petals", "sparkle", "light particles", "floating hair", "wind", "motion blur",
      "bubbles", "fireflies", "confetti", "snowflakes", "chromatic aberration", "film grain"],
  };

  const QUALITY = {
    standard: ["masterpiece", "best quality", "highres", "absurdres"],
    illustrious: ["masterpiece", "best quality", "amazing quality", "very aesthetic", "absurdres", "newest"],
    pony: ["score_9", "score_8_up", "score_7_up", "source_anime"],
    none: [],
  };
  const NEGATIVE = {
    standard: "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, jpeg artifacts, signature, watermark, username, blurry",
    illustrious: "lowres, worst quality, bad quality, bad anatomy, bad hands, jpeg artifacts, signature, watermark, text, old, early",
    pony: "score_4, score_5, score_6, lowres, bad anatomy, bad hands, text, watermark, signature",
    none: "",
  };

  const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
  const maybe = (p) => Math.random() < p;
  function pickN(arr, n) {
    const copy = arr.slice();
    const out = [];
    while (out.length < n && copy.length) out.push(copy.splice(Math.floor(Math.random() * copy.length), 1)[0]);
    return out;
  }

  /**
   * opts: { subject: 'random'|'1girl'|'1boy'|'2girls'|'1girl 1boy'|'scenery'|'animal',
   *         quality: 'standard'|'illustrious'|'pony'|'none', detail: 1-3, style: true|false }
   * returns { prompt, negative }
   */
  // animals (subject "animal")
  const ANIMAL_TAGS = ["cat", "dog", "fox", "wolf", "bird", "owl", "rabbit", "horse", "deer", "tiger", "lion", "bear",
    "panda", "red panda", "dragon", "fish", "butterfly", "frog", "squirrel", "hedgehog", "otter", "raccoon", "shiba inu",
    "penguin", "parrot", "snake", "turtle", "whale", "dolphin", "octopus", "cow", "pig", "sheep", "goat", "bison"];

  function generate(opts = {}) {
    const quality = QUALITY[opts.quality] ? opts.quality : "standard";
    const detail = Math.max(1, Math.min(3, opts.detail || 2));
    let subject = opts.subject || "random";
    if (subject === "random") subject = pick(["1girl", "1girl", "1girl", "1boy", "1boy", "2girls", "1girl 1boy", "scenery", "animal"]);

    const tags = [...QUALITY[quality]];
    if (subject === "animal") {
      tags.push("no humans", "animal focus", pick(ANIMAL_TAGS));
      tags.push(...pickN(["full body", "looking at viewer", "sitting", "standing", "lying", "running", "fluffy", "fur",
                          "feathers", "whiskers", "tail", "paws", "open mouth", "closed eyes"], 1 + detail));
      tags.push(pick(T.setting.filter((t) => !/(classroom|bedroom|bathroom|office|cafe|library)/.test(t))), pick(T.time_weather));
    } else if (subject === "scenery") {
      tags.push("no humans", "scenery");
      tags.push(...pickN(T.scenery_extra.filter((t) => t !== "no humans" && t !== "scenery"), 1 + detail));
      tags.push(pick(T.setting), pick(T.time_weather));
    } else {
      if (subject === "1girl 1boy") tags.push("1girl", "1boy");
      else tags.push(subject);
      if (subject === "1girl" || subject === "1boy") tags.push("solo");
      tags.push(pick(T.hair_color), pick(T.hair_length), pick(T.eyes));
      if (maybe(0.4 + 0.2 * detail)) tags.push(pick(T.hair_style));
      if (maybe(0.2 + 0.15 * detail)) tags.push(pick(T.hair_acc));
      tags.push(pick(T.expression));
      const girlish = subject.includes("girl");
      if (girlish && maybe(0.35)) tags.push(pick(T.dress));
      else {
        tags.push(pick(T.top));
        if (maybe(0.7)) tags.push(pick(girlish ? T.bottom : T.bottom.filter((b) => !b.includes("skirt"))));
      }
      if (girlish && maybe(0.25 * detail)) tags.push(pick(T.legwear));
      if (maybe(0.2 * detail)) tags.push(pick(T.shoes));
      tags.push(...pickN(T.accessory, detail - 1));
      tags.push(pick(T.pose), pick(T.composition));
      tags.push(pick(T.setting));
      if (maybe(0.6)) tags.push(pick(T.time_weather));
    }
    tags.push(...pickN(T.lighting, detail === 3 ? 2 : 1));
    if (maybe(0.3 * detail)) tags.push(pick(T.effects));
    if (opts.style !== false && maybe(0.6)) tags.push(pick(T.style));

    const seen = new Set();
    const prompt = tags.filter((t) => !seen.has(t) && seen.add(t)).join(", ");
    return { prompt, negative: NEGATIVE[quality] };
  }

  window.DanbooruPrompt = { generate, categories: Object.keys(T), qualityPresets: Object.keys(QUALITY) };
})();
