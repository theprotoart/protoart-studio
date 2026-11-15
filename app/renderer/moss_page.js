// 📰 MOSS-Image page layouts: comic pages and posters drawn by MOSS-Image from a JSON layout (its own prompt format:
// boxes [y_min, x_min, y_max, x_max] in pixels with the canvas centre as the origin, "yxyx" / "abs_centralized").
// Each panel is a picture box; its speech bubble and caption box are text boxes with the exact words.
(function () {
const PAGE_TEMPLATES = [
  { id: "3", label: "3 panels", hint: "three wide panels, top to bottom", cells: (W, H) => grid(W, H, 1, 3) },
  { id: "4", label: "4 panels", hint: "a 2 × 2 grid", cells: (W, H) => grid(W, H, 2, 2) },
  { id: "6", label: "6 panels", hint: "2 across, 3 down", cells: (W, H) => grid(W, H, 2, 3) },
  { id: "splash", label: "Splash + 2", hint: "one big panel, two small under it", cells: (W, H) => splash(W, H) },
  { id: "poster", label: "Poster", hint: "title, one picture, subtitle", poster: true, cells: (W, H) => [[24, 150, W - 24, H - 130]] },
];

const PAGE_STYLES = [
  "classic American comic book, bold black ink lines, flat bright colours, halftone dots",
  "Japanese manga, black and white, screentone shading, expressive line work",
  "European bande dessinée, clean ligne claire lines, soft flat colours",
  "children's picture-book comic, soft watercolour, rounded friendly shapes",
  "gritty noir comic, heavy shadows, muted colours",
];

const PAGE_SIZES = [
  { id: "draft", label: "Draft 768 × 1024", w: 768, h: 1024, hint: "for trying a layout: about 10 minutes on an 8 GB Mac" },
  { id: "full", label: "Full 832 × 1216", w: 832, h: 1216, hint: "for the final page: sharper, about 14 minutes on an 8 GB Mac" },
];

const M = 24, G = 16;  // page margin and gutter (px)
const TITLE = 110;      // the title band at the top of a comic page

function grid(W, H, cols, rows, top = M) {
  const cw = (W - 2 * M - (cols - 1) * G) / cols, ch = (H - top - M - (rows - 1) * G) / rows, out = [];
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      const x0 = M + c * (cw + G), y0 = top + r * (ch + G);
      out.push([x0, y0, x0 + cw, y0 + ch]);
    }
  return out;
}

function splash(W, H, top = M) {
  const h1 = (H - top - M - G) * 0.6, h2 = H - top - M - G - h1, cw = (W - 2 * M - G) / 2;
  return [[M, top, W - M, top + h1],
          [M, top + h1 + G, M + cw, top + h1 + G + h2],
          [M + cw + G, top + h1 + G, W - M, top + h1 + G + h2]];
}

// top-left pixels (x0, y0, x1, y1) -> MOSS centred [y_min, x_min, y_max, x_max]
const box = (x0, y0, x1, y1, W, H) => [y0 - H / 2, x0 - W / 2, y1 - H / 2, x1 - W / 2].map((v) => Math.round(v));

function pageCells(page, W, H) {
  const t = PAGE_TEMPLATES.find((x) => x.id === page.template) || PAGE_TEMPLATES[1];
  let cells = t.cells(W, H);
  if (!t.poster && (page.title || "").trim()) {  // room for the title at the top
    const shift = TITLE - M, scale = (H - TITLE - M) / (H - 2 * M);
    cells = cells.map(([x0, y0, x1, y1]) => [x0, TITLE + (y0 - M) * scale, x1, TITLE + (y1 - M) * scale]);
  }
  return { t, cells };
}

function panelCount(page) {
  const t = PAGE_TEMPLATES.find((x) => x.id === page.template) || PAGE_TEMPLATES[1];
  return t.cells(1000, 1000).length;
}

// The layout MOSS-Image reads, plus the boxes to draw as a preview: [{kind, x0, y0, x1, y1, text}]
function buildLayout(page, describe, W, H) {
  const { t, cells } = pageCells(page, W, H);
  const style = ((page.style || "").trim()) || (t.poster ? "clean modern poster design, striking colours" : PAGE_STYLES[0]);
  const items = [], boxes = [];
  const add = (label, caption, text, b) => {
    const it = { label, caption, bbox: box(...b, W, H) };
    if (text) it.text = text;
    items.push(it);
    boxes.push({ kind: label, x0: b[0], y0: b[1], x1: b[2], y1: b[3], text: text || caption.slice(0, 40) });
  };
  const title = (page.title || "").trim();
  if (t.poster) {
    if (title) add("text", `the poster's title in big bold display letters, ${page.title_look || "clean and striking"}`, title,
                   [M, M, W - M, 130]);
    const sub = (page.subtitle || "").trim();
    if (sub) add("text", "a smaller subtitle line under the picture", sub, [M + 40, H - 110, W - M - 40, H - M]);
  } else if (title) {
    add("text", "the comic's title in big bold comic lettering across the top of the page", title, [M, M, W - M, TITLE - 12]);
  }
  const panels = page.panels || [];
  cells.forEach((c, i) => {
    const p = panels[i] || {};
    const scene = (p.scene || "").trim() || (t.poster ? describe : `panel ${i + 1} of the story`);
    const words = !t.poster && ((p.bubble || "").trim() || (p.caption || "").trim());
    // a panel with no words of its own must say so, or MOSS-Image letters the scene description into it
    add("object", t.poster ? `${scene}, ${style}` : `comic panel ${i + 1} with a black border, a picture only` +
        `${words ? "" : " with no speech bubble and no writing"}, showing ${scene}`, "", c);
    const [x0, y0, x1, y1] = c, w = x1 - x0, h = y1 - y0;
    const say = (p.bubble || "").trim();
    if (say && !t.poster) {  // a speech bubble in the top part of the panel
      const bw = Math.min(w * 0.62, 360), bh = Math.min(h * 0.3, 140);
      const bx = p.bubble_side === "right" ? x1 - bw - 12 : x0 + 12;
      add("text", `a white speech bubble with a black outline and a tail, ${p.bubble_kind === "thought" ? "a cloud-shaped thought bubble, " : ""}in hand-lettered comic capitals`,
          say, [bx, y0 + 12, bx + bw, y0 + 12 + bh]);
    }
    const cap = (p.caption || "").trim();
    if (cap && !t.poster) {  // a narration box along the bottom
      const ch = Math.min(h * 0.18, 70);
      add("text", "a yellow rectangular narration box with black comic lettering", cap, [x0 + 8, y1 - ch - 8, x0 + Math.min(w - 16, 380), y1 - 8]);
    }
  });
  const kind = t.poster ? "poster" : `comic book page with ${cells.length} panels`;
  const caption = `A ${kind}${describe ? ": " + describe : ""}. Style: ${style}.` +
    (t.poster ? "" : " White gutters between the panels, every panel drawn in the same style with the same characters.");
  const layout = { bbox_format: "yxyx", bbox_coord_space: "abs_centralized",
                   summary: { meta: t.poster ? ["poster"] : ["comic"], canvas: t.poster ? "single" : "multi_panel", caption },
                   items };
  return { layout, boxes, caption };
}

  const api = { PAGE_TEMPLATES, PAGE_STYLES, PAGE_SIZES, pageCells, panelCount, buildLayout };
  if (typeof window !== "undefined") window.MossPage = api;
  if (typeof module !== "undefined") module.exports = api;
})();
