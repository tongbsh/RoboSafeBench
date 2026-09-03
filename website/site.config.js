/**
 * Public links and venue — fill these when the paper / repo go live.
 * Empty strings render as “coming soon” on the page (no fake URLs).
 */
window.ROBOSAFE_SITE = {
  name: "RoboSafeBench",
  year: 2026,
  venue: "",
  paperUrl: "",
  arxivUrl: "",
  codeUrl: "",
  docsUrl: "",
  hfUrl: "",
  contact: "",
  /**
   * Optional YouTube / Bilibili ids for a page-top teaser.
   * Example: teaserYoutube: "dQw4w9WgXcQ"
   */
  teaserYoutube: "",
  teaserBilibili: "",
  /**
   * Per-cell rollout video. Keys: "{backend}_l{level}".
   * If omitted, the page tries assets/videos/{key}.mp4.
   * Missing files show a short placeholder — no illustration.
   * Values: relative path, https URL, YouTube id/url, or
   * { src, youtube, poster }.
   */
  videoBase: "assets/videos/",
  videos: {},
  /**
   * Page shows every backend × L0–L3. Put files at
   * assets/videos/{backend}_l{level}.mp4
   */
  defaultDemo: { backend: "libero", level: 3 },
};
