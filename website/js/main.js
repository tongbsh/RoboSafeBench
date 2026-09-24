(() => {
  const cfg = window.ROBOSAFE_SITE || {};
  const dicts = window.ROBOSAFE_I18N;
  const storageKey = "robosafebench-lang";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  function currentLang() {
    const stored = localStorage.getItem(storageKey);
    if (stored === "zh" || stored === "en") return stored;
    return navigator.language.toLowerCase().startsWith("zh") ? "zh" : "en";
  }

  const DEMO_BACKENDS = [
    { id: "libero", label: "LIBERO" },
    { id: "simplerenv", label: "SimplerEnv" },
    { id: "robotwin", label: "RoboTwin" },
  ];
  const DEMO_LEVELS = [0, 1, 2, 3];

  function clipKey(backend, level) {
    return `${backend}_l${level}`;
  }

  function applyLang(lang) {
    const dict = dicts[lang];
    document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
    document.documentElement.dataset.lang = lang;
    $$("[data-i18n]").forEach((el) => {
      const key = el.getAttribute("data-i18n");
      if (dict[key] != null) el.textContent = dict[key];
    });
    $$("[data-i18n-aria]").forEach((el) => {
      const key = el.getAttribute("data-i18n-aria");
      if (dict[key] != null) el.setAttribute("aria-label", dict[key]);
    });
    localStorage.setItem(storageKey, lang);
    syncDemoEmpty();
  }

  function syncDemoEmpty() {
    const dict = dicts[currentLang()] || dicts.en;
    $$(".demo-cell-empty span").forEach((el) => {
      el.textContent = dict.demo_empty;
    });
  }

  function bindLinks() {
    const map = [
      ["#link-paper", cfg.paperUrl],
      ["#link-arxiv", cfg.arxivUrl],
      ["#link-code", cfg.codeUrl],
      ["#link-docs", cfg.docsUrl],
    ];
    map.forEach(([sel, url]) => {
      const a = $(sel);
      if (!a) return;
      if (url) {
        a.href = url;
        a.classList.remove("is-disabled");
        a.removeAttribute("aria-disabled");
      } else {
        a.href = "#";
        a.classList.add("is-disabled");
        a.setAttribute("aria-disabled", "true");
        a.title = (dicts[currentLang()] || dicts.en).soon;
      }
    });

    const venue = $("#venue-badge");
    if (venue) venue.textContent = cfg.venue || (dicts[currentLang()] || dicts.en).venue_fallback;

    if (cfg.teaserYoutube) {
      const box = $("#teaser-embed");
      if (box) {
        box.hidden = false;
        box.innerHTML = `<iframe title="RoboSafeBench teaser" src="https://www.youtube-nocookie.com/embed/${cfg.teaserYoutube}" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>`;
      }
    }
  }

  function bindNav() {
    const btn = $("#menu-btn");
    const links = $("#nav-links");
    btn?.addEventListener("click", () => {
      const open = links.classList.toggle("is-open");
      btn.setAttribute("aria-expanded", String(open));
    });
    $$("#nav-links a").forEach((a) => {
      a.addEventListener("click", () => links.classList.remove("is-open"));
    });

    const sections = $$("main section[id]");
    const navMap = new Map(
      $$("#nav-links a[href^='#']").map((a) => [a.getAttribute("href").slice(1), a]),
    );
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          navMap.forEach((el) => el.classList.remove("is-active"));
          navMap.get(entry.target.id)?.classList.add("is-active");
        });
      },
      { rootMargin: "-40% 0px -50% 0px", threshold: 0.01 },
    );
    sections.forEach((s) => io.observe(s));
  }

  function bindDemo() {
    const host = $("#demo-rows");
    const bar = $("#demo-bar");
    if (!host || !bar) return;
    const initial = (cfg.defaultDemo && cfg.defaultDemo.backend) || "libero";

    bar.innerHTML = `<div class="seg" role="group">${DEMO_BACKENDS.map((backend) => `
      <button type="button" data-backend="${backend.id}">${backend.label}</button>
    `).join("")}</div>`;

    host.innerHTML = DEMO_BACKENDS.map((backend) => `
      <div class="demo-row" data-demo-backend="${backend.id}">
        <div class="demo-row-clips">
          ${DEMO_LEVELS.map((level) => `
            <figure class="demo-cell" data-clip="${clipKey(backend.id, level)}" data-level="${level}">
              <div class="demo-cell-frame is-empty">
                <video muted loop playsinline controls preload="metadata" hidden></video>
                <div class="demo-cell-empty"><b>L${level}</b><span></span></div>
              </div>
              <figcaption><b>L${level}</b><span data-i18n="clip_${backend.id}_l${level}"></span></figcaption>
            </figure>
          `).join("")}
        </div>
      </div>
    `).join("");
    syncDemoEmpty();
    $$("#demo-rows .demo-cell").forEach(mountCell);
    bindSafety();

    function showBackend(id) {
      $$("#demo-bar [data-backend]").forEach((btn) => {
        const on = btn.getAttribute("data-backend") === id;
        btn.classList.toggle("is-on", on);
        btn.setAttribute("aria-pressed", String(on));
      });
      $$("#demo-rows .demo-row").forEach((row) => {
        const on = row.getAttribute("data-demo-backend") === id;
        row.hidden = !on;
        if (!on) {
          row.querySelectorAll("video").forEach((video) => video.pause());
        }
      });
    }

    $$("#demo-bar [data-backend]").forEach((btn) => {
      btn.addEventListener("click", () => showBackend(btn.getAttribute("data-backend")));
    });
    showBackend(initial);
    applyLang(currentLang());

    $$(".level-card[data-level]").forEach((card) => {
      card.style.cursor = "pointer";
      card.addEventListener("click", () => {
        const level = card.getAttribute("data-level");
        $("#demo")?.scrollIntoView({ behavior: "smooth", block: "start" });
        $$(".demo-row:not([hidden]) .demo-cell").forEach((cell) => {
          cell.classList.toggle("is-on", cell.getAttribute("data-level") === level);
        });
      });
    });
  }

  function bindSafety() {
    const host = $("#demo-compare");
    if (!host) return;
    const columns = (cfg.safetyCompare && cfg.safetyCompare.columns) || [];
    if (!columns.length) {
      const wrap = $("#demo-safety");
      if (wrap) wrap.hidden = true;
      return;
    }
    host.innerHTML = columns.map((col) => `
      <div class="demo-compare-col">
        <p class="demo-compare-task" data-i18n="clip_col_${col.id}"></p>
        ${safetyCell(col.hit, "hit")}
        ${safetyCell(col.ok, "ok")}
      </div>`).join("");
    host.querySelectorAll(".demo-cell").forEach(mountCell);
  }

  function safetyCell(key, kind) {
    return `
      <figure class="demo-cell" data-clip="${key}" data-kind="${kind}">
        <div class="demo-cell-frame is-empty">
          <span class="demo-kind" data-i18n="safety_kind_${kind}"></span>
          <video muted loop playsinline controls preload="metadata" hidden></video>
          <div class="demo-cell-empty"><span></span></div>
        </div>
      </figure>`;
  }

  function videoBase() {
    return cfg.videoBase || "assets/videos/";
  }

  function youtubeId(value) {
    if (!value) return "";
    const text = String(value);
    const m = text.match(/(?:youtu\.be\/|v=|embed\/)([\w-]{11})/) || text.match(/^([\w-]{11})$/);
    return m ? m[1] : "";
  }

  function resolveMedia(key) {
    const raw = cfg.videos && cfg.videos[key];
    const base = videoBase();
    const poster = `${base}${key}.jpg?v=c4`;
    if (raw && typeof raw === "object") {
      return {
        src: raw.src || "",
        youtube: youtubeId(raw.youtube || raw.src || ""),
        poster: raw.poster || poster,
      };
    }
    if (typeof raw === "string") {
      const yt = youtubeId(raw);
      return { src: yt ? "" : raw, youtube: yt, poster };
    }
    return { src: `${base}${key}.mp4?v=t1`, youtube: "", poster, auto: true };
  }

  function mountCell(cell) {
    const key = cell.getAttribute("data-clip");
    const frame = cell.querySelector(".demo-cell-frame");
    const video = cell.querySelector("video");
    if (!key || !frame || !video) return;
    const media = resolveMedia(key);

    function showEmpty() {
      video.pause();
      video.removeAttribute("src");
      video.hidden = true;
      frame.classList.add("is-empty");
    }

    if (media.youtube) {
      video.remove();
      frame.classList.remove("is-empty");
      frame.insertAdjacentHTML(
        "afterbegin",
        `<iframe title="${key}" src="https://www.youtube-nocookie.com/embed/${media.youtube}?rel=0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>`,
      );
      return;
    }
    if (!media.src) {
      showEmpty();
      return;
    }
    if (media.poster) video.poster = media.poster;
    video.onerror = showEmpty;
    video.onloadeddata = () => {
      video.hidden = false;
      frame.classList.remove("is-empty");
    };
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) video.play().catch(() => {});
          else video.pause();
        });
      },
      { threshold: 0.4 },
    );
    video.addEventListener("loadeddata", () => io.observe(video), { once: true });
    video.src = media.src;
    video.load();
  }

  function bindCopy() {
    $("#copy-bib")?.addEventListener("click", async () => {
      const text = $("#bibtex")?.innerText || "";
      try {
        await navigator.clipboard.writeText(text);
        const btn = $("#copy-bib");
        const lang = currentLang();
        btn.textContent = dicts[lang].copied;
        setTimeout(() => {
          btn.textContent = dicts[lang].copy;
        }, 1400);
      } catch {
        window.getSelection()?.selectAllChildren($("#bibtex"));
      }
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    const lang = currentLang();
    applyLang(lang);
    bindLinks();
    bindNav();
    bindDemo();
    bindCopy();
    $("#lang-btn")?.addEventListener("click", () => {
      applyLang(document.documentElement.dataset.lang === "zh" ? "en" : "zh");
      bindLinks();
    });
  });
})();
