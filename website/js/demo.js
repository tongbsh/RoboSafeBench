(() => {
  const BACKEND_COPY = {
    en: {
      libero: "LIBERO · Franka on a wooden tabletop. Target is a mug.",
      simplerenv: "SimplerEnv · WidowX on the Bridge metal table. Target is a carrot.",
      robotwin: "RoboTwin · dual-arm workstation. Both arms share a block.",
    },
    zh: {
      libero: "LIBERO · Franka 木桌台面。目标是杯子。",
      simplerenv: "SimplerEnv · WidowX 在 Bridge 金属台。目标是胡萝卜。",
      robotwin: "RoboTwin · 双臂工位。两臂共用一块积木。",
    },
  };

  const LEVEL_COPY = {
    en: {
      0: {
        t: "L0 · Baseline",
        d: "The arm reaches the target and lifts. Nothing else enters the workspace — this is the control every later level is compared against.",
      },
      1: {
        t: "L1 · Sudden appear",
        d: "As the gripper closes in, a marked object occupies the workspace. The policy has to notice and still finish the pick.",
      },
      2: {
        t: "L2 · Sliding obstacle",
        d: "A body crosses the table on a planned path, then settles. Contact during the transit counts; standing still is not evade.",
      },
      3: {
        t: "L3 · Shared reaching",
        d: "A human-hand proxy reaches toward the same object. The policy must share the workspace without contact.",
      },
    },
    zh: {
      0: {
        t: "L0 · 基线",
        d: "手臂伸向目标并抬起。没有额外物体进入工作空间——后面每一档都相对这一条比。",
      },
      1: {
        t: "L1 · 突然出现",
        d: "夹爪靠近时，带条纹的物体占住工作空间。策略必须察觉，并且仍把抓取做完。",
      },
      2: {
        t: "L2 · 滑动障碍",
        d: "物体沿规划路径穿过台面再落定。途中接触计入 CR；站着不动不算规避。",
      },
      3: {
        t: "L3 · 共享伸手",
        d: "人手代理伸向同一目标。策略必须在共享工作空间里避开接触。",
      },
    },
  };

  function clamp(v, a, b) {
    return Math.max(a, Math.min(b, v));
  }

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function ease(t) {
    return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
  }

  function ik(base, target, l1, l2) {
    const dx = target.x - base.x;
    const dy = target.y - base.y;
    const d = clamp(Math.hypot(dx, dy), 8, l1 + l2 - 2);
    const ang = Math.atan2(dy, dx);
    const cosE = clamp((l1 * l1 + d * d - l2 * l2) / (2 * l1 * d), -1, 1);
    const elbowOff = Math.acos(cosE);
    const a1 = ang - elbowOff;
    const elbow = { x: base.x + Math.cos(a1) * l1, y: base.y + Math.sin(a1) * l1 };
    return { elbow, wrist: target };
  }

  function svgEl(name, attrs) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attrs || {}).forEach(([k, v]) => el.setAttribute(k, String(v)));
    return el;
  }

  const SCENE = {
    libero: {
      sky: ["#3a3228", "#1b1713"],
      table: { cx: 400, cy: 300, rx: 250, ry: 58, fill: "url(#wood)", stroke: "#8a6a38", shape: "ellipse" },
      target: { x: 455, y: 248, kind: "mug" },
      arms: [{ base: { x: 168, y: 286 }, l1: 128, l2: 118, side: "left" }],
    },
    simplerenv: {
      sky: ["#2a3340", "#151920"],
      table: { x: 150, y: 268, w: 460, h: 72, fill: "url(#metal)", stroke: "#5c6770", shape: "rect" },
      target: { x: 470, y: 252, kind: "carrot" },
      arms: [{ base: { x: 210, y: 274 }, l1: 108, l2: 96, side: "left" }],
    },
    robotwin: {
      sky: ["#2c241c", "#14110e"],
      table: { x: 90, y: 276, w: 540, h: 78, fill: "url(#walnut)", stroke: "#6f5630", shape: "rect" },
      target: { x: 360, y: 250, kind: "block" },
      arms: [
        { base: { x: 148, y: 292 }, l1: 118, l2: 104, side: "left" },
        { base: { x: 572, y: 292 }, l1: 118, l2: 104, side: "right" },
      ],
    },
  };

  class StageDemo {
    constructor(host, opts = {}) {
      this.host = host;
      this.backend = opts.backend || "libero";
      this.level = opts.level ?? 0;
      this.duration = 7200;
      this.t0 = 0;
      this.raf = 0;
      this.lang = document.documentElement.lang === "zh" ? "zh" : "en";
      this.build();
    }

    build() {
      this.host.innerHTML = "";
      const svg = svgEl("svg", {
        viewBox: "0 0 720 420",
        role: "img",
        "aria-label": "RoboSafeBench disturbance demo",
      });
      const defs = svgEl("defs", {});
      defs.innerHTML = `
        <linearGradient id="wood" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="#d7b07a"/>
          <stop offset="1" stop-color="#b8894d"/>
        </linearGradient>
        <linearGradient id="metal" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="#c5ced6"/>
          <stop offset="1" stop-color="#8b96a1"/>
        </linearGradient>
        <linearGradient id="walnut" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="#c4a06a"/>
          <stop offset="1" stop-color="#8a6a38"/>
        </linearGradient>
        <pattern id="hazard" width="10" height="10" patternUnits="userSpaceOnUse" patternTransform="rotate(-45)">
          <rect width="10" height="10" fill="#c2410c"/>
          <rect width="5" height="10" fill="#f4efe6"/>
        </pattern>
      `;
      svg.appendChild(defs);

      this.bg = svgEl("rect", { width: 720, height: 420, fill: "#3a3228" });
      this.backWall = svgEl("rect", { x: 0, y: 0, width: 720, height: 168, fill: "#2a241c", opacity: "0.55" });
      this.surfaceShadow = svgEl("ellipse", { cx: 408, cy: 330, rx: 246, ry: 40, fill: "#0d0b09", opacity: "0.28" });
      this.surface = svgEl("ellipse", { cx: 400, cy: 300, rx: 250, ry: 58, fill: "url(#wood)" });
      this.surfaceEdge = svgEl("ellipse", {
        cx: 400, cy: 308, rx: 250, ry: 58, fill: "none", stroke: "#8a6a38", "stroke-width": 3,
      });
      this.pedestalL = svgEl("rect", { x: 142, y: 292, width: 52, height: 28, rx: 6, fill: "#2a2622" });
      this.pedestalR = svgEl("rect", { x: 526, y: 292, width: 52, height: 28, rx: 6, fill: "#2a2622", opacity: "0" });

      this.target = svgEl("g", {});
      this.obstacle = svgEl("g", { opacity: "0" });
      this.obstacle.innerHTML = `
        <rect x="-18" y="-18" width="36" height="36" rx="5" fill="url(#hazard)" stroke="#161310" stroke-width="2"/>
      `;
      this.hand = svgEl("g", { opacity: "0" });
      this.hand.innerHTML = `
        <ellipse cx="0" cy="8" rx="16" ry="22" fill="#e8c4a8" stroke="#b08968" stroke-width="2"/>
        <rect x="-7" y="-28" width="8" height="24" rx="4" fill="#e8c4a8"/>
        <rect x="2" y="-26" width="7" height="22" rx="4" fill="#e8c4a8"/>
        <rect x="-16" y="-8" width="8" height="16" rx="4" fill="#e8c4a8"/>
      `;

      this.armNodes = [0, 1].map(() => ({
        link1: svgEl("line", { stroke: "#f4efe6", "stroke-width": 14, "stroke-linecap": "round" }),
        link2: svgEl("line", { stroke: "#e8d9c4", "stroke-width": 11, "stroke-linecap": "round" }),
        jointA: svgEl("circle", { r: 9, fill: "#c2410c" }),
        jointB: svgEl("circle", { r: 7, fill: "#c2410c" }),
        grip: svgEl("g", {}),
      }));

      this.label = svgEl("text", {
        x: 28, y: 36, fill: "#f4efe6", "font-size": 18, "font-weight": 700,
        "font-family": "IBM Plex Sans, sans-serif",
      });
      this.sublabel = svgEl("text", {
        x: 28, y: 58, fill: "#d7cbb8", "font-size": 12,
        "font-family": "IBM Plex Sans, sans-serif",
      });

      svg.append(
        this.bg, this.backWall, this.surfaceShadow, this.surface, this.surfaceEdge,
        this.pedestalL, this.pedestalR, this.target, this.obstacle, this.hand,
      );
      this.armNodes.forEach((arm) => {
        svg.append(arm.link1, arm.link2, arm.jointA, arm.jointB, arm.grip);
      });
      svg.append(this.label, this.sublabel);
      this.host.appendChild(svg);
      this.svg = svg;
    }

    setLang(lang) {
      this.lang = lang === "zh" || lang === "zh-CN" ? "zh" : "en";
      this.syncCopy();
    }

    setBackend(backend) {
      this.backend = backend;
      this.restart();
    }

    setLevel(level) {
      this.level = Number(level);
      this.restart();
    }

    syncCopy() {
      const pack = LEVEL_COPY[this.lang][this.level];
      const extra = BACKEND_COPY[this.lang][this.backend] || "";
      const title = document.getElementById("demo-title");
      const body = document.getElementById("demo-body");
      const chips = document.getElementById("demo-chips");
      if (title) title.textContent = pack.t;
      if (body) body.textContent = `${extra} ${pack.d}`;
      if (chips) {
        chips.innerHTML = `
          <span class="chip">${this.backend}</span>
          <span class="chip">L${this.level}</span>
        `;
      }
      this.label.textContent = `RoboSafeBench · ${this.backend.toUpperCase()}`;
      this.sublabel.textContent = extra;
    }

    restart() {
      this.t0 = performance.now();
      this.syncCopy();
      if (!this.raf) this.loop();
    }

    loop = () => {
      const u = ((performance.now() - this.t0) % this.duration) / this.duration;
      this.draw(u);
      this.raf = requestAnimationFrame(this.loop);
    };

    scene() {
      return SCENE[this.backend] || SCENE.libero;
    }

    setTargetArt(kind) {
      if (this._targetKind === kind) return;
      this._targetKind = kind;
      if (kind === "carrot") {
        this.target.innerHTML = `
          <ellipse cx="0" cy="10" rx="28" ry="8" fill="#d8dde3" stroke="#8b96a1" stroke-width="2"/>
          <rect x="-6" y="-22" width="12" height="28" rx="6" fill="#e07a2f"/>
          <path d="M-4 -22 q4 -12 8 0" fill="#3f7d4e"/>
        `;
      } else if (kind === "block") {
        this.target.innerHTML = `
          <rect x="-16" y="-16" width="32" height="32" rx="4" fill="#c2410c" stroke="#161310" stroke-width="2"/>
          <rect x="-10" y="-10" width="20" height="8" rx="2" fill="#f4efe6" opacity="0.35"/>
        `;
      } else {
        this.target.innerHTML = `
          <rect x="-14" y="-22" width="28" height="28" rx="4" fill="#3d5a80"/>
          <path d="M14 -12 h10 a8 8 0 0 1 0 16 h-10" fill="none" stroke="#d7e3ef" stroke-width="3"/>
          <rect x="-10" y="-18" width="20" height="8" rx="2" fill="#7aa2c7"/>
        `;
      }
    }

    layoutTable(scene) {
      const sky = scene.sky;
      this.bg.setAttribute("fill", sky[0]);
      this.backWall.setAttribute("fill", sky[1]);
      const t = scene.table;
      if (t.shape === "rect") {
        if (this.surface.tagName !== "rect") {
          const next = svgEl("rect", { rx: 10 });
          this.surface.replaceWith(next);
          this.surface = next;
          const edge = svgEl("rect", { fill: "none", "stroke-width": 3, rx: 10 });
          this.surfaceEdge.replaceWith(edge);
          this.surfaceEdge = edge;
        }
        this.surface.setAttribute("x", t.x);
        this.surface.setAttribute("y", t.y);
        this.surface.setAttribute("width", t.w);
        this.surface.setAttribute("height", t.h);
        this.surfaceEdge.setAttribute("x", t.x);
        this.surfaceEdge.setAttribute("y", t.y + 6);
        this.surfaceEdge.setAttribute("width", t.w);
        this.surfaceEdge.setAttribute("height", t.h);
        this.surfaceShadow.setAttribute("cx", t.x + t.w / 2);
        this.surfaceShadow.setAttribute("cy", t.y + t.h + 16);
        this.surfaceShadow.setAttribute("rx", t.w / 2 - 10);
      } else {
        if (this.surface.tagName !== "ellipse") {
          const next = svgEl("ellipse", {});
          this.surface.replaceWith(next);
          this.surface = next;
          const edge = svgEl("ellipse", { fill: "none", "stroke-width": 3 });
          this.surfaceEdge.replaceWith(edge);
          this.surfaceEdge = edge;
        }
        this.surface.setAttribute("cx", t.cx);
        this.surface.setAttribute("cy", t.cy);
        this.surface.setAttribute("rx", t.rx);
        this.surface.setAttribute("ry", t.ry);
        this.surfaceEdge.setAttribute("cx", t.cx);
        this.surfaceEdge.setAttribute("cy", t.cy + 8);
        this.surfaceEdge.setAttribute("rx", t.rx);
        this.surfaceEdge.setAttribute("ry", t.ry);
        this.surfaceShadow.setAttribute("cx", t.cx + 8);
        this.surfaceShadow.setAttribute("cy", t.cy + 18);
        this.surfaceShadow.setAttribute("rx", t.rx - 4);
      }
      this.surface.setAttribute("fill", t.fill);
      this.surfaceEdge.setAttribute("stroke", t.stroke);

      const dual = scene.arms.length > 1;
      this.pedestalL.setAttribute("opacity", "1");
      this.pedestalL.setAttribute("x", scene.arms[0].base.x - 26);
      this.pedestalL.setAttribute("y", scene.arms[0].base.y + 6);
      this.pedestalR.setAttribute("opacity", dual ? "1" : "0");
      if (dual) {
        this.pedestalR.setAttribute("x", scene.arms[1].base.x - 26);
        this.pedestalR.setAttribute("y", scene.arms[1].base.y + 6);
      }
    }

    drawArm(node, spec, ee, u) {
      const arm = ik(spec.base, ee, spec.l1, spec.l2);
      node.link1.setAttribute("opacity", "1");
      node.link2.setAttribute("opacity", "1");
      node.jointA.setAttribute("opacity", "1");
      node.jointB.setAttribute("opacity", "1");
      node.grip.setAttribute("opacity", "1");
      node.link1.setAttribute("x1", spec.base.x);
      node.link1.setAttribute("y1", spec.base.y);
      node.link1.setAttribute("x2", arm.elbow.x);
      node.link1.setAttribute("y2", arm.elbow.y);
      node.link2.setAttribute("x1", arm.elbow.x);
      node.link2.setAttribute("y1", arm.elbow.y);
      node.link2.setAttribute("x2", ee.x);
      node.link2.setAttribute("y2", ee.y);
      node.jointA.setAttribute("cx", spec.base.x);
      node.jointA.setAttribute("cy", spec.base.y);
      node.jointB.setAttribute("cx", arm.elbow.x);
      node.jointB.setAttribute("cy", arm.elbow.y);
      const open = u > 0.62 && u < 0.86 ? 4 : 10;
      const dir = spec.side === "right" ? -1 : 1;
      node.grip.innerHTML = "";
      node.grip.append(
        svgEl("line", {
          x1: ee.x - 2 * dir, y1: ee.y - 2, x2: ee.x - open * dir, y2: ee.y + 14,
          stroke: "#f4efe6", "stroke-width": 4, "stroke-linecap": "round",
        }),
        svgEl("line", {
          x1: ee.x + 2 * dir, y1: ee.y - 2, x2: ee.x + open * dir, y2: ee.y + 14,
          stroke: "#f4efe6", "stroke-width": 4, "stroke-linecap": "round",
        }),
      );
    }

    hideArm(node) {
      ["link1", "link2", "jointA", "jointB", "grip"].forEach((k) => node[k].setAttribute("opacity", "0"));
    }

    evade(ee, u) {
      if (this.level >= 1 && u > 0.42 && u < 0.62) {
        ee.y -= 18;
        ee.x -= 22;
      }
      if (this.level >= 2 && u > 0.35 && u < 0.7) {
        ee.y -= 28;
        ee.x -= 8;
      }
      if (this.level === 3 && u > 0.38 && u < 0.72) {
        ee.x -= 30;
        ee.y -= 10;
      }
      return ee;
    }

    draw(u) {
      const scene = this.scene();
      this.layoutTable(scene);
      this.setTargetArt(scene.target.kind);
      const mug = scene.target;
      this.target.setAttribute("transform", `translate(${mug.x} ${mug.y})`);

      scene.arms.forEach((spec, i) => {
        const inward = spec.side === "right" ? 8 : -8;
        let ee = {
          x: lerp(spec.base.x + (spec.side === "right" ? -70 : 82), mug.x + inward, ease(clamp((u - 0.08) / 0.5, 0, 1))),
          y: lerp(spec.base.y - 76, mug.y - 6, ease(clamp((u - 0.08) / 0.5, 0, 1))),
        };
        ee = this.evade({ ...ee }, u);
        if (spec.side === "right") {
          if (this.level >= 1 && u > 0.42 && u < 0.62) ee.x += 44;
          if (this.level === 3 && u > 0.38 && u < 0.72) ee.x += 60;
        }
        this.drawArm(this.armNodes[i], spec, ee, u);
      });
      if (scene.arms.length < 2) this.hideArm(this.armNodes[1]);

      this.drawHazard(u, mug);
    }

    drawHazard(u, mug) {
      this.obstacle.setAttribute("opacity", "0");
      this.hand.setAttribute("opacity", "0");

      if (this.level === 1) {
        const drop = ease(clamp((u - 0.28) / 0.18, 0, 1));
        const x = mug.x - 70;
        const y = lerp(40, mug.y - 8, drop);
        this.obstacle.setAttribute("opacity", String(u > 0.26 ? 1 : 0));
        this.obstacle.setAttribute("transform", `translate(${x} ${y})`);
      }

      if (this.level === 2) {
        const slide = ease(clamp((u - 0.18) / 0.42, 0, 1));
        const x = lerp(640, mug.x - 56, slide);
        const y = mug.y - 4;
        this.obstacle.setAttribute("opacity", "1");
        this.obstacle.setAttribute("transform", `translate(${x} ${y}) rotate(${slide * 18})`);
      }

      if (this.level === 3) {
        const reach = ease(clamp((u - 0.2) / 0.35, 0, 1));
        const back = ease(clamp((u - 0.68) / 0.22, 0, 1));
        const x = lerp(mug.x - 150, mug.x - 36, reach * (1 - back));
        const y = lerp(120, mug.y - 18, reach);
        this.hand.setAttribute("opacity", "1");
        this.hand.setAttribute("transform", `translate(${x} ${y}) rotate(${12 - back * 20})`);
      }
    }
  }

  window.RoboSafeDemo = { StageDemo, LEVEL_COPY, BACKEND_COPY };
})();
