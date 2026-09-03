# Demo videos

Demo 区**只播执行视频**，不再回退到插画。缺文件时只显示一行文件名占位。

## Drop-in files

```text
website/assets/videos/libero_l0.mp4
website/assets/videos/libero_l1.mp4
website/assets/videos/libero_l2.mp4
website/assets/videos/libero_l3.mp4
website/assets/videos/simplerenv_l3.mp4
website/assets/videos/robotwin_l3.mp4
```

任意 `{backend}_l{level}.mp4` 都可以。或在 `site.config.js` 的 `videos` 里写 URL / YouTube id。

```js
videos: {
  libero_l3: "assets/videos/libero_l3.mp4",
  simplerenv_l3: "https://…/simplerenv_l3.mp4",
  robotwin_l3: "dQw4w9WgXcQ",
}
```

Prefer H.264 + AAC, 16:9. 大文件走 Release / Hugging Face / B 站，再把链接写进 `videos`。
