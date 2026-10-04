// Exercise actual editor event handlers without launching models or a browser.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

async function editor() {
  const elements = new Map(), images = [], requests = [];
  class Element {
    constructor() { this.events = {}; this.style = {}; this.dataset = {}; this.value = '0'; this.children = []; }
    addEventListener(name, fn) { this.events[name] = fn; }
    emit(name, arg = {}) { return this.events[name]?.({target: this, ...arg}); }
    setAttribute() {} setCustomValidity() {} reportValidity() {}
    replaceChildren() { this.children = []; } append(...values) { this.children.push(...values); }
  }
  const get = id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
  const video = get('video-player');
  Object.assign(video, {paused: true, seeking: false, readyState: 4, videoWidth: 32, videoHeight: 24,
    duration: 10000, _time: 0, seeks: 0, marker: 'paused-pixels'});
  Object.defineProperty(video, 'currentTime', {get() { return this._time; }, set(n) { this._time = n; this.seeks++; }});
  video.pause = () => { if (!video.paused) { video.paused = true; video.emit('pause'); } };
  const context = {
    document: {getElementById: get, querySelectorAll: () => [], createElement: tag => {
      if (tag !== 'canvas') return new Element();
      let marker;
      return {getContext: () => ({drawImage: image => { marker = image.marker; }}), toDataURL: () => 'data:image/png;base64,' + marker};
    }},
    Image: class { constructor() { this.naturalWidth = 32; this.naturalHeight = 24; images.push(this); } },
    XMLHttpRequest: class { constructor() { this.upload = {}; } open() {} send() {
      this.status = 200; this.responseText = JSON.stringify({video_id: 'video', frame_count: 300000, fps: 30}); this.onload();
    } abort() {} },
    URL: class extends URL { static createObjectURL() { return 'blob:test'; } static revokeObjectURL() {} }, URLSearchParams,
    crypto: {randomUUID: () => 'test'}, AbortController, console,
    setTimeout: () => 1, clearTimeout() {}, setInterval: () => 1, clearInterval() {}, init: async () => {},
    fetch: async (url, options) => {
      if (url === '/api/analyze_live') { requests.push(JSON.parse(options.body)); return {ok: true, json: async () => ({status:'ok'})}; }
      return {ok:true, status:200, json:async () => ({status:'completed', stages:{}})};
    },
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../static/video_editor.js'), 'utf8'), context);
  get('video-upload').files = [{size: 10}];
  await get('video-upload').emit('change');
  function load(image = images.at(-1), marker = 'indexed-pixels') { image.marker = marker; image.onload(); }
  load(); video.emit('seeked');
  return {get, video, images, requests, load};
}

(async () => {
  // Pausing between timeupdate events must use displayed pixels, never seek backwards.
  const a = await editor();
  a.video.paused = false; a.video.emit('play');
  a.video._time = 38273 / 30; a.video.emit('timeupdate');
  a.video._time = 38276 / 30; a.video.marker = 'ball-visible'; a.video.pause();
  const before = a.video.seeks;
  await a.get('analyze-btn').emit('click');
  assert.equal(a.video.seeks, before);
  assert.equal(a.requests[0].estimated_frame, 38276);
  assert.equal(a.requests[0].image, 'data:image/png;base64,ball-visible');
  assert.equal(a.requests[0].source_kind, 'DISPLAYED_VIDEO_FRAME');

  // Explicit frame input waits for its preview; seek events may not change its index.
  const b = await editor();
  b.get('frame-number').value = '38276'; b.get('frame-number').emit('input'); b.get('frame-number').emit('change');
  b.video.emit('seeked');
  const running = b.get('analyze-btn').emit('click');
  assert.equal(b.requests.length, 0);
  b.load(undefined, 'exact-38276'); await running;
  assert.equal(b.requests[0].estimated_frame, 38276);
  assert.equal(b.requests[0].image, 'data:image/png;base64,exact-38276');

  // An old preview arriving last must not replace the more recent selection.
  const c = await editor();
  c.get('frame-next').emit('click'); const stale = c.images.at(-1);
  c.get('frame-next').emit('click'); c.load(undefined, 'frame-2'); c.load(stale, 'frame-1');
  await c.get('analyze-btn').emit('click');
  assert.equal(c.requests[0].estimated_frame, 2);
  assert.equal(c.requests[0].image, 'data:image/png;base64,frame-2');

  // Native seek still in progress must not send the pre-seek frame.
  const d = await editor();
  d.video._time = 400; d.video.seeking = true; d.video.emit('seeking');
  await d.get('analyze-btn').emit('click');
  assert.equal(d.requests.length, 0);
  console.log('4 frame-selection event regressions passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
