'use strict';
(() => {
  const el = id => document.getElementById(id);
  const player = el('video-player'), timeline = el('video-timeline');
  const input = el('frame-number'), slider = el('frame-slider');
  const preview = el('selected-preview'), status = el('video-status');
  let source = null, selected = 0, version = 0, localUrl = null, busy = false, timer;
  let previewVersion = 0;
  let cancelPreviewLoad = null;
  let selection = null, selectionMode = 'video', pendingSeek = null, inputDirty = false;
  let presentedTime = null;
  const snapshot = image => {
    const canvas = document.createElement('canvas');
    canvas.width = image.videoWidth || image.naturalWidth;
    canvas.height = image.videoHeight || image.naturalHeight;
    if (!canvas.width || !canvas.height) throw new Error('Ảnh nguồn chưa sẵn sàng.');
    canvas.getContext('2d').drawImage(image, 0, 0);
    return canvas.toDataURL('image/png');
  };
  let uploadRequest = null;
  const stageNames = ['Hiệu chỉnh camera', 'Phát hiện và theo dõi người', 'Pose 2D', 'Pose 3D trên sân', 'Nhận diện đội', 'Bóng và người chạm bóng', 'Ngữ cảnh trận đấu', 'Dựng đường tham chiếu', 'Phân loại vị trí việt vị'];
  let progressTicket = 0, progressTimer, elapsedTimer;
  function renderProgress(data) {
    let done = 0;
    el('progress-stages').replaceChildren();
    stageNames.forEach((name, i) => {
      let value = data.stages?.[i + 1] || 'waiting';
      if (data.status === 'failed' && data.stage === i + 1 && value === 'running') value = 'failed';
      if (value === 'completed') done++;
      const row = document.createElement('li'); row.dataset.status = value;
      const label = document.createElement('span'); label.textContent = 'S' + (i + 1) + ' · ' + name;
      const badge = document.createElement('strong');
      badge.textContent = {waiting: 'Chờ', running: 'Đang chạy…', completed: 'Hoàn tất', failed: 'Lỗi'}[value] || value;
      row.append(label, badge); el('progress-stages').append(row);
    });
    el('progress-bar').value = done;
    el('progress-summary').textContent = done + ' / 9 stage hoàn tất';
    el('progress-title').textContent = data.status === 'completed' ? 'Xử lý hoàn tất' :
      data.status === 'failed' ? 'Xử lý thất bại' + (data.stage ? ' tại Stage ' + data.stage : '') :
      data.stages?.[data.stage] === 'running' ? 'Đang chạy S' + data.stage + ' · ' + stageNames[data.stage - 1] :
      data.message || 'Đang gửi yêu cầu và chuẩn bị ảnh nguồn…';
    if (data.error) {
      el('progress-error-wrap').hidden = false;
      el('progress-error').textContent = data.error;
    }
  }
  function startProgress(requestId) {
    const ticket = ++progressTicket, start = Date.now();
    let finished = false, unavailable = false, failures = 0, controller = null;
    clearTimeout(progressTimer); clearInterval(elapsedTimer);
    el('analysis-progress').hidden = false;
    el('progress-error-wrap').hidden = true; el('progress-error').textContent = '';
    renderProgress({status: 'pending'});
    const clock = () => {
      const seconds = Math.floor((Date.now() - start) / 1000);
      el('progress-elapsed').textContent = Math.floor(seconds / 60) + ' phút ' + seconds % 60 + ' giây';
    };
    clock(); elapsedTimer = setInterval(clock, 1000);
    async function poll() {
      if (ticket !== progressTicket || unavailable || finished) return;
      controller = new AbortController();
      const timeout = setTimeout(() => controller?.abort(), 5000);
      try {
        const response = await fetch('/api/analysis_progress?request_id=' + encodeURIComponent(requestId), {cache: 'no-store', signal: controller.signal});
        if (ticket !== progressTicket) return;
        if (response.status >= 400 && response.status < 500) {
          unavailable = true;
          el('progress-title').textContent = 'Không có API tiến độ khả dụng (HTTP ' + response.status + ')';
          el('progress-summary').textContent = 'Nếu pipeline đang chạy, hãy chờ kết quả. Sau đó dừng backend, chạy lại bản mới và tải trang bằng Ctrl+F5.';
          return;
        }
        if (!response.ok) throw new Error('Không đọc được tiến độ (HTTP ' + response.status + ').');
        const data = await response.json(); if (ticket !== progressTicket) return;
        if (!['pending', 'running', 'completed', 'failed'].includes(data.status)) throw new Error('Phản hồi tiến độ không đúng định dạng.');
        if (data.status === 'pending' && Date.now() - start > 30000) {
          unavailable = true;
          el('progress-title').textContent = 'Máy chủ chưa tìm thấy tiến độ của lần chạy này';
          el('progress-summary').textContent = 'Đã dừng cập nhật tiến độ. Yêu cầu phân tích vẫn đang chờ kết quả.';
          return;
        }
        failures = 0;
        renderProgress(data);
        finished = ['completed', 'failed'].includes(data.status);
        if (finished) clearInterval(elapsedTimer);
      } catch (error) {
        if (ticket !== progressTicket) return;
        failures++;
        unavailable = failures >= 4;
        el('progress-title').textContent = unavailable ? 'Đã dừng cập nhật tiến độ sau 4 lần lỗi' : 'Mất cập nhật tiến độ · thử lại ' + failures + '/4';
        el('progress-error-wrap').hidden = false;
        el('progress-error').textContent = (error.name === 'AbortError' ? 'Máy chủ không trả tiến độ trong 5 giây.' : error.message) + '\nYêu cầu phân tích vẫn chờ kết quả; lỗi cập nhật không đồng nghĩa pipeline đã thất bại.';
      } finally {
        clearTimeout(timeout);
        controller = null;
      }
      if (ticket === progressTicket && !finished && !unavailable) progressTimer = setTimeout(poll, failures ? Math.min(1000 * 2 ** failures, 8000) : 1000);
    }
    poll();
    return async (error) => {
      ++progressTicket; clearTimeout(progressTimer); clearInterval(elapsedTimer);
      controller?.abort();
      if (!error) {
        el('progress-error-wrap').hidden = true;
        renderProgress({status: 'completed', stages: Object.fromEntries(stageNames.map((_, i) => [i + 1, 'completed']))});
      }
      if (error && !finished) {
        el('progress-title').textContent = 'Không nhận được kết quả xử lý';
        el('progress-error-wrap').hidden = false; el('progress-error').textContent = error.message;
      }
    };
  }
  function uploadVideo(file, ticket) {
    return new Promise((resolve, reject) => {
      const request = new XMLHttpRequest();
      uploadRequest = request;
      request.open('POST', '/api/video_source');
      request.upload.onprogress = event => {
        if (ticket !== version) return;
        const total = event.lengthComputable ? event.total : file.size;
        status.textContent = 'Đang tải video: ' + Math.min(100, Math.round(event.loaded / total * 100)) + '% (' +
          (event.loaded / 1024 ** 2).toFixed(0) + ' / ' + (total / 1024 ** 2).toFixed(0) + ' MB)';
      };
      request.upload.onload = () => {
        if (ticket === version) status.textContent = 'Đã tải xong. Máy chủ đang đọc thông tin video…';
      };
      request.onload = () => {
        if (request.status < 200 || request.status >= 300) {
          reject(new Error(request.responseText || 'Máy chủ từ chối video (HTTP ' + request.status + ').')); return;
        }
        try { resolve(JSON.parse(request.responseText)); }
        catch (_) { reject(new Error('Phản hồi video không hợp lệ. Khởi động lại máy chủ rồi tải lại trang.')); }
      };
      request.onerror = () => reject(new Error('Kết nối bị ngắt khi tải video. Kiểm tra máy chủ đã khởi động lại với bản sửa mới và xem dòng “Video upload failed” trong terminal.'));
      request.onabort = () => reject(new Error('Đã hủy tải video trước.'));
      request.send(file);
    });
  }
  const frameUrl = (n, thumb = false) => '/api/video_frame.jpg?' + new URLSearchParams({video_id: source.video_id, frame: n, thumb: thumb ? '1' : '0'});
  function controls() {
    document.querySelectorAll('[data-frame-control]').forEach(node => node.disabled = !source || busy);
    el('video-upload').disabled = busy;
    player.controls = !busy;
  }
  function displayFrame(n, time = n / source.fps) {
    selected = Math.max(0, Math.min(source.frame_count - 1, Math.round(n)));
    if (!inputDirty) input.value = selected;
    slider.value = selected;
    el('video-frame').textContent = selected;
    el('video-time').textContent = time.toFixed(3);
    el('timeline-playhead').style.left = (source.frame_count > 1 ? selected / (source.frame_count - 1) * 100 : 0) + '%';
    slider.setAttribute('aria-valuetext', 'Frame ' + selected);
  }
  function capturePausedFrame() {
    if (!source || busy || player.seeking || player.readyState < 2) return false;
    const time = presentedTime != null && Math.abs(presentedTime - player.currentTime) <= 2 / source.fps
      ? presentedTime : player.currentTime;
    ++previewVersion;
    clearTimeout(timer); cancelPreviewLoad?.(); cancelPreviewLoad = null;
    selectionMode = 'video';
    displayFrame(Math.floor(time * source.fps + 1e-5), time);
    const image = snapshot(player);
    selection = {frame: selected, time, image, kind: 'DISPLAYED_VIDEO_FRAME', ready: Promise.resolve(true)};
    preview.src = image; preview.hidden = false;
    el('preview-caption').textContent = 'Ảnh video đang dừng · pipeline xử lý chính ảnh này';
    return true;
  }
  function select(n, debounce = false) {
    if (!source || busy) return;
    inputDirty = false;
    selectionMode = 'indexed';
    displayFrame(n);
    const chosen = selected, ticket = ++previewVersion;
    clearTimeout(timer); cancelPreviewLoad?.(); cancelPreviewLoad = null;
    const item = {frame: chosen, time: chosen / source.fps, kind: 'INDEXED_FRAME_PREVIEW', image: null};
    selection = item;
    player.pause();
    // Seek inside the frame, not on a floating-point frame boundary.
    const duration = Number.isFinite(player.duration) ? player.duration : source.frame_count / source.fps;
    pendingSeek = Math.min((chosen + 0.5) / source.fps, Math.max(0, duration - 0.001));
    player.currentTime = pendingSeek;
    preview.hidden = true;
    el('preview-caption').textContent = 'Đang lấy đúng frame ' + chosen + '…';
    item.ready = new Promise(resolve => {
      cancelPreviewLoad = () => resolve(false);
      const load = () => {
      const image = new Image();
      image.onload = () => {
        if (ticket !== previewVersion) { resolve(false); return; }
        try {
          item.image = snapshot(image);
          preview.src = item.image; preview.hidden = false;
          el('preview-caption').textContent = 'Frame ' + chosen + ' · pipeline xử lý chính ảnh này';
          resolve(true);
        } catch (error) { status.textContent = error.message; resolve(false); }
      };
      image.onerror = () => {
        if (ticket === previewVersion) status.textContent = 'Không lấy được frame. Chọn lại frame hoặc tải lại video.';
        resolve(false);
      };
      image.src = frameUrl(chosen);
      };
      if (debounce) timer = setTimeout(load, 120); else load();
    });
  }
  function commit() {
    if (!source) return false;
    const n = Number(input.value);
    if (input.value.trim() === '' || !Number.isInteger(n) || n < 0 || n >= source.frame_count) {
      input.setCustomValidity('Nhập số nguyên từ 0 đến ' + (source.frame_count - 1));
      input.reportValidity(); return false;
    }
    input.setCustomValidity(''); select(n); return true;
  }
  el('video-upload').addEventListener('change', async event => {
    const file = event.target.files[0]; if (!file) return;
    const ticket = ++version;
    if (uploadRequest) { uploadRequest.abort(); uploadRequest = null; }
    source = null; selection = null; pendingSeek = null; presentedTime = null; inputDirty = false; selectionMode = 'video'; ++previewVersion;
    clearTimeout(timer); cancelPreviewLoad?.(); cancelPreviewLoad = null; controls();
    preview.hidden = true; timeline.replaceChildren();
    el('video-player-wrap').style.display = 'block';
    if (localUrl) URL.revokeObjectURL(localUrl);
    player.src = localUrl = URL.createObjectURL(file);
    status.textContent = 'Đang tải và đọc thông tin video…';
    try {
      const response = await fetch('/api/video_source?size=' + file.size);
      if (response.status === 404) throw new Error('Máy chủ đang chạy bản cũ. Dừng web demo, chạy lại rồi tải trang bằng Ctrl+F5.');
      if (!response.ok) throw new Error(await response.text());
      if (ticket !== version) return;
      const data = await uploadVideo(file, ticket); if (ticket !== version) return;
      source = data;
      input.max = slider.max = source.frame_count - 1;
      input.setCustomValidity('');
      status.textContent = source.frame_count.toLocaleString('vi-VN') + ' frame · ' + source.fps.toFixed(3) + ' fps · Đánh số từ 0';
      const count = Math.min(16, source.frame_count);
      for (let i = 0; i < count; i++) {
        const n = Math.round(i * (source.frame_count - 1) / Math.max(1, count - 1));
        const cell = document.createElement('div'); cell.className = 'timeline-cell';
        const img = document.createElement('img'); img.src = frameUrl(n, true); img.alt = 'Frame ' + n; img.loading = 'lazy'; img.draggable = false;
        const caption = document.createElement('span'); caption.textContent = '#' + n;
        cell.append(img, caption); timeline.append(cell);
      }
      controls(); select(0);
    } catch (error) { if (ticket === version) {
      status.textContent = error instanceof TypeError ? 'Không kết nối được máy chủ. Hãy khởi động lại web demo và tải lại trang bằng Ctrl+F5.' : error.message;
      controls();
    } }
  });
  slider.addEventListener('input', () => select(Number(slider.value), true));
  input.addEventListener('input', () => { inputDirty = true; input.setCustomValidity(''); });
  input.addEventListener('change', commit);
  input.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); commit(); } });
  el('frame-prev').addEventListener('click', () => select(selected - 1));
  el('frame-next').addEventListener('click', () => select(selected + 1));
  if (player.requestVideoFrameCallback) {
    const presented = (_, metadata) => {
      presentedTime = metadata.mediaTime;
      if (source && !busy && !player.paused && !inputDirty) displayFrame(Math.floor(metadata.mediaTime * source.fps + 1e-5), metadata.mediaTime);
      player.requestVideoFrameCallback(presented);
    };
    player.requestVideoFrameCallback(presented);
  }
  player.addEventListener('play', () => {
    if (busy) { player.pause(); return; }
    selectionMode = 'video'; pendingSeek = null; selection = null; ++previewVersion;
    preview.hidden = true;
  });
  player.addEventListener('timeupdate', () => {
    if (source && !busy && !player.paused && !player.requestVideoFrameCallback && !inputDirty)
      displayFrame(Math.floor(player.currentTime * source.fps + 1e-5), player.currentTime);
  });
  player.addEventListener('pause', () => {
    if (selectionMode === 'video' && !inputDirty) capturePausedFrame();
  });
  player.addEventListener('seeking', () => {
    if (busy) return;
    if (pendingSeek != null && Math.abs(player.currentTime - pendingSeek) < 0.0001) return;
    pendingSeek = null; selectionMode = 'video'; selection = null; presentedTime = null; ++previewVersion;
    preview.hidden = true;
  });
  player.addEventListener('seeked', () => {
    if (!source || busy) return;
    if (pendingSeek != null && Math.abs(player.currentTime - pendingSeek) < 0.0001) {
      pendingSeek = null; return;
    }
    pendingSeek = null;
    if (player.paused && !inputDirty) capturePausedFrame();
  });
  el('analyze-btn').addEventListener('click', async () => {
    if (!source || busy) return;
    if (inputDirty && !commit()) return;
    if (selectionMode === 'video') {
      if (player.seeking) { status.textContent = 'Đang tua video. Chờ ảnh dừng hiển thị rồi bấm xử lý.'; return; }
      player.pause();
      if (!capturePausedFrame()) { status.textContent = 'Ảnh video chưa sẵn sàng.'; return; }
    }
    const chosen = selection;
    if (!chosen) return;
    busy = true; controls();
    if (!await chosen.ready || !chosen.image || selection !== chosen) {
      busy = false; controls(); status.textContent = 'Chưa có ảnh nguồn chính xác. Hãy chọn lại frame.'; return;
    }
    const requestId = crypto.randomUUID();
    const payload = {request_id: requestId, video_id: source.video_id, estimated_frame: chosen.frame,
      time_sec: chosen.time, source_kind: chosen.kind, image: chosen.image};
    status.textContent = 'Đang xử lý ảnh đã chọn tại frame ' + chosen.frame + '…';
    const stopProgress = startProgress(requestId);
    let failure = null;
    const button = el('analyze-btn'); button.textContent = 'Đang xử lý frame ' + chosen.frame + '…';
    try {
      const response = await fetch('/api/analyze_live', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
      if (!response.ok) throw new Error(await response.text());
      const result = await response.json(); if (result.status !== 'ok') throw new Error(result.error || 'Phân tích không hoàn tất.');
      await init();
      ['stages', 'workspace-section', 'bottom-grid-section'].forEach(id => el(id).style.display = '');
      status.textContent = 'Đã phân tích frame ' + payload.estimated_frame + '. Kết quả ở bên dưới.';
    } catch (error) { failure = error; status.textContent = 'Không thể phân tích: ' + error.message; }
    finally { await stopProgress(failure); busy = false; controls(); button.textContent = 'Chạy pipeline frame đã chọn'; }
  });
  controls();
})();
