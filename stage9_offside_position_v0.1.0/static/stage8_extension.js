'use strict';

const stage8Description = 'Xếp hạng đối phương theo hướng tấn công, chọn đối phương áp chót và so sánh với biên dọc của bóng để dựng mốc tham chiếu.';
const stage9Description = 'So sánh điểm hợp lệ gần khung thành nhất của từng cầu thủ tấn công với mốc Stage 8 và điều kiện nửa sân đối phương.';
const originalSelectStage = selectStage;

selectStage = function (id) {
  if (id !== 8 && id !== 9) {
    originalSelectStage(id);
    return;
  }
  chosenStage = id;
  layers = new Set(id === 9 ? [6, 7, 8, 9] : [2, 5, 6, 7, 8]);
  raw = false;
  document.querySelectorAll('.stage-card').forEach(function (el) {
    const active = Number(el.dataset.stage) === id;
    el.classList.toggle('active', active);
    el.setAttribute('aria-pressed', String(active));
  });
  const stage = state.stages[id - 1];
  document.getElementById('detail-title').textContent = 'S' + id + ' · ' + stage.title;
  document.getElementById('detail-desc').textContent = id === 9 ? stage9Description : stage8Description;
  document.getElementById('stage-detail').textContent = stage.summary + (
    id === 9
      ? (state.offside_position?.mode === 'DEFENDER_REFERENCE' ? ' · Phân loại theo hậu vệ áp chót, chưa xét bóng.'
        : state.offside_position?.mode === 'TENTATIVE_CONTACT_REFERENCE' ? ' · Phân loại theo vạch S8; người chạm bóng mới được xác định tạm thời.'
        : ' · Chỉ là phân loại vị trí; không kết luận lỗi việt vị.')
      : state.reference?.reference_only ? ' · Vạch hậu vệ áp chót tham khảo; chưa xác nhận bóng/người chạm bóng.'
      : stage.status === 'DEGRADED' ? ' · Mốc tạm thời; xem giới hạn dữ liệu bên dưới.' : ''
  );
  if (id === 8 && state.second_last_opponent?.track_id) {
    document.getElementById('stage-detail').textContent += ' · Hậu vệ áp chót: ' + state.second_last_opponent.track_id.replace('track_', '#');
  }
  document.getElementById('provenance').textContent = JSON.stringify({
    nguon: stage.source,
    trang_thai: stage.status,
    chi_tiet: stage.detail
  }, null, 2);
  refreshFrame();
};

refreshFrame = function () {
  const params = new URLSearchParams({
    overlay: raw ? '0' : '1',
    labels: document.getElementById('labels').checked ? '1' : '0',
    revision: frameRevision
  });
  for (const id of selectedTracks) params.append('track', id);
  for (let i = 1; i <= state.stages.length; i += 1) {
    params.set('s' + i, layers.has(i) ? '1' : '0');
  }
  document.getElementById('frame').src = '/api/frame.jpg?' + params;
  document.getElementById('raw').textContent = raw ? 'Trở lại các lớp' : 'Xem ảnh gốc';
  document.getElementById('raw').setAttribute('aria-pressed', String(raw));
  document.getElementById('view-label').textContent = raw
    ? 'Ảnh gốc từ replay'
    : 'Lớp đang bật: ' + ([...layers].sort().map(function (i) { return 'S' + i; }).join(' + ') || 'không có');
  document.querySelectorAll('[data-layer]').forEach(function (el) {
    const active = layers.has(Number(el.dataset.layer));
    el.classList.toggle('active', active);
    el.setAttribute('aria-pressed', String(active));
  });
};

// Keep every raw track in /api/state, but remove downstream-noise rows from
// the visible roster. Stage 2 can still be inspected in full on its own card.
function updateStage9Roster() {
  if (!state || !Array.isArray(state.tracks)) return;
  let visible = 0;
  document.querySelectorAll('#tracks [data-row]').forEach(function (element) {
    const row = state.tracks.find(function (candidate) {
      return candidate.track_id === element.dataset.row;
    });
    const useful = row && (
      (row.pose2d && row.pose2d.length) || row.root_world_m || row.toucher
      || row.group === 'attackers' || row.group === 'opponents' || row.offside_label
    );
    if (row && element.children.length < 6) {
      const cell = document.createElement('td');
      const labels = {
        OFFSIDE_POSITION: 'OFFSIDE',
        ONSIDE: 'ONSIDE',
        TOUCHER_EXCLUDED: 'PASSER',
        UNAVAILABLE: 'Chưa rõ'
      };
      cell.textContent = labels[row.offside_label] || '—';
      if (row.offside_label === 'TOUCHER_EXCLUDED' && state.offside_position?.mode === 'TENTATIVE_CONTACT_REFERENCE') cell.textContent = 'PASSER?';
      if (row.offside_label === 'UNAVAILABLE' && state.reference?.reference_only && row.offside_delta_q_m != null) {
        cell.textContent = row.offside_delta_q_m > 1e-9 ? 'Vượt vạch HV*' : 'Sau/ngang vạch HV*';
        cell.title = 'So với vạch hậu vệ áp chót; chưa phải kết luận vị trí việt vị.';
      }
      if (row.offside_label === 'OFFSIDE_POSITION') cell.style.color = '#ff7777';
      if (row.offside_label === 'ONSIDE') cell.style.color = '#9ed9b8';
      element.appendChild(cell);
    }
    element.hidden = !useful;
    if (useful) visible += 1;
  });
  const hidden = state.tracks.length - visible;
  document.getElementById('track-count').textContent = visible + ' track hữu ích' + (hidden ? ' · ' + hidden + ' đã lọc' : '');
  document.getElementById('quality-note').textContent =
    'Camera: ' + state.stages[0].status + '. ' + (state.offside_position?.mode === 'DEFENDER_REFERENCE'
      ? 'ONSIDE / OFFSIDE_POSITION được phân loại theo vạch hậu vệ áp chót. Chưa xét bóng; chưa xác nhận thời điểm chạm bóng.'
      : state.offside_position?.mode === 'TENTATIVE_CONTACT_REFERENCE'
      ? 'Đã phân loại ONSIDE / OFFSIDE_POSITION theo vạch S8. Nhóm ATT/OPP và người chạm bóng dựa trên tiếp xúc tạm thời; chưa xác nhận theo thời gian.'
      : state.reference?.reference_only
      ? 'Vạch tham khảo từ hậu vệ áp chót. Đội phòng ngự có thể được suy luận từ thủ môn; chưa đủ dữ liệu để kết luận vị trí việt vị.'
      : 'Stage 9 chạy strict và chỉ phân loại vị trí tại frame đang xem; kết quả chưa được xem là quyết định lỗi việt vị.');
}
const rosterObserver = new MutationObserver(updateStage9Roster);
rosterObserver.observe(document.getElementById('tracks'), {childList: true});
// The initial state request may finish before this extension is loaded.
updateStage9Roster();
