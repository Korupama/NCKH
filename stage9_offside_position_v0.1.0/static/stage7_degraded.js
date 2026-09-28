'use strict';

// Stage 7 may resolve a useful team context from Stage 6's nearest spatial
// contact while explicitly keeping the result DEGRADED until temporal contact
// evidence is available. Keep that distinction visible in the demo.
group = function (row) {
  if (row.toucher) {
    return row.toucher_tentative ? 'Chạm bóng tạm thời' : 'Chạm bóng';
  }
  return ({
    attackers: 'Tấn công',
    opponents: 'Đối phương',
    referees_excluded: 'Trọng tài',
    unknown_team_excluded: 'Chưa rõ đội',
    inactive_excluded: 'Không hoạt động'
  })[row.group] || (row.active ? 'Chưa xác định' : 'Không có ở frame này');
};

document.addEventListener('click', function (event) {
  const card = event.target.closest('[data-stage="7"]');
  if (!card || !state || state.game_state.status !== 'DEGRADED') return;
  setTimeout(function () {
    const detail = document.getElementById('stage-detail');
    if (detail) {
      detail.textContent += ' · Người chạm bóng tạm suy ra từ khoảng cách ảnh; thiếu xác nhận theo thời gian.';
    }
  }, 0);
});
