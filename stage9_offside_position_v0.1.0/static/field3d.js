'use strict';

let field3dView = false;
let sidelineView = false;
let sidelineCenterX = 0;
let sidelineDrag = null;
const drawExistingPitchView = drawPitch;

function spatialTrackColor(row) {
  if (typeof layers !== 'undefined' && layers.has(9)) {
    return {
      OFFSIDE_POSITION: '#ff5e62',
      ONSIDE: '#68d391',
      TOUCHER_EXCLUDED: '#f8eb50',
      UNAVAILABLE: '#a9b2bc'
    }[row.offside_label] || color(row);
  }
  return color(row);
}

function spatialTrackLabel(row) {
  const suffix = {
    OFFSIDE_POSITION: ' OFF', ONSIDE: ' ON', TOUCHER_EXCLUDED: ' PASSER', UNAVAILABLE: ' ON?'
  }[row.offside_label] || '';
  return row.track_id.replace('track_', '#') + suffix;
}

function field3dProjector(width, height, center, yaw, zoom) {
  const length = Number(state.pitch.length_m || 105);
  const pitchWidth = Number(state.pitch.width_m || 68);
  const elevation = 28 * Math.PI / 180;
  const cos = Math.cos(yaw);
  const sin = Math.sin(yaw);
  const corners = [
    [-length / 2, -pitchWidth / 2], [-length / 2, pitchWidth / 2],
    [length / 2, -pitchWidth / 2], [length / 2, pitchWidth / 2]
  ];
  const rotated = corners.map(function (point) {
    return [point[0] * cos - point[1] * sin, point[0] * sin + point[1] * cos];
  });
  const spanU = Math.max(...rotated.map(function (p) { return p[0]; })) - Math.min(...rotated.map(function (p) { return p[0]; }));
  const spanDepth = Math.max(...rotated.map(function (p) { return p[1]; })) - Math.min(...rotated.map(function (p) { return p[1]; }));
  const baseScale = Math.min((width - 44) / spanU, (height - 58) / (spanDepth * Math.sin(elevation) + 2.5 * Math.cos(elevation)));
  const scale = baseScale * zoom;
  return function (point) {
    const x = Number(point[0]) - center[0];
    const y = Number(point[1]) - center[1];
    const z = Number(point[2] || 0);
    const u = x * cos - y * sin;
    const depth = x * sin + y * cos;
    return [
      width / 2 + u * scale,
      height * 0.67 + depth * scale * Math.sin(elevation) - z * scale * Math.cos(elevation),
      depth
    ];
  };
}

function field3dPath(context, points, project, close) {
  context.beginPath();
  points.forEach(function (point, index) {
    const screen = project(point);
    if (index) context.lineTo(screen[0], screen[1]);
    else context.moveTo(screen[0], screen[1]);
  });
  if (close) context.closePath();
}

function drawStadium3D() {
  if (!state) return;
  const canvas = document.getElementById('pitch');
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  const context = canvas.getContext('2d');
  context.scale(ratio, ratio);
  const width = rect.width;
  const height = rect.height;
  const background = context.createLinearGradient(0, 0, 0, height);
  background.addColorStop(0, '#0b1820');
  background.addColorStop(1, '#07110f');
  context.fillStyle = background;
  context.fillRect(0, 0, width, height);

  const rows = state.tracks.filter(function (row) {
    return row.root_world_m && row.joints_world && row.joints_world.some(Boolean);
  });
  let center = [0, 0];
  const focused = rows.find(function (row) { return row.track_id === focus; });
  const toucher = rows.find(function (row) { return row.toucher; });
  if (focused || toucher) {
    const root = (focused || toucher).root_world_m;
    center = [root[0], root[1]];
  } else if (rows.length) {
    center = [
      rows.reduce(function (sum, row) { return sum + row.root_world_m[0]; }, 0) / rows.length,
      rows.reduce(function (sum, row) { return sum + row.root_world_m[1]; }, 0) / rows.length
    ];
  }
  const yaw = Number(document.getElementById('angle').value) * Math.PI / 180;
  const zoom = Number(document.getElementById('field-zoom').value) / 100;
  const project = field3dProjector(width, height, center, yaw, zoom);
  const length = Number(state.pitch.length_m || 105);
  const pitchWidth = Number(state.pitch.width_m || 68);

  for (let stripe = 0; stripe < 10; stripe += 1) {
    const x0 = -length / 2 + stripe * length / 10;
    const x1 = x0 + length / 10;
    field3dPath(context, [[x0, -pitchWidth / 2, 0], [x1, -pitchWidth / 2, 0], [x1, pitchWidth / 2, 0], [x0, pitchWidth / 2, 0]], project, true);
    context.fillStyle = stripe % 2 ? '#123d31' : '#174637';
    context.fill();
  }
  context.strokeStyle = '#9dc8ad';
  context.lineWidth = 1;
  state.pitch_lines.forEach(function (polyline) {
    field3dPath(context, polyline.map(function (point) { return [point[0], point[1], point[2] || 0.01]; }), project, false);
    context.stroke();
  });

  const referenceX = state.reference && state.reference.X_world_m;
  if (referenceX != null) {
    field3dPath(context, [[referenceX, -pitchWidth / 2, 0.04], [referenceX, pitchWidth / 2, 0.04]], project, false);
    context.strokeStyle = '#ff6969';
    context.lineWidth = 2;
    context.setLineDash([7, 5]);
    context.stroke();
    context.setLineDash([]);
  }

  rows.sort(function (a, b) {
    return project(a.root_world_m)[2] - project(b.root_world_m)[2];
  }).forEach(function (row) {
    const dimmed = focus && row.track_id !== focus;
    context.globalAlpha = dimmed ? 0.28 : 1;
    context.strokeStyle = spatialTrackColor(row);
    context.fillStyle = spatialTrackColor(row);
    context.lineWidth = row.track_id === focus ? 3 : 2;
    state.skeleton_edges.forEach(function (edge) {
      const a = row.joints_world[edge[0]];
      const b = row.joints_world[edge[1]];
      if (!a || !b) return;
      const pa = project(a);
      const pb = project(b);
      context.beginPath();
      context.moveTo(pa[0], pa[1]);
      context.lineTo(pb[0], pb[1]);
      context.stroke();
    });
    row.joints_world.filter(Boolean).forEach(function (joint) {
      const point = project(joint);
      context.beginPath();
      context.arc(point[0], point[1], row.track_id === focus ? 2.5 : 1.8, 0, Math.PI * 2);
      context.fill();
    });
    const visibleJoints = row.joints_world.filter(Boolean).map(project);
    const labelY = Math.min(...visibleJoints.map(function (point) { return point[1]; })) - 5;
    const root = project(row.root_world_m);
    context.font = '10px Segoe UI';
    context.fillStyle = '#eef7f1';
    context.fillText(spatialTrackLabel(row), root[0] + 4, labelY);
  });
  context.globalAlpha = 1;

  const ball = state.ball.center_xyz_world_m;
  if (ball) {
    const point = project(ball);
    context.fillStyle = '#fff4a3';
    context.strokeStyle = '#9d7b12';
    context.lineWidth = 1.5;
    context.beginPath();
    context.arc(point[0], point[1], 5, 0, Math.PI * 2);
    context.fill();
    context.stroke();
  }
  context.font = '10px Segoe UI';
  context.fillStyle = '#afc6ba';
  context.fillText('Sân đúng kích thước ' + length + ' × ' + pitchWidth + ' m · skeleton từ Stage 4 · đường đỏ từ Stage 8', 14, height - 10);
}

function sidelineVisibleSpan() {
  return 64 / (Number(document.getElementById('field-zoom').value) / 100);
}

function drawSideline() {
  if (!state) return;
  const canvas = document.getElementById('pitch');
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  const context = canvas.getContext('2d');
  context.scale(ratio, ratio);
  const width = rect.width;
  const height = rect.height;
  const length = Number(state.pitch.length_m || 105);
  const pitchWidth = Number(state.pitch.width_m || 68);
  const visibleSpan = sidelineVisibleSpan();
  const horizontalScale = width / visibleSpan;
  const depthScale = Math.min(3.2, height * 0.34 / pitchWidth);
  const groundY = height * 0.91;
  const project = function (point) {
    return [
      width / 2 + (Number(point[0]) - sidelineCenterX) * horizontalScale,
      groundY - (Number(point[1]) + pitchWidth / 2) * depthScale - Number(point[2] || 0) * horizontalScale * 1.05
    ];
  };

  const sky = context.createLinearGradient(0, 0, 0, height);
  sky.addColorStop(0, '#0b3152');
  sky.addColorStop(0.48, '#0b263b');
  sky.addColorStop(0.49, '#123c32');
  sky.addColorStop(1, '#071713');
  context.fillStyle = sky;
  context.fillRect(0, 0, width, height);

  field3dPath(context, [
    [-length / 2, -pitchWidth / 2, 0], [length / 2, -pitchWidth / 2, 0],
    [length / 2, pitchWidth / 2, 0], [-length / 2, pitchWidth / 2, 0]
  ], project, true);
  context.fillStyle = '#164c38';
  context.fill();
  context.strokeStyle = '#a7d0b4';
  context.lineWidth = 1;
  state.pitch_lines.forEach(function (polyline) {
    field3dPath(context, polyline.map(function (point) { return [point[0], point[1], point[2] || 0.01]; }), project, false);
    context.stroke();
  });

  const referenceX = state.reference && state.reference.X_world_m;
  if (referenceX != null) {
    const reference = project([referenceX, 0, 0]);
    context.strokeStyle = '#ff6868';
    context.lineWidth = 2;
    context.setLineDash([7, 5]);
    context.beginPath();
    context.moveTo(reference[0], 8);
    context.lineTo(reference[0], height - 8);
    context.stroke();
    context.setLineDash([]);
    context.fillStyle = '#ff9a9a';
    context.font = '10px Segoe UI';
    context.fillText('MỐC S8', reference[0] + 6, 18);
  }

  const rows = state.tracks.filter(function (row) {
    return row.root_world_m && row.joints_world && row.joints_world.some(Boolean);
  }).sort(function (a, b) {
    return b.root_world_m[1] - a.root_world_m[1];
  });
  rows.forEach(function (row) {
    const dimmed = focus && row.track_id !== focus;
    context.globalAlpha = dimmed ? 0.25 : 1;
    context.strokeStyle = spatialTrackColor(row);
    context.fillStyle = spatialTrackColor(row);
    context.lineWidth = row.track_id === focus ? 3.5 : 2.2;
    state.skeleton_edges.forEach(function (edge) {
      const a = row.joints_world[edge[0]];
      const b = row.joints_world[edge[1]];
      if (!a || !b) return;
      const pa = project(a);
      const pb = project(b);
      context.beginPath();
      context.moveTo(pa[0], pa[1]);
      context.lineTo(pb[0], pb[1]);
      context.stroke();
    });
    const joints = row.joints_world.filter(Boolean).map(project);
    joints.forEach(function (point) {
      context.beginPath();
      context.arc(point[0], point[1], row.track_id === focus ? 2.7 : 2, 0, Math.PI * 2);
      context.fill();
    });
    const root = project(row.root_world_m);
    const top = Math.min(...joints.map(function (point) { return point[1]; }));
    context.fillStyle = '#eef7f1';
    context.font = '10px Segoe UI';
    context.fillText(spatialTrackLabel(row), root[0] + 5, top - 5);
  });
  context.globalAlpha = 1;

  const ball = state.ball.center_xyz_world_m;
  if (ball) {
    const point = project(ball);
    context.fillStyle = '#fff4a3';
    context.strokeStyle = '#775f16';
    context.lineWidth = 1.5;
    context.beginPath();
    context.arc(point[0], point[1], 5, 0, Math.PI * 2);
    context.fill();
    context.stroke();
  }
  context.fillStyle = '#d3e2da';
  context.font = '10px Segoe UI';
  context.fillText('SIDELINE · kéo trái/phải để di chuyển camera dọc sân', 14, height - 10);
  const position = document.getElementById('sideline-position');
  position.textContent = 'Camera X: ' + sidelineCenterX.toFixed(1) + ' m';
}

drawPitch = function () {
  if (sidelineView) drawSideline();
  else if (field3dView) drawStadium3D();
  else drawExistingPitchView();
};

function leaveSpatialView() {
  field3dView = false;
  sidelineView = false;
  document.getElementById('field3d').classList.remove('active');
  document.getElementById('sideline').classList.remove('active');
  document.getElementById('field-zoom-control').hidden = true;
  document.getElementById('sideline-position').hidden = true;
  document.getElementById('angle-control').hidden = false;
  document.getElementById('pitch').style.cursor = '';
  document.getElementById('pitch').style.touchAction = '';
}

document.getElementById('topdown').addEventListener('click', leaveSpatialView);
document.getElementById('poseview').addEventListener('click', leaveSpatialView);
document.getElementById('field3d').addEventListener('click', function () {
  field3dView = true;
  sidelineView = false;
  poseView = false;
  document.getElementById('topdown').classList.remove('active');
  document.getElementById('poseview').classList.remove('active');
  document.getElementById('sideline').classList.remove('active');
  document.getElementById('field3d').classList.add('active');
  document.getElementById('rotation').hidden = false;
  document.getElementById('field-zoom-control').hidden = false;
  document.getElementById('sideline-position').hidden = true;
  document.getElementById('angle-control').hidden = false;
  document.getElementById('pitch-help').textContent = 'Các skeleton Stage 4 được đặt theo tọa độ sân thật. Xoay và phóng đại để quan sát khoảng cách với bóng và mốc Stage 8.';
  drawPitch();
});
document.getElementById('sideline').addEventListener('click', function () {
  sidelineView = true;
  field3dView = false;
  poseView = false;
  const preferred = state && state.tracks.find(function (row) { return row.track_id === focus && row.root_world_m; });
  const toucher = state && state.tracks.find(function (row) { return row.toucher && row.root_world_m; });
  if (preferred || toucher) sidelineCenterX = Number((preferred || toucher).root_world_m[0]);
  document.getElementById('topdown').classList.remove('active');
  document.getElementById('field3d').classList.remove('active');
  document.getElementById('poseview').classList.remove('active');
  document.getElementById('sideline').classList.add('active');
  document.getElementById('rotation').hidden = false;
  document.getElementById('angle-control').hidden = true;
  document.getElementById('field-zoom-control').hidden = false;
  document.getElementById('sideline-position').hidden = false;
  document.getElementById('pitch-help').textContent = 'Góc nhìn ngang từ đường biên. Kéo trực tiếp sang trái/phải để tịnh tiến camera dọc toàn bộ chiều dài sân.';
  document.getElementById('pitch').style.cursor = 'grab';
  document.getElementById('pitch').style.touchAction = 'none';
  drawPitch();
});
document.getElementById('field-zoom').addEventListener('input', drawPitch);

const fieldCanvas = document.getElementById('pitch');
fieldCanvas.addEventListener('pointerdown', function (event) {
  if (!sidelineView) return;
  sidelineDrag = {pointerId: event.pointerId, startX: event.clientX, centerX: sidelineCenterX};
  fieldCanvas.setPointerCapture(event.pointerId);
  fieldCanvas.style.cursor = 'grabbing';
});
fieldCanvas.addEventListener('pointermove', function (event) {
  if (!sidelineView || !sidelineDrag || sidelineDrag.pointerId !== event.pointerId) return;
  const rect = fieldCanvas.getBoundingClientRect();
  const metersPerPixel = sidelineVisibleSpan() / rect.width;
  const halfLength = Number(state.pitch.length_m || 105) / 2;
  sidelineCenterX = Math.max(-halfLength, Math.min(halfLength, sidelineDrag.centerX - (event.clientX - sidelineDrag.startX) * metersPerPixel));
  drawPitch();
});
function finishSidelineDrag(event) {
  if (!sidelineDrag || sidelineDrag.pointerId !== event.pointerId) return;
  sidelineDrag = null;
  fieldCanvas.style.cursor = sidelineView ? 'grab' : '';
}
fieldCanvas.addEventListener('pointerup', finishSidelineDrag);
fieldCanvas.addEventListener('pointercancel', finishSidelineDrag);
fieldCanvas.addEventListener('wheel', function (event) {
  if (!sidelineView) return;
  event.preventDefault();
  const zoom = document.getElementById('field-zoom');
  zoom.value = Math.max(Number(zoom.min), Math.min(Number(zoom.max), Number(zoom.value) - Math.sign(event.deltaY) * 15));
  drawPitch();
}, {passive: false});
