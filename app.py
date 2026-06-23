import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path

st.set_page_config(
    page_title="바벨 궤적 분석기",
    page_icon="🏋️",
    layout="centered"
)

TRAIL_COLOR = (255, 255, 255)
TRAIL_THICKNESS = 3
CIRCLE_RADIUS = 6
FADE_FRAMES = 90
MAX_FAIL = 30
TRACK_SCALE = 0.5   # 50% 해상도 추적 → 2~3배 속도


@st.cache_resource(show_spinner=False)
def load_model():
    from ultralytics import YOLO
    return YOLO("yolov8n-pose.pt")


def draw_trail(frame, points):
    if len(points) < 2:
        return
    draw_pts = points[-FADE_FRAMES:]
    total = len(draw_pts)
    for i in range(1, total):
        alpha = i / total
        cv2.line(frame, draw_pts[i - 1], draw_pts[i], TRAIL_COLOR,
                 max(1, int(TRAIL_THICKNESS * alpha)), cv2.LINE_AA)
    cv2.circle(frame, draw_pts[-1], CIRCLE_RADIUS, (0, 200, 255), -1, cv2.LINE_AA)


def detect_barbell_roi(frame, model):
    """
    YOLO pose → 귀/어깨/손목으로 바 높이 추정 →
    해당 높이 구간에서 밝기(크롬/금속) 기반으로 바 가장자리(끝) 위치 탐색.

    Returns:
        (roi, edge_ok)
        roi     — (x, y, w, h) 추적 박스
        edge_ok — True: 바벨 가장자리 감지 성공 / False: 폴백(중심 기반)
    """
    h, w = frame.shape[:2]
    results = model(frame, verbose=False, conf=0.3)

    for result in results:
        if result.keypoints is None or len(result.keypoints.xy) == 0:
            continue
        kp_xy = result.keypoints.xy[0].cpu().numpy()
        kp_conf = result.keypoints.conf[0].cpu().numpy()

        def cf(i): return float(kp_conf[i])
        def kp(i): return float(kp_xy[i][0]), float(kp_xy[i][1])

        ear_ys    = [kp(i)[1] for i in [3, 4] if cf(i) > 0.4]
        shld_ys   = [kp(i)[1] for i in [5, 6] if cf(i) > 0.4]
        shld_xs   = [kp(i)[0] for i in [5, 6] if cf(i) > 0.4]
        wrist_pts = [kp(i)    for i in [9, 10] if cf(i) > 0.4]

        shld_y  = sum(shld_ys) / len(shld_ys) if shld_ys else None
        shld_x  = sum(shld_xs) / len(shld_xs) if shld_xs else w // 2
        ear_y   = sum(ear_ys)  / len(ear_ys)  if ear_ys  else None
        wrist_y = sum(p[1] for p in wrist_pts) / len(wrist_pts) if wrist_pts else None
        wrist_x = sum(p[0] for p in wrist_pts) / len(wrist_pts) if wrist_pts else None

        if shld_y is None and wrist_y is None:
            continue

        in_squat = shld_y and wrist_y and abs(wrist_y - shld_y) < h // 5
        if in_squat:
            ref_y = int(ear_y) if ear_y else int(shld_y)
        elif wrist_y:
            ref_y = int(wrist_y)
        else:
            ref_y = int(shld_y)

        body_cx = int(shld_x)
        size = max(80, h // 12)

        # --- 밝기로 바 위치 탐색 ---
        scan_top = max(0, ref_y - h // 5)
        scan_bot = min(h, ref_y + h // 10)
        strip = frame[scan_top:scan_bot]
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)

        BRIGHT_THR = 160
        MIN_BAR_W  = w // 6

        best_score = 0
        bar_row = -1
        bar_bright_cols = None

        for row_idx in range(gray.shape[0]):
            row = gray[row_idx]
            bright_mask = row > BRIGHT_THR
            n = int(bright_mask.sum())
            if n < MIN_BAR_W:
                continue
            score = n * float(row[bright_mask].mean())
            if score > best_score:
                best_score = score
                bar_row = row_idx
                bar_bright_cols = np.where(bright_mask)[0]

        if bar_row < 0 or bar_bright_cols is None:
            # 폴백: 손목/어깨 중심 기반
            ref_x = int(wrist_x) if wrist_x else body_cx
            x = max(0, min(ref_x - size // 2, w - size))
            y = max(0, min(ref_y - size // 2, h - size))
            return (x, y, size, size), False

        bar_y = scan_top + bar_row

        # 바벨 가장자리(끝) 선택: 몸 중심에서 가장 먼 쪽
        left_edge  = int(np.percentile(bar_bright_cols, 2))   # 왼쪽 끝 (노이즈 제거)
        right_edge = int(np.percentile(bar_bright_cols, 98))  # 오른쪽 끝

        if abs(left_edge - body_cx) >= abs(right_edge - body_cx):
            edge_x = left_edge
        else:
            edge_x = right_edge

        # 프레임 경계에 너무 가까우면 반대편 가장자리 시도
        MARGIN = w // 15
        if edge_x < MARGIN or edge_x > w - MARGIN:
            edge_x = right_edge if edge_x == left_edge else left_edge

        # 반대편도 경계에 있으면 폴백
        if edge_x < MARGIN or edge_x > w - MARGIN:
            ref_x = int(wrist_x) if wrist_x else body_cx
            x = max(0, min(ref_x - size // 2, w - size))
            y = max(0, min(ref_y - size // 2, h - size))
            return (x, y, size, size), False

        x = max(0, min(edge_x - size // 2, w - size))
        y = max(0, min(bar_y  - size // 2, h - size))
        return (x, y, size, size), True

    return None, False


def process_video(input_path, output_path, roi, on_progress):
    cap = cv2.VideoCapture(input_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return False, "첫 프레임을 읽을 수 없습니다."

    x, y, bw, bh = roi
    s = TRACK_SCALE

    # 절반 해상도로 트래커 초기화 (속도 향상)
    small_first = cv2.resize(first_frame, (0, 0), fx=s, fy=s)
    tracker = cv2.TrackerCSRT.create()
    tracker.init(small_first, (int(x*s), int(y*s), int(bw*s), int(bh*s)))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not out.isOpened():
        cap.release()
        return False, "출력 파일을 생성할 수 없습니다."

    trail = [(x + bw // 2, y + bh // 2)]
    f0 = first_frame.copy()
    draw_trail(f0, trail)
    out.write(f0)

    frame_idx = 1
    fail_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        small = cv2.resize(frame, (0, 0), fx=s, fy=s)
        success, bbox = tracker.update(small)

        if success:
            fail_count = 0
            tx = int(bbox[0] / s)
            ty = int(bbox[1] / s)
            tw = int(bbox[2] / s)
            th = int(bbox[3] / s)
            trail.append((tx + tw // 2, ty + th // 2))
        else:
            fail_count += 1
            if fail_count >= MAX_FAIL:
                fail_count = 0

        draw_trail(frame, trail)
        out.write(frame)

        # 30프레임마다 진행률 업데이트
        if total > 0 and frame_idx % 30 == 0:
            on_progress(frame_idx / total)

    cap.release()
    out.release()
    return True, None


def main():
    st.title("🏋️ 바벨 궤적 분석기")
    st.write("운동 영상을 올리면 바벨 궤적을 자동으로 추적합니다.")
    st.caption("지원 종목: 스쿼트 · 벤치프레스 · 데드리프트")
    st.divider()

    uploaded = st.file_uploader(
        "영상 파일 선택 (mp4 / mov / avi / mkv)",
        type=["mp4", "mov", "avi", "mkv"],
    )

    if uploaded is None:
        return

    st.success(f"업로드 완료: **{uploaded.name}**  ({uploaded.size / 1024 / 1024:.1f} MB)")

    if st.button("🔍 분석 시작", type="primary", use_container_width=True):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path  = os.path.join(tmpdir, "input.mp4")
            output_path = os.path.join(tmpdir, "output.mp4")
            with open(input_path, "wb") as f:
                f.write(uploaded.getbuffer())

            # 1단계: 바벨 위치 감지
            with st.spinner("바벨 위치 감지 중..."):
                model = load_model()
                cap = cv2.VideoCapture(input_path)
                ret, first_frame = cap.read()
                cap.release()

            if not ret:
                st.error("❌ 영상을 읽을 수 없습니다.")
                return

            roi, edge_ok = detect_barbell_roi(first_frame, model)
            if roi is None:
                st.error("❌ 사람을 감지하지 못했습니다.\n사람이 화면에 잘 보이는 영상인지 확인해주세요.")
                return

            # 감지 위치 미리보기
            x, y, bw, bh = roi
            preview = first_frame.copy()
            cv2.rectangle(preview, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
            # 트래킹 포인트(박스 중심) 빨간 점으로 표시
            cx, cy = x + bw // 2, y + bh // 2
            cv2.circle(preview, (cx, cy), 8, (0, 0, 255), -1, cv2.LINE_AA)
            st.image(
                cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                caption="감지된 바벨 시작 위치 (초록 박스 / 빨간 점 = 트래킹 포인트) — 바벨 위에 있으면 정상",
                use_container_width=True
            )

            if not edge_ok:
                st.error(
                    "❌ 바벨 가장자리 감지 실패\n\n"
                    "트래킹 포인트(빨간 점)가 바벨의 끝(가장자리)에 위치하지 않았습니다.\n"
                    "다른 영상으로 다시 시도하거나, 바벨 끝이 잘 보이는 각도에서 촬영해주세요."
                )
                return

            st.success("✅ 바벨 가장자리 감지 성공 — 트래킹을 시작합니다.")

            # 2단계: 영상 처리
            st.write("**분석 중...**")
            progress_bar = st.progress(0.0)
            status = st.empty()

            def on_progress(val):
                progress_bar.progress(min(val, 1.0))
                status.write(f"{int(val * 100)}% 완료")

            success, error = process_video(input_path, output_path, roi, on_progress)

            if success:
                progress_bar.progress(1.0)
                status.write("100% 완료")
                st.success("✅ 분석 완료!")
                with open(output_path, "rb") as f:
                    result_bytes = f.read()
                st.download_button(
                    label="⬇️ 결과 영상 다운로드",
                    data=result_bytes,
                    file_name=f"{Path(uploaded.name).stem}_barpath.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
            else:
                st.error(f"❌ {error}")


if __name__ == "__main__":
    main()
