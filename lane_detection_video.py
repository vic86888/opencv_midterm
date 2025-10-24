# -*- coding: utf-8 -*-
"""
Lane detection (LEFT-ANCHORED, robust):
- 轉灰階 → GaussianBlur → (可選 close) → Dynamic Canny
- ROI：偏向『左側錨點』的動態梯形（若無則回退靜態）
- HoughLinesP → 依角度分左右 → 以所有端點做最小平方法擬合 (y = s*x + b)
- 一致性檢核 + 保守回退（左側優先）：
  - 左側為『錨點』：若左側候選不合理，絕不以右側替換；維持上一幀左線
  - 右側若偏離（出口誘導等），則以『左側 + 既有寬度』重建右線
- 影格間 EMA 平滑 (EMA_ALPHA)
- 疊圖：半透明綠區 + 中心偏移 + 偵測狀態
- 參數比例化（隨影像高 HH 調整 Hough）

此版本專門解你的案例：『出口在右側、右實線延伸到出口』，
以**左側虛線為錨**，保持綠色區域穩定貼在左虛線上，不被右出口拉走。

另外新增：
- 左下角（offset 上方）顯示當下 FPS（綠字）
- 左下角右側顯示學號姓名（綠字）
"""

import cv2
import numpy as np
import math
import time  # ← 新增：用來量測當下處理 FPS

# ===== 影像尺寸與路徑 =====
WW, HH = 640, 400        # 縮放後寬高
RH, R  = 0.60, 3         # ROI 高度比例、底部邊界內縮像素
VIDEO_PATH = 'LaneVideo.mp4'  # ← 改成你的影片路徑

# ===== Hough 參數（比例化） =====
HOUGH_THRESHOLD = 40                      # 票數門檻
HOUGH_MIN_LINE_LENGTH_RATIO = 0.08        # 以 HH 為基礎
HOUGH_MAX_LINE_GAP_RATIO   = 0.15

# ===== 顯示選項 =====
EDGES_USE_ROI = True      # 右側只顯示 ROI 內的 Canny（建議開啟）
SHOW_OVERLAY  = True      # 左側疊合綠區/中心線
DEBUG_TEXT    = True

# ===== 角度與平滑 =====
ANGLE_LEFT  = (-85, -25)  # 左車道線角度範圍（度）
ANGLE_RIGHT = (25, 85)    # 右車道線角度範圍（度）
EMA_ALPHA   = 0.7         # 越大越穩但反應慢 (0~1)

# FPS 平滑（顯示更穩定）
FPS_EMA_ALPHA = 0.9       # 0.9 越穩，0.6 反應更快

# ===== 連續性守門（左錨優先） =====
DYNAMIC_ROI            = True
ROI_MARGIN_EXPAND      = 18
LEFT_ANGLE_TOL_DEG     = 8.0
RIGHT_ANGLE_TOL_DEG    = 14.0
LEFT_X_BOTTOM_TOL_RATIO  = 0.06
RIGHT_X_BOTTOM_TOL_RATIO = 0.12
WIDTH_TOL_RATIO        = 0.30
PARALLEL_TOL_DEG       = 18.0

# 寬度限制與名目寬度
MIN_WIDTH_PX = 120
MAX_WIDTH_PX = 320
NOMINAL_WIDTH_PX = 180

# ===== 狀態快取 =====
_prev_left    = None   # (s, b)
_prev_right   = None   # (s, b)
_prev_center_b = None
_prev_width_b  = None
_roi_cache    = None

# ===== 小工具：帶黑框的綠色字 =====
def draw_green_text(img, text, org, scale=0.7, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)   # 外框
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), thick, cv2.LINE_AA)     # 綠字


# ---------- 工具 ----------
# 通用：先畫黑邊再畫彩色字
def draw_text_with_stroke(img, text, org, color, scale=0.7, thickness=2, stroke=3):
    # 黑色外框
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + stroke, cv2.LINE_AA)
    # 目標顏色
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

# 淺藍色（BGR）：約等於 #66CCFF
LIGHT_BLUE = (255, 204, 102)

def dynamic_canny(img_gray_blurred: np.ndarray) -> np.ndarray:
    v = float(np.median(img_gray_blurred))
    lo = int(max(0, 0.66 * v))
    hi = int(min(255, 1.33 * v))
    return cv2.Canny(img_gray_blurred, lo, hi)

def build_roi_mask_static(width: int, height: int, rh: float, margin: int) -> np.ndarray:
    global _roi_cache
    if _roi_cache is not None:
        return _roi_cache
    xx1, yy1 = int(width * 0.40), int(height * rh)
    xx2, yy2 = int(width * 0.60), int(height * rh)
    p1 = (margin, height - margin)
    p2 = (width - margin, height - margin)
    p3 = (xx2, yy2)
    p4 = (xx1, yy2)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array([p1, p2, p3, p4], np.int32)], 255)
    _roi_cache = mask
    return mask

def build_roi_mask_dynamic_by_left(width: int, height: int, y_top: int,
                                   sL: float, bL: float,
                                   width_b: float, expand: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    def clamp_x(x):
        return int(max(0, min(width - 1, x)))
    y_bottom = height - R
    xl_b = (y_bottom - bL) / sL
    xl_t = (y_top    - bL) / sL
    xr_b = xl_b + width_b
    xr_t = xl_t + width_b
    if any(map(lambda v: isinstance(v, float) and (math.isnan(v) or math.isinf(v)), [xl_b, xl_t])):
        return build_roi_mask_static(width, height, RH, R)
    pts = np.array([
        (clamp_x(xl_b - expand), y_bottom),
        (clamp_x(xr_b + expand), y_bottom),
        (clamp_x(xr_t + expand), y_top),
        (clamp_x(xl_t - expand), y_top),
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask

def halfplane_below_line_mask(width: int, height: int, s: float, b: float, margin_px: int = 0) -> np.ndarray:
    X = np.arange(width, dtype=np.float32)[None, :]
    Y = np.arange(height, dtype=np.float32)[:, None]
    Yline = s * X + b + margin_px
    return np.where(Y >= Yline, 255, 0).astype(np.uint8)

def translate_line(s: float, b: float, dx_px: float = 0.0, dy_px: float = 0.0):
    return s, b + dy_px - s * dx_px

def split_and_fit_lines(lines: np.ndarray, width: int, height: int):
    left_pts, right_pts = [], []
    if lines is None:
        return 0, (0.0, 0.0), (0.0, 0.0)
    for x1, y1, x2, y2 in lines[:, 0]:
        if x2 == x1:
            continue
        dx, dy = (x2 - x1), (y2 - y1)
        ang = math.degrees(math.atan2(dy, dx))
        length = math.hypot(dx, dy)
        if min(x1, x2) < 20 or max(x1, x2) > width - 20 or length < 10:
            continue
        if ANGLE_LEFT[0] <= ang <= ANGLE_LEFT[1]:
            left_pts.extend([(x1, y1), (x2, y2)])
        elif ANGLE_RIGHT[0] <= ang <= ANGLE_RIGHT[1]:
            right_pts.extend([(x1, y1), (x2, y2)])

    def fit(points):
        if len(points) < 4:
            return None
        xs = np.array([p[0] for p in points], dtype=np.float32)
        ys = np.array([p[1] for p in points], dtype=np.float32)
        A = np.vstack([xs, np.ones_like(xs)]).T
        s, b = np.linalg.lstsq(A, ys, rcond=None)[0]
        return float(s), float(b)

    L = fit(left_pts)
    R = fit(right_pts)
    done = (1 if L is not None else 0) | (2 if R is not None else 0)
    return done, (L if L else (0.0, 0.0)), (R if R else (0.0, 0.0))

def ema(prev, cur):
    if prev is None:
        return cur
    p = np.array(prev, dtype=np.float32)
    c = np.array(cur,  dtype=np.float32)
    return (EMA_ALPHA * p + (1 - EMA_ALPHA) * c).tolist()

def slope_to_angle_deg(s: float) -> float:
    return math.degrees(math.atan(s))

def x_at_y(s: float, b: float, y: float) -> float:
    if abs(s) < 1e-6:
        return float('nan')
    return (y - b) / s

def continuity_gate(prev: tuple | None, cand: tuple | None,
                    prev_xb: float | None, cand_xb: float | None,
                    tol_deg: float, tol_x_px: float) -> bool:
    if cand is None:
        return False
    if prev is None:
        return True
    if cand_xb is None or prev_xb is None or math.isnan(cand_xb) or math.isnan(prev_xb):
        return True
    s_prev, _ = prev
    s_cand, _ = cand
    ang_prev = slope_to_angle_deg(s_prev)
    ang_cand = slope_to_angle_deg(s_cand)
    if abs(ang_cand - ang_prev) > tol_deg:
        return False
    if abs(cand_xb - prev_xb) > tol_x_px:
        return False
    return True

def synthesize_right_from_left(xl_b: float, s_left: float, y_bottom: int,
                               prev_right: tuple | None,
                               prefer_width: float) -> tuple:
    xr_b = xl_b + prefer_width
    if prev_right is not None:
        s2 = prev_right[0]
    else:
        s2 = abs(s_left) if s_left < 0 else max(1.0, s_left)
    b2 = y_bottom - s2 * xr_b
    return (s2, b2, xr_b)

# ---------- 單幀處理 ----------
def process_frame(frame: np.ndarray, fps_display: float | None = None) -> np.ndarray:
    """fps_display 會以綠字顯示在左半邊左下角（offset 上方）"""
    global _prev_left, _prev_right, _prev_center_b, _prev_width_b

    # 縮放
    img1 = cv2.resize(frame, (WW, HH))

    # 前處理
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    blur  = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel, iterations=1)
    edge  = dynamic_canny(blur)

    # Hough 參數（比例化）
    min_len = int(HH * HOUGH_MIN_LINE_LENGTH_RATIO)
    max_gap = int(HH * HOUGH_MAX_LINE_GAP_RATIO)

    # ROI
    y_top = int(HH * RH)
    if DYNAMIC_ROI and _prev_left is not None and (_prev_width_b is not None or NOMINAL_WIDTH_PX):
        width_b_for_roi = _prev_width_b if _prev_width_b is not None else NOMINAL_WIDTH_PX
        base_mask = build_roi_mask_dynamic_by_left(WW, HH, y_top, _prev_left[0], _prev_left[1], width_b_for_roi, ROI_MARGIN_EXPAND)
    else:
        base_mask = build_roi_mask_static(WW, HH, RH, R)

    edge_roi = cv2.bitwise_and(edge, base_mask)

    # Hough 檢出
    lines = cv2.HoughLinesP(edge_roi, 1, np.pi/180, HOUGH_THRESHOLD, None, min_len, max_gap)
    done, (sL, bL), (sR, bR) = split_and_fit_lines(lines, WW, HH)

    # ---- 左錨一致性檢核 ----
    y_bottom = HH - R
    X_TOL_L = WW * LEFT_X_BOTTOM_TOL_RATIO
    X_TOL_R = WW * RIGHT_X_BOTTOM_TOL_RATIO

    xl_b_cand = x_at_y(sL, bL, y_bottom) if (done & 1) else float('nan')
    xr_b_cand = x_at_y(sR, bR, y_bottom) if (done & 2) else float('nan')
    xl_b_prev = x_at_y(_prev_left[0],  _prev_left[1],  y_bottom) if _prev_left  is not None else float('nan')
    xr_b_prev = x_at_y(_prev_right[0], _prev_right[1], y_bottom) if _prev_right is not None else float('nan')

    cand_left  = (sL, bL) if (done & 1) else None
    cand_right = (sR, bR) if (done & 2) else None

    okL = continuity_gate(_prev_left,  cand_left,  xl_b_prev, xl_b_cand, LEFT_ANGLE_TOL_DEG,  X_TOL_L)
    if okL and cand_left:
        _prev_left = ema(_prev_left, cand_left)

    # ---- 右側：以左為錨 ----
    okR = continuity_gate(_prev_right, cand_right, xr_b_prev, xr_b_cand, RIGHT_ANGLE_TOL_DEG, X_TOL_R) if cand_right else False

    s1, b1 = (_prev_left if _prev_left is not None else (0.0, 0.0))
    if _prev_left is not None:
        xl_b = x_at_y(s1, b1, y_bottom)
        expect_w = _prev_width_b if _prev_width_b is not None else NOMINAL_WIDTH_PX
        use_cand_right = False
        if okR and cand_right and not math.isnan(xl_b) and not math.isnan(xr_b_cand):
            width_b_cand = xr_b_cand - xl_b
            if MIN_WIDTH_PX <= width_b_cand <= MAX_WIDTH_PX:
                if _prev_width_b is None or abs(width_b_cand - _prev_width_b) <= WIDTH_TOL_RATIO * _prev_width_b:
                    angL = slope_to_angle_deg(s1)
                    angR = slope_to_angle_deg(cand_right[0])
                    if abs((angR + angL)) <= PARALLEL_TOL_DEG:
                        use_cand_right = True
        if use_cand_right:
            _prev_right = ema(_prev_right, cand_right)
            xr_b = xr_b_cand
        else:
            s2, b2, xr_b_syn = synthesize_right_from_left(xl_b, s1, y_bottom, _prev_right, expect_w)
            _prev_right = ema(_prev_right, (s2, b2))
            xr_b = xr_b_syn
    else:
        if okR and cand_right:
            _prev_right = ema(_prev_right, cand_right)

    # 組合最終線
    s1, b1 = _prev_left  if _prev_left  is not None else (0.0, 0.0)
    s2, b2 = _prev_right if _prev_right is not None else (0.0, 0.0)
    have_both = (_prev_left is not None) and (_prev_right is not None)

    edges_to_show = edge_roi if EDGES_USE_ROI else edge

    if _prev_right is not None:
        sR, bR = _prev_right
        s_cut, b_cut = translate_line(sR, bR, dx_px=12, dy_px=-12)
        cut_mask = halfplane_below_line_mask(WW, HH, s_cut, b_cut, margin_px=0)
        edges_to_show = cv2.bitwise_and(edges_to_show, cut_mask)

    right = cv2.cvtColor(edges_to_show, cv2.COLOR_GRAY2BGR)

    # 左側疊圖
    left = img1.copy()
    if SHOW_OVERLAY:
        # 底部刻度
        cx, cy = int(WW / 2) - 1, HH - R
        cv2.line(left, (cx, cy), (cx, cy - 12), (255, 0, 0), 2)
        for i in range(1, 10):
            cv2.line(left, (cx - i * 15, cy), (cx - i * 15, cy - 3), (0, 255, 0), 2)
            cv2.line(left, (cx + i * 15, cy), (cx + i * 15, cy - 3), (0, 255, 0), 2)

        if have_both and abs(s1) > 1e-3 and abs(s2) > 1e-3:
            y_top2   = int(HH - HH * 0.175)
            xl_b = x_at_y(s1, b1, y_bottom)
            xr_b = x_at_y(s2, b2, y_bottom)
            xl_t = x_at_y(s1, b1, y_top2)
            xr_t = x_at_y(s2, b2, y_top2)
            if not any(map(lambda v: math.isnan(v) or math.isinf(v), [xl_b, xr_b, xl_t, xr_t])):
                poly = np.array([
                    (int(xl_b), y_bottom),
                    (int(xr_b), y_bottom),
                    (int(xr_t), y_top2),
                    (int(xl_t), y_top2)
                ], dtype=np.int32)
                shade = np.zeros_like(left)
                cv2.fillPoly(shade, [poly], (0, 80, 0))
                left = cv2.addWeighted(left, 1.0, shade, 0.35, 0.0)

                # 中心線與偏移
                x_center_b = int((xl_b + xr_b) * 0.5)
                x_center_t = int((xl_t + xr_t) * 0.5)
                cv2.line(left, (x_center_b, y_bottom), (x_center_t, y_top2), (0, 0, 255), 3)

                offset_px = int(x_center_b - (WW / 2))
                if DEBUG_TEXT:
                    draw_text_with_stroke(
                        left,
                        f"offset: {offset_px:+d}px",
                        (12, HH - 12),          # 與原本同位置
                        color=LIGHT_BLUE,       # 淺藍
                        scale=0.7,
                        thickness=2,
                        stroke=3                # 黑邊厚度，可依喜好 2~4
                    )

                # ★ 新增：在 offset 上方顯示 FPS（綠色字）
                if fps_display is not None:
                    fps_text = f"FPS: {fps_display:.1f}"
                    (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    y_fps = HH - 12 - th - 6  # 緊貼 offset 上方 6px
                    draw_green_text(left, fps_text, (12, y_fps), scale=0.7, thick=2)

                # 更新寬度與中心（EMA）
                width_b = xr_b - xl_b
                if width_b > 0:
                    width_b = max(MIN_WIDTH_PX, min(MAX_WIDTH_PX, width_b))
                    if _prev_center_b is None:
                        _prev_center_b = x_center_b
                    else:
                        _prev_center_b = int(EMA_ALPHA * _prev_center_b + (1 - EMA_ALPHA) * x_center_b)
                    if _prev_width_b is None:
                        _prev_width_b = width_b
                    else:
                        _prev_width_b = (EMA_ALPHA * _prev_width_b + (1 - EMA_ALPHA) * width_b)

    # 右側：Canny 畫面
    right = cv2.cvtColor(edges_to_show, cv2.COLOR_GRAY2BGR)

    # 標題與偵測狀態
    cv2.putText(left,  "Original / Overlay", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    tag = "Canny Edges" + (" (ROI)" if EDGES_USE_ROI else "")
    cv2.putText(right, tag, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    if DEBUG_TEXT:
        stL = "OK" if _prev_left  is not None else "-"
        stR = "OK" if _prev_right is not None else "-"
        cv2.putText(left, f"L:{stL} R:{stR}", (WW - 140, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2, cv2.LINE_AA)

    # ★ 新增：左半邊右下角放上學號姓名（綠色字）
    id_text = "411106202"
    (tw, th), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    draw_green_text(left, id_text, (WW - tw - 12, HH - 12), scale=0.8, thick=2)

    combined = np.hstack([left, right])
    return combined

# ---------- 主程式 ----------
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f'無法開啟影片：{VIDEO_PATH}')

    out_size = (WW * 2, HH)
    fourcc = cv2.VideoWriter_fourcc(*('m','p','4','v'))
    fps_base = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = cv2.VideoWriter('output_OK.mp4', fourcc, fps_base, out_size)

    # FPS 量測用
    fps_disp = fps_base         # 顯示值（先用原始 fps 當起始）
    last_t = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.perf_counter()
        if last_t is not None:
            dt = now - last_t
            if dt > 0:
                fps_inst = 1.0 / dt
                fps_disp = FPS_EMA_ALPHA * fps_disp + (1.0 - FPS_EMA_ALPHA) * fps_inst
        last_t = now

        result = process_frame(frame, fps_display=fps_disp)
        out.write(result)

        cv2.imshow('lane (left: overlay, right: canny)', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print('完成：output_OK.mp4')

if __name__ == '__main__':
    main()
