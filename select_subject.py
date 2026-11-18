"""① 피험자 선택 (제안서 4.3 ①). extract_pose.py가 저장한 모든 트랙에서 라벨을 붙일 한 사람을 고른다.

1) 끊긴 트랙 합치기: 트랙이 끝난 뒤 3초 안에, 마지막 박스 긴 변 이내 거리에서 새 트랙이 시작되면 같은 사람
2) 학습·검증 영상: 넘어지는 중·쓰러짐 구간에 가장 많이 잡힌 사람. 1등이 2등의 1.5배가 안 되면 제외
   (낙상 구간이 없는 일상 영상은 고를 기준이 없으므로 테스트와 같이 가장 오래 잡힌 사람)
3) 테스트 영상: 정답을 보지 않고 가장 오래 잡힌 사람
4) 고른 사람이 잡힌 프레임이 50% 미만이면 제외
5) (제안서에 없음) 움직이는 사람이 있으면, 처음부터 끝까지 제자리인 검출은 후보에서 뺀다.
   GMDCSA24 피험자 2·3의 방에서 벽에 걸린 옷이 영상 내내 사람으로 잡혀 피험자를 밀어내는 것을 확인했다.
   기준: 박스 중심의 이동 범위가 박스 긴 변의 15% 미만 (걸린 옷은 3~10%)
"""
import numpy as np

MERGE_GAP_S = 3.0       # 제안서 값은 1초. GMDCSA24에서 넘어지는 동안 1~2초 놓쳤다가 다시 잡히는 영상이 많아 늘림
MERGE_OVERLAP = 5      # 프레임. ID가 바뀌는 순간 옛 트랙과 새 트랙이 잠깐 겹치는 것을 허용
MERGE_GATE = 1.0        # 박스 긴 변의 배수
DOMINANCE = 1.5
MIN_COVERAGE = 0.5
STATIC_MOVE = 0.15


def _center(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def merge_tracks(frame, tid, box, fps=15):
    """트랙 ID -> 사람 번호. 시작이 빠른 트랙부터 보면서 이미 끝난 사람 중 가장 가까운 쪽에 붙인다."""
    max_gap = int(round(MERGE_GAP_S * fps))
    tracks = []
    for t in np.unique(tid):
        idx = np.where(tid == t)[0]
        idx = idx[np.argsort(frame[idx], kind="stable")]
        tracks.append((int(frame[idx[0]]), int(frame[idx[-1]]), int(t), box[idx[0]], box[idx[-1]]))
    tracks.sort(key=lambda a: (a[0], a[2]))
    groups, owner = [], {}                      # groups[g] = [마지막 프레임, 마지막 박스]
    for first, last, t, b_first, b_last in tracks:
        best, best_d = None, None
        for g, (g_last, g_box) in enumerate(groups):
            gap = first - g_last
            if gap <= -MERGE_OVERLAP or gap > max_gap:   # 같은 시간에 오래 함께 있던 트랙은 다른 사람
                continue
            d = float(np.linalg.norm(_center(b_first) - _center(g_box)))
            side = max(g_box[2] - g_box[0], g_box[3] - g_box[1])
            if d <= MERGE_GATE * side and (best is None or d < best_d):
                best, best_d = g, d
        if best is None:
            groups.append([last, b_last])
            owner[t] = len(groups) - 1
        else:
            groups[best] = [last, b_last]
            owner[t] = best
    return owner


def movement(b):
    """박스 중심이 움직인 범위 / 박스 긴 변 평균"""
    c = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
    side = np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]).mean()
    return float(np.ptp(c, 0).max() / max(side, 1e-6))


def select_subject(pose, frame_labels=None, use_labels=False):
    """pose: extract_pose.py의 npz(또는 같은 키의 dict). frame_labels: (n_frames,) 3클래스 정답, 제외는 -1.
    반환 dict: status(ok | no_person | ambiguous | low_coverage), kpts (T,17,3), box (T,4), valid (T,),
              coverage, n_tracks, n_people"""
    T = int(pose["n_frames"])
    frame, tid = np.asarray(pose["det_frame"]), np.asarray(pose["det_tid"])
    kpts, box, conf = np.asarray(pose["det_kpts"]), np.asarray(pose["det_box"]), np.asarray(pose["det_conf"])
    res = dict(status="no_person", kpts=np.zeros((T, 17, 3), np.float32), box=np.zeros((T, 4), np.float32),
               valid=np.zeros(T, bool), coverage=0.0, n_tracks=int(len(np.unique(tid))), n_people=0)
    if len(frame) == 0 or T == 0:
        return res
    owner = merge_tracks(frame, tid, box, float(pose["fps"]))
    person = np.array([owner[int(t)] for t in tid])
    people = np.unique(person)
    res["n_people"] = int(len(people))
    moving = [p for p in people if movement(box[person == p]) >= STATIC_MOVE]
    if moving:
        people = np.array(moving)

    length = {p: len(np.unique(frame[person == p])) for p in people}
    pick = max(people, key=lambda p: (length[p], -p))
    if use_labels and frame_labels is not None and (frame_labels > 0).any():
        in_fall = frame_labels[frame] > 0
        cnt = {p: len(np.unique(frame[(person == p) & in_fall])) for p in people}
        ranked = sorted(people, key=lambda p: (-cnt[p], -length[p], p))
        if cnt[ranked[0]] > 0:
            pick = ranked[0]
            if len(ranked) > 1 and cnt[ranked[0]] < DOMINANCE * cnt[ranked[1]]:
                res["status"] = "ambiguous"
                return res

    idx = np.where(person == pick)[0]
    idx = idx[np.argsort(conf[idx], kind="stable")]        # 같은 프레임이 겹치면 신뢰도 높은 쪽이 남는다
    res["kpts"][frame[idx]] = kpts[idx]
    res["box"][frame[idx]] = box[idx]
    res["valid"][frame[idx]] = True
    res["coverage"] = float(res["valid"].mean())
    res["status"] = "ok" if res["coverage"] >= MIN_COVERAGE else "low_coverage"
    return res
