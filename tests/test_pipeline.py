"""가짜 관절 영상으로 각 단계가 설계대로 동작하는지 확인한다 (제안서의 '파일럿'에 해당).
실행: python -m unittest discover -s tests
여기의 수치는 코드 검증용이고 실제 성능과 무관하다.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import augment
import dataset
from dataset import EXCLUDE, Corpus, frame_labels, gather
from evaluate import frame_metrics
from extract_pose import target_frames
from features import ANGLE, ASPECT, CY_V, HEAD_V, FeatureState, compute_features
from infer import StreamClassifier, predict_corpus, predict_stream
from models import MODELS, build_model, load_ckpt, n_params, save_ckpt
from select_subject import select_subject
from train import class_weights, epoch_order, focal_loss

# 서 있는 사람의 관절 (엉덩이 중심 기준, 키 약 1.7). COCO 17 순서, y는 아래가 +
STAND = np.array([
    [0, -0.95], [-0.03, -0.98], [0.03, -0.98], [-0.07, -0.95], [0.07, -0.95],     # 코, 눈, 귀
    [-0.18, -0.6], [0.18, -0.6], [-0.22, -0.3], [0.22, -0.3], [-0.24, 0.0], [0.24, 0.0],   # 어깨, 팔꿈치, 손목
    [-0.12, 0.0], [0.12, 0.0], [-0.13, 0.4], [0.13, 0.4], [-0.13, 0.8], [0.13, 0.8]])      # 엉덩이, 무릎, 발목


def person(angle_deg=0.0, center=(320, 240), size=100.0, conf=0.9):
    """몸을 angle만큼 기울인 사람 한 프레임 -> kpts (17,3), box (4,)"""
    a = np.radians(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    xy = STAND @ R.T * size + np.array(center)
    k = np.concatenate([xy, np.full((17, 1), conf)], 1)
    return k, np.array([xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()])


def video(angles, centers=None, **kw):
    T = len(angles)
    centers = centers if centers is not None else [(320, 240)] * T
    ks, bs = zip(*[person(a, c, **kw) for a, c in zip(angles, centers)])
    return np.stack(ks), np.stack(bs), np.ones(T, bool)


def fall_video(T=60, t_fall=20, dur=8):
    """서 있다가 t_fall부터 dur프레임 동안 넘어져 누운 채로 끝난다. 반환: kpts, box, valid, 정답"""
    ang = np.concatenate([np.zeros(t_fall), np.linspace(0, 90, dur), np.full(T - t_fall - dur, 90.0)])
    cen = [(320, 240 + 0.8 * a) for a in ang]
    y = np.concatenate([np.zeros(t_fall), np.ones(dur), np.full(T - t_fall - dur, 2)]).astype(np.int8)
    return (*video(ang, cen), y)


def fake_corpus(n=12, seed=0):
    """낙상 영상과 서 있는 영상을 섞은 작은 Corpus"""
    rng = np.random.default_rng(seed)
    c = Corpus.__new__(Corpus)
    feats, ys, lens = [], [], []
    c.names, c.dataset, c.night = [], [], []
    for i in range(n):
        if i % 2 == 0:
            k, b, v, y = fall_video(60, int(rng.integers(10, 30)), int(rng.integers(5, 10)))
        else:
            k, b, v = video(rng.uniform(-5, 5, 60))
            y = np.zeros(60, np.int8)
        feats.append(compute_features(k, b, v))
        ys.append(y)
        lens.append(60)
        c.names.append(f"v{i}")
        c.dataset.append("of-syn" if i < n - 2 else "GMDCSA24")
        c.night.append(False)
    c.feat, c.y = np.concatenate(feats), np.concatenate(ys).astype(np.int64)
    c.det = np.ones(len(c.y), bool)
    c.start = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    return c


class TestFeatures(unittest.TestCase):
    def test_standing_and_lying(self):
        f = compute_features(*video([0, 0, 90, 90]))
        self.assertAlmostEqual(f[1, ANGLE], 0.0, places=5)
        self.assertAlmostEqual(f[3, ANGLE], 1.0, places=5)
        self.assertLess(f[1, ASPECT], 0)                    # 서 있으면 세로로 긴 박스
        self.assertGreater(f[3, ASPECT], 0)                 # 누우면 가로로 긴 박스
        self.assertLessEqual(np.abs(f[:, :51].reshape(-1, 17, 3)[..., :2]).max(), 1.0 + 1e-6)

    def test_position_and_size_invariant(self):
        a = compute_features(*video([0, 30, 60, 90]))
        b = compute_features(*video([0, 30, 60, 90], centers=[(900, 500)] * 4, size=37.0))
        np.testing.assert_allclose(a, b, atol=1e-5)

    def test_lying_keeps_body_size(self):
        """누워도 몸 크기가 부풀지 않는다 (박스 높이로 나누면 누울 때 좌표가 커짐)"""
        f = compute_features(*video([0, 90])).reshape(2, 55)[:, :51].reshape(2, 17, 3)
        spread = np.linalg.norm(f[..., :2], axis=2).max(1)
        self.assertAlmostEqual(spread[0], spread[1], places=5)

    def test_missing_joint_holds_previous(self):
        k, b, v = video([0, 0, 0])
        k[1:, 9, 2] = 0.1                                   # 왼 손목 신뢰도가 0.3 미만으로 떨어짐
        k[1:, 9, :2] = 0                                    # 좌표도 엉뚱한 값
        f = compute_features(k, b, v).reshape(3, 55)[:, :51].reshape(3, 17, 3)
        np.testing.assert_allclose(f[1, 9, :2], f[0, 9, :2], atol=1e-6)
        self.assertEqual(f[1, 9, 2], 0.0)
        self.assertGreater(f[0, 9, 2], 0.0)

    def test_missing_person(self):
        k, b, v = video([0, 0, 0, 0])
        v[:1] = False                                       # 처음엔 아직 못 봄 -> 전부 0
        v[2] = False                                        # 중간에 놓침 -> 좌표 유지, 신뢰도 0
        f = compute_features(k, b, v)
        self.assertFalse(f[0].any())
        j = f[:, :51].reshape(4, 17, 3)
        np.testing.assert_allclose(j[2, :, :2], j[1, :, :2], atol=1e-6)
        self.assertFalse(j[2, :, 2].any())
        self.assertEqual(f[2, CY_V], 0.0)

    def test_velocity_sign_and_unit(self):
        k, b, v = video([0, 0], centers=[(320, 240), (320, 250)])      # 아래로 10픽셀
        f = compute_features(k, b, v)
        scale = 0.6 * 100 * 2.5                              # 몸통 길이 60픽셀 x 2.5
        self.assertAlmostEqual(f[1, CY_V], 10 / scale * 15 / 3, places=4)
        self.assertAlmostEqual(f[1, HEAD_V], f[1, CY_V], places=4)
        self.assertEqual(f[0, CY_V], 0.0)

    def test_stream_equals_batch(self):
        k, b, v, _ = fall_video()
        v[[7, 8, 30]] = False
        batch = compute_features(k, b, v)
        st = FeatureState()
        one = np.stack([st.step(k[t], b[t]) if v[t] else st.step(None, None) for t in range(len(v))])
        np.testing.assert_array_equal(batch, one)

    def test_causal(self):
        """뒤 프레임을 바꿔도 앞 프레임 특징은 그대로"""
        k, b, v, _ = fall_video()
        a = compute_features(k, b, v)
        k2 = k.copy()
        k2[40:] += 50
        np.testing.assert_array_equal(a[:40], compute_features(k2, b, v)[:40])


class TestExtract(unittest.TestCase):
    def test_target_frames(self):
        np.testing.assert_array_equal(target_frames(9, 30.0), [0, 2, 4, 6, 8])
        np.testing.assert_array_equal(target_frames(4, 15.0), [0, 1, 2, 3])
        t = target_frames(80, 16.0)                          # OF-Syn 16fps, 5초
        self.assertEqual(len(t), 75)
        self.assertTrue((np.abs(t / 16.0 - np.arange(75) / 15.0) <= 0.5 / 16 + 1e-9).all())


class TestLabels(unittest.TestCase):
    def test_frame_labels(self):
        # 걷기 0~1초, 넘어짐 1~2초, 쓰러짐 2~3초, 눕기(lie_down) 3~4초, 누워 있음(lying) 4~5초, 5초 이후 라벨 없음
        y = frame_labels([(0, 1, 0), (1, 2, 1), (2, 3, 2), (3, 4, 5), (4, 5, 6)], 90)
        self.assertEqual(y[:15].tolist(), [0] * 15)
        self.assertEqual(y[15:30].tolist(), [1] * 15)
        self.assertEqual(y[30:45].tolist(), [2] * 15)
        self.assertEqual(y[45:60].tolist(), [0] * 15)        # 천천히 눕는 동작은 정상
        self.assertEqual(y[60:].tolist(), [EXCLUDE] * 30)    # 누워 있는 구간과 라벨 없는 구간은 제외

    def test_windows_and_gather(self):
        c = fake_corpus(4)
        vid, end = c.windows()
        self.assertEqual(end[vid == 0].tolist(), list(range(4, 60, 5)))
        X, Y = gather(c, vid[:1], end[:1])                   # 창 끝 = 4번 프레임: 앞 25칸은 채움
        self.assertFalse(X[0, :25].any())
        self.assertEqual(Y[0, :25].tolist(), [EXCLUDE] * 25)
        np.testing.assert_array_equal(X[0, 25:], c.feat[:5])
        np.testing.assert_array_equal(Y[0, 25:], c.y[:5])
        X, Y = gather(c, np.array([1]), np.array([59]))      # 꽉 찬 창
        np.testing.assert_array_equal(X[0], c.feat[60 + 30:120])

    def test_all_excluded_window_dropped(self):
        c = fake_corpus(2)
        c.y[:30] = EXCLUDE
        vid, end = c.windows()
        self.assertEqual(end[vid == 0].min(), 34)


class TestSelect(unittest.TestCase):
    def pose(self, tracks, T):
        """tracks: {tid: (프레임 목록, 중심)}"""
        fr, tid, kp, bx = [], [], [], []
        for t, (frames, cen) in tracks.items():
            for f in frames:
                k, b = person(0, cen)
                fr.append(f); tid.append(t); kp.append(k); bx.append(b)
        return dict(n_frames=T, fps=15.0, det_frame=np.array(fr), det_tid=np.array(tid),
                    det_kpts=np.array(kp, np.float32), det_box=np.array(bx, np.float32),
                    det_conf=np.full(len(fr), 0.9, np.float32))

    def test_merge_broken_track(self):
        p = self.pose({1: (range(0, 20), (300, 240)), 2: (range(25, 60), (310, 250))}, 60)
        r = select_subject(p)
        self.assertEqual((r["status"], r["n_people"]), ("ok", 1))
        self.assertEqual(int(r["valid"].sum()), 55)

    def test_no_merge_when_far_or_late(self):
        far = self.pose({1: (range(0, 20), (100, 240)), 2: (range(25, 60), (900, 240))}, 60)
        late = self.pose({1: (range(0, 20), (300, 240)), 2: (range(70, 120), (300, 240))}, 120)
        self.assertEqual(select_subject(far)["n_people"], 2)
        self.assertEqual(select_subject(late)["n_people"], 2)

    def test_train_picks_person_in_fall_segment(self):
        y = np.zeros(60, np.int8); y[40:] = 2
        p = self.pose({1: (range(0, 45), (100, 240)), 2: (range(30, 60), (900, 240))}, 60)
        test = select_subject(p, y, use_labels=False)        # 테스트: 가장 오래 잡힌 1번
        train = select_subject(p, y, use_labels=True)        # 학습: 낙상 구간에 20프레임 잡힌 2번 (1번은 5프레임)
        self.assertTrue(test["valid"][0])
        self.assertFalse(train["valid"][0])
        self.assertTrue(train["valid"][59])
        self.assertEqual(train["status"], "ok")

    def test_ambiguous_and_low_coverage(self):
        y = np.zeros(60, np.int8); y[30:] = 1
        both = self.pose({1: (range(0, 60), (100, 240)), 2: (range(0, 60), (900, 240))}, 60)
        self.assertEqual(select_subject(both, y, use_labels=True)["status"], "ambiguous")
        short = self.pose({1: (range(0, 20), (100, 240))}, 60)
        self.assertEqual(select_subject(short)["status"], "low_coverage")
        empty = self.pose({}, 60)
        self.assertEqual(select_subject(empty)["status"], "no_person")


class TestModels(unittest.TestCase):
    def test_shapes_and_params(self):
        x = torch.randn(4, 30, 55)
        for name in MODELS:
            self.assertEqual(tuple(build_model(name)(x).shape), (4, 30, 3), name)
        self.assertTrue(18000 < n_params(build_model("tcnte")) < 25000)   # 제안서: 약 2.2만

    def test_no_future_leak(self):
        for name in MODELS:
            m = build_model(name).eval()
            x = torch.randn(2, 30, 55)
            x2 = x.clone()
            x2[:, 18:] += 3.0
            with torch.no_grad():
                a, b = m(x), m(x2)
            self.assertTrue(torch.allclose(a[:, :18], b[:, :18], atol=1e-5), name)
            self.assertFalse(torch.allclose(a[:, 18:], b[:, 18:], atol=1e-5), name)

    def test_ckpt_roundtrip(self):
        m = build_model("tcnte").eval()
        with tempfile.TemporaryDirectory() as d:
            save_ckpt(Path(d) / "m.pt", m, "tcnte", seed=7)
            m2, ck = load_ckpt(Path(d) / "m.pt")
        x = torch.randn(1, 30, 55)
        with torch.no_grad():
            self.assertTrue(torch.equal(m(x), m2(x)))
        self.assertEqual(ck["seed"], 7)


class TestAugment(unittest.TestCase):
    def test_flip_equals_mirrored_video(self):
        k, b, v, _ = fall_video()
        f = compute_features(k, b, v)
        km, bm = k.copy(), b.copy()
        km[..., 0] = 640 - k[..., 0]
        km = km[:, augment.SWAP]
        bm[:, 0], bm[:, 2] = 640 - b[:, 2], 640 - b[:, 0]
        np.testing.assert_allclose(augment.flip(f[None])[0], compute_features(km, bm, v), atol=1e-5)

    def test_speed(self):
        c = fake_corpus(2)
        vid, end = np.array([0]), np.array([59])
        X1, Y1 = gather(c, vid, end, np.array([1.0]))
        X0, Y0 = gather(c, vid, end)
        np.testing.assert_array_equal(X1, X0)
        np.testing.assert_array_equal(Y1, Y0)
        X, Y = gather(c, vid, end, np.array([1.2]))          # 1.2배속: 30칸에 원본 약 35프레임
        np.testing.assert_allclose(X[0, -1], c.feat[59], atol=1e-6)      # 창 끝은 고정
        np.testing.assert_allclose(X[0, 4], c.feat[29], atol=1e-6)       # 59 - 25 x 1.2 = 29
        self.assertEqual(Y[0, 4], c.y[29])
        Xv = augment.scale_velocity(X0.copy(), np.array([1.2]))
        np.testing.assert_allclose(Xv[0, :, CY_V], np.clip(X0[0, :, CY_V] * 1.2, -1, 1), atol=1e-6)

    def test_drop_joints_shape(self):
        k, b, v, _ = fall_video()
        x = compute_features(k, b, v)[10:40]
        out = augment.drop_joints(x, np.random.default_rng(0))
        J0, J1 = x[:, :51].reshape(30, 17, 3), out[:, :51].reshape(30, 17, 3)
        gone = (J1[..., 2] == 0) & (J0[..., 2] > 0)
        self.assertTrue(1 <= gone.any(0).sum() <= 4)
        for j in np.where(gone.any(0))[0]:
            t0 = int(np.argmax(gone[:, j]))
            hold = J0[max(t0 - 1, 0), j, :2]
            np.testing.assert_allclose(J1[gone[:, j], j, :2], np.tile(hold, (gone[:, j].sum(), 1)), atol=1e-6)
        np.testing.assert_array_equal(out[:, 51:], x[:, 51:])

    def test_batch_runs_with_any_mask(self):
        c = fake_corpus(4)
        vid, end = c.windows()
        rng = np.random.default_rng(0)
        for _ in range(20):                                  # 어떤 증강도 안 걸리는 경우 포함
            sp = augment.sample_speed(3, rng)
            X, _ = gather(c, vid[:3], end[:3], sp)
            self.assertEqual(augment.augment_batch(X, sp, rng).shape, (3, 30, 55))


class TestTrainEval(unittest.TestCase):
    def test_focal_loss(self):
        logits = torch.zeros(1, 4, 3)
        y = torch.tensor([[0, 1, 2, -1]])
        alpha = torch.tensor([1.0, 2.0, 3.0])
        want = np.mean([a * (1 - 1 / 3) ** 3 * np.log(3) for a in (1, 2, 3)])   # 제외 프레임은 빠짐
        self.assertAlmostEqual(focal_loss(logits, y, alpha, 3.0).item(), want, places=5)

    def test_real_fraction(self):
        c = fake_corpus(12)                                  # 영상 10개 합성, 2개 실제
        vid, end = c.windows()
        w, cnt = class_weights(c, vid, end)
        self.assertGreater(w[1], w[0])                       # 적은 클래스가 더 큰 가중치
        self.assertLessEqual(w.max(), 30.0)
        order = epoch_order(c, vid, np.random.default_rng(0), 0.5)
        real = np.array([d == "GMDCSA24" for d in c.dataset])[vid][order]
        self.assertAlmostEqual(real.mean(), 0.5, delta=0.12)
        c.dataset = ["GMDCSA24"] * 12                        # 실제 영상뿐이면 섞기만 한다
        self.assertEqual(sorted(epoch_order(c, vid, np.random.default_rng(0), 0.2)), list(range(len(vid))))

    def test_metrics(self):
        y = np.array([0, 0, 1, 2, 2, -1])
        p = np.array([0, 1, 1, 2, 0, 2])
        m = frame_metrics(y, p)
        self.assertEqual(m["n_frames"], 5)
        self.assertEqual(m["confusion"], [[1, 1, 0], [0, 1, 0], [1, 0, 1]])
        self.assertAlmostEqual(m["macro_f1"], (0.5 + 2 / 3 + 2 / 3) / 3, places=6)

    def test_stream_paths_agree(self):
        """영상 통째 예측, Corpus 예측, 한 프레임씩 넣는 StreamClassifier가 같은 확률을 낸다"""
        c = fake_corpus(2)
        k, b, v, _ = fall_video(60, 15, 6)
        v[[20, 21]] = False
        feat = compute_features(k, b, v)
        c.feat[:60] = feat
        with tempfile.TemporaryDirectory() as d:
            save_ckpt(Path(d) / "m.pt", build_model("tcnte"), "tcnte")
            clf = StreamClassifier(Path(d) / "m.pt")
        whole = predict_stream(clf.model, feat)
        np.testing.assert_allclose(predict_corpus(clf.model, c)[:60], whole, atol=1e-5)
        live = np.stack([clf.step(1, k[t], b[t]) if v[t] else clf.step(1, None, None) for t in range(60)])
        np.testing.assert_allclose(live, whole, atol=1e-5)

    def test_summarize_merges_parallel_seeds(self):
        """시드마다 따로 학습한 결과를 모아 검증 점수가 가장 높은 시드를 best.pt로 고른다"""
        import json
        from types import SimpleNamespace
        from train import summarize
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for s, f in [(0, 0.5), (1, 0.9), (2, 0.7)]:
                save_ckpt(d / f"seed{s}.pt", build_model("gru"), "gru", seed=s)
                json.dump(dict(seed=s, val_macro_f1=f, epoch=3), open(d / f"seed{s}.json", "w"))
            summarize(d, SimpleNamespace(model="gru", datasets=["x"]))
            self.assertEqual(load_ckpt(d / "best.pt")[1]["seed"], 1)
            sm = json.load(open(d / "summary.json"))
            self.assertEqual((sm["best_seed"], len(sm["seeds"])), (1, 3))
            self.assertAlmostEqual(sm["val_macro_f1_mean"], 0.7, places=6)

    def test_learns_fake_falls(self):
        """가짜 낙상 영상으로 짧게 학습하면 검증 영상의 세 클래스를 대부분 맞힌다"""
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        tr, va = fake_corpus(40, seed=1), fake_corpus(12, seed=2)
        vid, end = tr.windows()
        w, _ = class_weights(tr, vid, end)
        model = build_model("tcnte")
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        first = last = None
        for ep in range(40):
            model.train()
            tot = 0.0
            order = rng.permutation(len(vid))
            for i in range(0, len(order), 64):
                bi = order[i:i + 64]
                sp = augment.sample_speed(len(bi), rng)
                X, Y = gather(tr, vid[bi], end[bi], sp)
                X = augment.augment_batch(X, sp, rng)
                opt.zero_grad()
                loss = focal_loss(model(torch.from_numpy(X)), torch.from_numpy(Y),
                                  torch.tensor(w, dtype=torch.float32), 3.0)
                loss.backward()
                opt.step()
                tot += loss.item()
            first = tot if first is None else first
            last = tot
        self.assertLess(last, first * 0.5)
        m = frame_metrics(va.y, predict_corpus(model, va).argmax(1))
        self.assertGreater(m["macro_f1"], 0.8, m)


if __name__ == "__main__":
    unittest.main()
