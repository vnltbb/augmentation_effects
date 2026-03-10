import os
import shutil
import random
import hashlib
import cv2
import csv
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
SAMPLE_IMAGES = {}
AUG_STATS = defaultdict(int)

AUG_STATS_BY_CLASS = defaultdict(lambda: defaultdict(int))
# technique -> prefix id (for filename)
# NOTE: This mapping is overwritten at runtime from AUG_CONFIG order (see run())
TECH_PREFIX = {
    'Rain': 0,
    'Shadow': 1,
    'Snow': 9,
    'RGBShift': 2,
    'Solarize': 3,
    'Sepia': 4,
    'Blur': 5,
    'Geo_Transform': 6,
}


def get_active_prefix_map(config):
    """Return active technique -> prefix using the user-declared TECH_PREFIX.

    Policy:
    - Preserve the numeric labels declared in TECH_PREFIX.
    - Ignore disabled techniques.
    - Treat any enabled Geo_* combination as Geo_Transform.
    - Fail fast if an enabled technique has no prefix or if prefixes collide.
    """
    active = {}

    geo_enabled = any(_is_enabled(config.get(k, False)) for k in ['Geo_Rotate', 'Geo_Shift', 'Geo_Scale'])
    for tech in ['Rain', 'Shadow', 'Snow', 'RGBShift', 'Solarize', 'Sepia', 'Blur']:
        if _is_enabled(config.get(tech, False)):
            if tech not in TECH_PREFIX:
                raise KeyError(f"[ConfigError] Enabled technique '{tech}' is missing in TECH_PREFIX")
            active[tech] = TECH_PREFIX[tech]

    if geo_enabled:
        if 'Geo_Transform' not in TECH_PREFIX:
            raise KeyError("[ConfigError] Geo transform is enabled, but 'Geo_Transform' is missing in TECH_PREFIX")
        active['Geo_Transform'] = TECH_PREFIX['Geo_Transform']

    prefixes = list(active.values())
    if len(prefixes) != len(set(prefixes)):
        dupes = sorted({x for x in prefixes if prefixes.count(x) > 1})
        raise ValueError(f"[ConfigError] Duplicate prefix id(s) in active TECH_PREFIX: {dupes}")

    return active

def get_image_paths(cls_path):
    if not os.path.exists(cls_path):
        return []
    return [
        os.path.join(cls_path, f)
        for f in sorted(os.listdir(cls_path))
        if f.lower().endswith(VALID_EXTENSIONS)
    ]

# -----------------------------------------------------------
# 1. Count Logic (클래스별 Max Count 계산)  [Split 로직 제거]
# -----------------------------------------------------------
def get_needed_counts(base_dir):
    """
    base_dir 내의 클래스 폴더들을 스캔하여 데이터 수를 세고,
    Max Count에 맞추기 위한 부족분(diff)을 계산합니다.

    기대 구조:
      base_dir/class_name/image.png
    """
    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    counts = {}
    for cls in classes:
        counts[cls] = len(get_image_paths(os.path.join(base_dir, cls)))

    if not counts:
        return {}

    max_cnt = max(counts.values())
    print(f"[DATASET] Max Class Size: {max_cnt}")

    needed = {}
    for cls, cnt in counts.items():
        diff = max_cnt - cnt
        if diff > 0:
            needed[cls] = diff
    return needed

# -----------------------------------------------------------
# -----------------------------------------------------------
# 1-2. Fixed Count Logic (클래스당 증강량 고정)
# -----------------------------------------------------------
def get_fixed_counts(base_dir, per_class_aug):
    """base_dir 내 *각 클래스마다* per_class_aug 장을 생성합니다.

    - 예) per_class_aug=500 이면, 클래스가 N개일 때 총 생성량은 500*N
    - per_class_aug <= 0 이면 빈 dict
    """
    if per_class_aug is None or per_class_aug <= 0:
        return {}

    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    classes = sorted(classes)
    if not classes:
        return {}

    fixed = {cls: int(per_class_aug) for cls in classes}
    return fixed


# -----------------------------------------------------------
# 2. Augmentation Logic (Core)
# -----------------------------------------------------------
def _is_enabled(v):
    # bool or dict with {"enable": bool}
    if isinstance(v, dict):
        return bool(v.get("enable", False))
    return bool(v)

def _get_param(v, key, default):
    if isinstance(v, dict) and key in v:
        return v[key]
    return default

def _sample_range(val, *, is_int=False):
    """
    val:
      - scalar -> 그대로
      - (min, max) / [min, max] -> random uniform/int
    """
    if isinstance(val, (tuple, list)) and len(val) == 2:
        lo, hi = val
        if is_int:
            lo_i = int(round(lo))
            hi_i = int(round(hi))
            if lo_i > hi_i:
                lo_i, hi_i = hi_i, lo_i
            return random.randint(lo_i, hi_i)
        else:
            lo_f = float(lo)
            hi_f = float(hi)
            if lo_f > hi_f:
                lo_f, hi_f = hi_f, lo_f
            return random.uniform(lo_f, hi_f)
    return int(val) if is_int else float(val)

def _normalize_rgbshift_limit(val):
    """Albumentations RGBShift expects limits in [-1, 1] (normalized to 1.0 == 255).
    Accepts:
      - scalar (int/float): treated as (-abs(v), +abs(v))
      - (min, max) / [min, max]
    Also accepts pixel-range values (e.g., -20..20); they will be auto-normalized by /255.
    """
    if isinstance(val, (tuple, list)) and len(val) == 2:
        lo, hi = float(val[0]), float(val[1])
    else:
        v = float(val)
        lo, hi = -abs(v), abs(v)

    # auto-normalize if user gave pixel range
    if max(abs(lo), abs(hi)) > 1.0:
        lo /= 255.0
        hi /= 255.0

    # ensure order and clamp
    if lo > hi:
        lo, hi = hi, lo
    lo = max(-1.0, min(1.0, lo))
    hi = max(-1.0, min(1.0, hi))
    return (lo, hi)

def _rng_choice(rng, seq):
    return seq[rng.randrange(len(seq))]

def _hash_image_rgb(img_rgb):
    # strict pixel-level hash: includes shape and dtype
    h = hashlib.sha256()
    h.update(str(img_rgb.shape).encode('utf-8'))
    h.update(str(img_rgb.dtype).encode('utf-8'))
    h.update(img_rgb.tobytes())
    return h.hexdigest() 

def apply_augmentation(src_root, dest_root, cls, count, config, weather_ratio, rng_select, rng_method, rng_name, seen_hashes, used_pairs):
    """
    src_root/cls 의 원본 이미지를 사용하여
    dest_root/cls 에 count 만큼 증강 이미지 저장

    저장 구조(요구사항):
      - 원본 데이터는 유지
      - 증강 데이터는 dataset_aug/class/image.png 로 저장
    """
    if count <= 0:
        return

    # ---- Weather transform builders (강도/범위는 config에서 샘플링) ----
    def build_rain():
        c = config.get('Rain', {})
        return A.RandomRain(
            brightness_coefficient=_sample_range(_get_param(c, 'brightness_coefficient', 0.9)),
            drop_width=_sample_range(_get_param(c, 'drop_width', 1), is_int=True),
            blur_value=_sample_range(_get_param(c, 'blur_value', 3), is_int=True),
            p=1
        )

    def build_shadow():
        c = config.get('Shadow', {})
        return A.RandomShadow(
            num_shadows_lower=_sample_range(_get_param(c, 'num_shadows_lower', 1), is_int=True),
            num_shadows_upper=_sample_range(_get_param(c, 'num_shadows_upper', 1), is_int=True),
            shadow_dimension=_sample_range(_get_param(c, 'shadow_dimension', 5), is_int=True),
            p=1
        )

    def build_snow():
        c = config.get('Snow', {})
        return A.RandomSnow(
            brightness_coeff=_sample_range(_get_param(c, 'brightness_coeff', 2.5)),
            p=1
        )

    def build_rgbshift():
        c = config.get('RGBShift', {})
        # NOTE: RGBShift in recent albumentations validates ranges in [-1, 1] (normalized).
        # We pass ranges (not sampled scalars) to avoid min/max inversion errors.
        return A.RGBShift(
            r_shift_limit=_normalize_rgbshift_limit(_get_param(c, 'r_shift_limit', (-20, 20))),
            g_shift_limit=_normalize_rgbshift_limit(_get_param(c, 'g_shift_limit', (-20, 20))),
            b_shift_limit=_normalize_rgbshift_limit(_get_param(c, 'b_shift_limit', (-20, 20))),
            p=1
        )

    def build_solarize():
        return A.Solarize(p=1)

    def build_sepia():
        return A.ToSepia(p=1)

    def build_blur():
        c = config.get('Blur', {})
        blur_limit = _get_param(c, 'blur_limit', 7)
        # albumentations Blur: blur_limit can be int or (min,max)
        if isinstance(blur_limit, (tuple, list)) and len(blur_limit) == 2:
            blur_limit = tuple(int(round(x)) for x in blur_limit)
        else:
            blur_limit = int(round(blur_limit))
        return A.Blur(blur_limit=blur_limit, p=1)

    weather_builders = {
        'Rain': build_rain,
        'Shadow': build_shadow,
        'Snow': build_snow,
        'RGBShift': build_rgbshift,
        'Solarize': build_solarize,
        'Sepia': build_sepia,
        'Blur': build_blur,
    }

    # ---- Geometric transform builders ----
    def build_geo_transform():
        c_shift = config.get('Geo_Shift', {})
        c_scale = config.get('Geo_Scale', {})
        c_rot = config.get('Geo_Rotate', {})

        shift_limit = 0.0
        if _is_enabled(c_shift):
            shift_limit = _get_param(c_shift, 'shift_limit', 0.1)

        scale_limit = 0.0
        if _is_enabled(c_scale):
            scale_limit = _get_param(c_scale, 'scale_limit', 0.2)

        rotate_limit = 0
        if _is_enabled(c_rot):
            rotate_limit = _get_param(c_rot, 'rotate_limit', 30)

        # ShiftScaleRotate: 각 인자에 (min,max) 가능
        return A.ShiftScaleRotate(
            shift_limit=shift_limit,
            scale_limit=scale_limit,
            rotate_limit=rotate_limit,
            border_mode=cv2.BORDER_REFLECT,
            p=1
        )

    geo_builders = {}
    if any(_is_enabled(config.get(k, False)) for k in ['Geo_Rotate', 'Geo_Shift', 'Geo_Scale']):
        geo_builders['Geo_Transform'] = build_geo_transform

    # 활성화된 기법 필터링
    active_weather = [k for k in weather_builders.keys() if _is_enabled(config.get(k, False))]
    active_geo = list(geo_builders.keys())

    # 경로 설정
    src_cls_path = os.path.join(src_root, cls)
    dest_cls_path = os.path.join(dest_root, cls)
    os.makedirs(dest_cls_path, exist_ok=True)

    src_files = get_image_paths(src_cls_path)
    if not src_files:
        return

    # 증강 실행
    # Allocation rule:
    # - If both Weather and Geo are active: split by weather_ratio
    # - If only one category is active: assign all 'count' to that category
    if active_weather and active_geo:
        n_weather = int(count * weather_ratio)
        n_geo = count - n_weather
    elif active_weather:
        n_weather = count
        n_geo = 0
    elif active_geo:
        n_weather = 0
        n_geo = count
    else:
        return

    tasks = [
        ('Weather', n_weather, active_weather, weather_builders),
        ('Geo', n_geo, active_geo, geo_builders),
    ]

    for cat_name, n_tasks, tech_list, builder_map in tasks:
        if not tech_list or n_tasks <= 0:
            continue

        generated = 0
        attempts = 0
        max_attempts = max(n_tasks * 30, 200)

        # track "condition tokens" (img_path, technique) to avoid immediate repeats during retries
        if isinstance(used_pairs, dict):
            used_pairs_cat = used_pairs.setdefault(cat_name, set())
        else:
            used_pairs_cat = used_pairs

        # If the combination space is small, don't permanently block all pairs.
        total_pairs = len(src_files) * len(tech_list)

        while generated < n_tasks and attempts < max_attempts:
            attempts += 1

            tech = _rng_choice(rng_method, tech_list)
            img_path = _rng_choice(rng_select, src_files)

            pair = (img_path, tech)
            if pair in used_pairs_cat:
                # if exhausted, reset and continue
                if total_pairs > 0 and len(used_pairs_cat) >= total_pairs:
                    used_pairs_cat.clear()
                else:
                    continue
            used_pairs_cat.add(pair)

            aug_func = builder_map[tech]()  # 강도/범위 샘플링 포함

            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            aug_img = aug_func(image=image)['image']

            # strict duplicate check (pixel-level)
            img_hash = _hash_image_rgb(aug_img)
            if img_hash in seen_hashes:
                continue

            # Save
            prefix = TECH_PREFIX.get(tech, 99)
            base_name = os.path.basename(img_path)
            fname = f"{prefix}-{base_name}"
            out_path = os.path.join(dest_cls_path, fname)
            if os.path.exists(out_path):
                stem, ext = os.path.splitext(base_name)
                fname = f"{prefix}-{stem}_{rng_name.randint(100000, 999999)}{ext}"
                out_path = os.path.join(dest_cls_path, fname)

            ok = cv2.imwrite(
                out_path,
                cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
            )
            if not ok:
                continue

            seen_hashes.add(img_hash)
            generated += 1

            # Logging
            AUG_STATS[tech] += 1
            AUG_STATS_BY_CLASS[cls]['total-augmented'] += 1
            AUG_STATS_BY_CLASS[cls][tech] += 1
            if tech not in SAMPLE_IMAGES:
                SAMPLE_IMAGES[tech] = aug_img

        if generated < n_tasks:
            print(f"[Warn] {cls}/{cat_name}: generated {generated}/{n_tasks} (attempts={attempts}, pairs={total_pairs})")


# -----------------------------------------------------------
# 3. Utilities
# -----------------------------------------------------------
def save_report(dest_root, active_prefix_map=None):
    # CSV
    with open(os.path.join(dest_root, 'aug_stats.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Technique', 'Count'])
        for k, v in AUG_STATS.items():
            writer.writerow([k, v])

    # CSV (by class)
    if AUG_STATS_BY_CLASS:
        # collect all techniques used (exclude the total column)
        techs = set()
        for _, d in AUG_STATS_BY_CLASS.items():
            for k in d.keys():
                if k != 'total-augmented':
                    techs.add(k)
        sort_map = active_prefix_map or TECH_PREFIX
        techs = sorted(list(techs), key=lambda x: sort_map.get(x, 999))
        out_csv = os.path.join(dest_root, 'aug_stats_by_class.csv')
        with open(out_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Class', 'total-augmented'] + techs)
            for cls in sorted(AUG_STATS_BY_CLASS.keys()):
                row = [cls, AUG_STATS_BY_CLASS[cls].get('total-augmented', 0)]
                row += [AUG_STATS_BY_CLASS[cls].get(t, 0) for t in techs]
                writer.writerow(row)

        # console preview
        header = ['Class', 'total-augmented'] + techs
        print("\n[Augment Stats] By Class x Technique")
        print("|" + "|".join(header) + "|")
        print("|" + "|".join(['---'] * len(header)) + "|")
        for cls in sorted(AUG_STATS_BY_CLASS.keys()):
            row = [cls, str(AUG_STATS_BY_CLASS[cls].get('total-augmented', 0))]
            row += [str(AUG_STATS_BY_CLASS[cls].get(t, 0)) for t in techs]
            print("|" + "|".join(row) + "|")

    # Figure
    if not SAMPLE_IMAGES:
        return
    keys = list(SAMPLE_IMAGES.keys())
    rows = (len(keys) + 2) // 3
    plt.figure(figsize=(12, 4 * rows))
    for i, k in enumerate(keys):
        plt.subplot(rows, 3, i + 1)
        plt.imshow(SAMPLE_IMAGES[k])
        plt.title(k)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(dest_root, 'aug_examples.jpg'))
    print("[Done] Report saved.")

# -----------------------------------------------------------
# Main Execution
# -----------------------------------------------------------
def run(src_dir, dest_dir, weather_ratio, config, total_aug=None, seed=None):
    # 0. Global stats reset (for re-runs in same process)
    SAMPLE_IMAGES.clear()
    AUG_STATS.clear()
    AUG_STATS_BY_CLASS.clear()

    # 0.1 Reproducibility (pragmatic)
    # - Same seed + same environment + same input dir => same sampling / filenames / counts
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 0.2 Prefix mapping: preserve the user-declared TECH_PREFIX
    active_prefix_map = get_active_prefix_map(config)
    if active_prefix_map:
        print("[Prefix Map] Active technique labels")
        for tech, prefix in sorted(active_prefix_map.items(), key=lambda kv: kv[1]):
            print(f"  - {prefix}: {tech}")
    else:
        print("[Prefix Map] No active techniques")

    # 1. 증강 폴더 초기화
    if os.path.exists(dest_dir):
        # Safety guard: prevent deleting dangerous paths
        abs_dest = os.path.abspath(dest_dir)
        abs_src = os.path.abspath(src_dir)
        if abs_dest in [os.path.abspath('.'), os.path.abspath('..'), os.path.sep] or abs_dest == abs_src:
            raise ValueError(f"[SafetyError] Refuse to delete dest_dir='{dest_dir}' (abs='{abs_dest}')")
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    # 2. 부족분 계산 (클래스 전체 기준)
    needed_info = get_needed_counts(src_dir)
    if total_aug is not None:
        needed_info = get_fixed_counts(src_dir, total_aug)

    # 3. 증강 수행
    print(f"[Start] Augmenting... (Weather Ratio: {weather_ratio}) | Per-class: {total_aug}")
    for cls, count in needed_info.items():
        print(f"  - {cls}: Generating +{count}")
        # per-class dedupe state
        seen_hashes = set()
        used_pairs = {'Weather': set(), 'Geo': set()}

        # stable per-class RNG streams
        if seed is None:
            rng_select = random.Random()
            rng_method = random.Random()
            rng_name = random.Random()
        else:
            cls_salt = int(hashlib.md5(cls.encode('utf-8')).hexdigest()[:8], 16)
            rng_select = random.Random(seed + cls_salt + 0)
            rng_method = random.Random(seed + cls_salt + 1)
            rng_name = random.Random(seed + cls_salt + 2)

        apply_augmentation(src_dir, dest_dir, cls, count, config, weather_ratio, rng_select, rng_method, rng_name, seen_hashes, used_pairs)

    save_report(dest_dir, active_prefix_map=active_prefix_map)
    print("\n[Complete] Structure:")
    print(f" Original:  {src_dir}/<class>")
    print(f" Augmented: {dest_dir}/<class>")

if __name__ == "__main__":
    # 설정
    SRC_DIR = './dataset'   # 원본: dataset/class/image.png
    DEST_DIR = './dataset_aug'      # 증강: dataset_aug/class/image.png
    WEATHER_RATIO = 0.7

    # 각 증강기법 강도/범위는 여기서 조절
    # - enable: 사용 여부
    # - 값이 (min, max)면 매 샘플마다 랜덤 샘플링
    AUG_CONFIG = {
        # ---- Weather 계열 ----
        'Rain': {
            'enable': True,
            'brightness_coefficient': (0.7, 1.0),
            'drop_width': (1, 2),
            'blur_value': (3, 5),
        },
        'Shadow': {
            'enable': True,
            'num_shadows_lower': (1, 2),
            'num_shadows_upper': (1, 4),
            'shadow_dimension': (4, 6),
        },
        'Snow': {
            'enable': False,
            'brightness_coeff': (2.0, 3.0),
        },
        'RGBShift': {
            'enable': True,
            'r_shift_limit': (-10, 10),
            'g_shift_limit': (-10, 10),
            'b_shift_limit': (-10, 10),
        },
        'Solarize': {'enable': False},
        'Sepia': {'enable': False},
        'Blur': {
            'enable': True,
            'blur_limit': (3, 7),
        },

        # ---- Geo 계열 (ShiftScaleRotate는 아래 3개 enable 조합으로 구성) ----
        'Geo_Rotate': {'enable': True, 'rotate_limit': (-30, 30)},
        'Geo_Shift': {'enable': True, 'shift_limit': (-0.05, 0.10)},
        'Geo_Scale': {'enable': True, 'scale_limit': (-0.05, 0.20)},
    }

    AUG_PER_CLASS = 500  # 클래스당 증강 이미지 생성량

    SEED = 42  # 재현성: 동일 seed면 동일 결과(현실적 범위)

    run(SRC_DIR, DEST_DIR, WEATHER_RATIO, AUG_CONFIG, total_aug=AUG_PER_CLASS, seed=SEED)
