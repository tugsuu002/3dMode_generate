#!/usr/bin/env bash
# COLMAP фотограмметрийн бүрэн процесс (Docker + CUDA).
# Хэрэглээ: ./recon.sh <job_dir>
#   <job_dir>/images/ дотор зурагнууд байх ёстой.
set -euo pipefail

JOB="${1:?job_dir шаардлагатай}"
JOB="$(cd "$JOB" && pwd)"
IMG_DIR="$JOB/images"
# colmap-cudnn нь албан ёсны образ дээр libcudnn9 нэмсэн хувилбар
# (Dockerfile.cudnn-ээс барина). Байгаа бол түүнийг сонгоно — ALIKED/LightGlue
# GPU дээр ажиллана. Байхгүй бол албан ёсны образ руу буцна.
if [ -z "${COLMAP_IMAGE:-}" ] && docker image inspect colmap-cudnn:latest >/dev/null 2>&1; then
  IMAGE=colmap-cudnn:latest
  HAS_CUDNN=1
else
  IMAGE="${COLMAP_IMAGE:-colmap/colmap:latest}"
  HAS_CUDNN=0
fi

# --- GTX 1050 Ti = 4GB VRAM. Эдгээр хязгаарыг түүнд тааруулсан. ---
# Онцлог ялгагчийн горим:
#   sift       — стандарт, хурдан. Текстуртэй объектод хангалттай.
#   sift-hard  — илүү олон цэг, доогуур босго. Удаан (CPU SIFT) ч мэдрэг.
#   aliked     — сурсан (deep) ялгагч + LightGlue тааруулагч.
#                Бараан, гөлгөр, текстур багатай гадаргуунд хамгийн сайн.
FEATURES="${FEATURES:-sift}"
FEAT_MAX="${FEAT_MAX:-1600}"     # ялгагчид өгөх зургийн дээд тал
DENSE_MAX="${DENSE_MAX:-1000}"   # нягт стерео — VRAM-д хамгийн мэдрэг параметр
CACHE_GB="${CACHE_GB:-4}"
# Poisson-ий стандарт (depth=13, trim=10) нь их, цэвэрхэн цэгэн үүлд зориулагдсан.
# Цөөн/шуугиантай цэгэнд бүх зүйлийг таслаад хоосон тор үлдээдэг.
POISSON_DEPTH="${POISSON_DEPTH:-10}"
POISSON_TRIM="${POISSON_TRIM:-6}"

log(){ echo "[$(date +%H:%M:%S)] $*"; }
stage(){ echo "@@STAGE $1"; log "── $2"; }

# --- GPU-ийн дараалал ---
# Нэг GPU дээр хэд хэдэн ажил зэрэг явбал бие биенээ удаашруулж,
# VRAM дүүрвэл унадаг. Тиймээс нэг дор зөвхөн нэг ажил гүйнэ.
LOCKFILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.gpu.lock"
exec 9>"$LOCKFILE"
if ! flock -n 9; then
  log "Өөр ажил GPU ашиглаж байна — дараалалд орлоо, хүлээж байна…"
  flock 9
  log "Дараалал дуусаж, эхэллээ"
fi

N=$(find "$IMG_DIR" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
log "Зураг: $N ширхэг · образ: $IMAGE"
[ "$N" -ge 8 ] || { echo "АЛДАА: дор хаяж 8 зураг хэрэгтэй (одоо $N)."; exit 2; }
[ "$N" -le 150 ] || log "АНХААР: $N зураг олон байна — тааруулалт (O(n²)) удаан болно."

FREE_GB=$(df -BG --output=avail "$JOB" | tail -1 | tr -dc '0-9')
log "Сул диск: ${FREE_GB} GB"
[ "$FREE_GB" -ge 6 ] || { echo "АЛДАА: диск хүрэлцэхгүй (${FREE_GB} GB). Дор хаяж 6 GB хэрэгтэй."; exit 3; }

# Docker дотор COLMAP-ыг таны эрхээр ажиллуулна → гарц нь root-ийнх болохгүй
CACHE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cache"
mkdir -p "$CACHE"

STEP_TIMEOUT="${STEP_TIMEOUT:-7200}"

dcolmap(){
  timeout --signal=KILL "$STEP_TIMEOUT" \
  docker run --rm --gpus all \
    -u "$(id -u):$(id -g)" \
    -v "$JOB":/work -w /work \
    -v "$CACHE":/cache -e HOME=/cache -e XDG_CACHE_HOME=/cache \
    "$IMAGE" colmap "$@"
}

# COLMAP 4.x дээр SiftExtraction/SiftMatching нь FeatureExtraction/FeatureMatching
# болж нэр солигдсон. Образ дээрх бодит нэрийг асууж тодорхойлно.
log "COLMAP-ийн сонголтын нэрийг тодорхойлж байна…"
HELP="$(docker run --rm --entrypoint bash "$IMAGE" -c 'colmap feature_extractor -h' 2>&1 || true)"
if grep -q 'FeatureExtraction\.max_image_size' <<<"$HELP"; then
  EXT=FeatureExtraction; MAT=FeatureMatching
else
  EXT=SiftExtraction;    MAT=SiftMatching
fi
VER="$(grep -oE 'COLMAP [0-9][^ ]*' <<<"$HELP" | head -1 || true)"
log "${VER:-COLMAP} · сонголтын угтвар: $EXT / $MAT"

rm -rf "$JOB/sparse" "$JOB/dense" "$JOB/db.db"
mkdir -p "$JOB/sparse"

EXT_ARGS=(); MAT_ARGS=(); MAP_ARGS=()
case "$FEATURES" in
  aliked)
    # libcudnn9 байхгүй образ дээр ONNX-ийн CUDA хэсэг ачаалагдахгүй бөгөөд
    # COLMAP нь CPU руу шилжихийн оронд abort() хийж docker-ийг өлгөдөг.
    # Тиймээс cuDNN-гүй бол зориуд CPU-г сонгоно.
    if [ "$HAS_CUDNN" = 1 ]; then AGPU=1; else AGPU=0; fi
    log "Горим: ALIKED_N32 + LightGlue ($([ "$AGPU" = 1 ] && echo GPU || echo CPU))"
    EXT_ARGS=( --$EXT.type ALIKED_N32
               --$EXT.use_gpu "$AGPU"
               --AlikedExtraction.max_num_features 4096
               --AlikedExtraction.min_score 0.10 )
    MAT_ARGS=( --$MAT.type ALIKED_LIGHTGLUE
               --$MAT.use_gpu "$AGPU" )
    ;;
  sift-hard)
    log "Горим: SIFT (мэдрэг тохиргоо — удаан)"
    # affine_shape ба domain_size_pooling нь CPU SIFT руу шилжүүлдэг
    EXT_ARGS=( --SiftExtraction.max_num_features 16384
               --SiftExtraction.peak_threshold 0.003
               --SiftExtraction.estimate_affine_shape 1
               --SiftExtraction.domain_size_pooling 1 )
    ;;
  *)
    log "Горим: SIFT (стандарт)"
    EXT_ARGS=( --$EXT.use_gpu 1 )
    MAT_ARGS=( --$MAT.use_gpu 1 )
    ;;
esac
if [ "$FEATURES" != "sift" ]; then
  # хүнд тохиолдолд SfM-ийн босгыг сулруулна
  MAP_ARGS=( --Mapper.min_num_matches 10
             --Mapper.init_min_num_inliers 50
             --Mapper.abs_pose_min_num_inliers 20
             --Mapper.filter_max_reproj_error 6 )
fi

stage 1 "Онцлог цэг ялгаж байна ($FEATURES)"
dcolmap feature_extractor \
  --database_path /work/db.db --image_path /work/images \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model SIMPLE_RADIAL \
  --$EXT.max_image_size "$FEAT_MAX" \
  "${EXT_ARGS[@]}"

stage 2 "Зургуудыг хооронд нь тааруулж байна"
dcolmap exhaustive_matcher --database_path /work/db.db "${MAT_ARGS[@]}"

stage 3 "Камерын байрлалыг сэргээж байна (SfM)"
dcolmap mapper \
  --database_path /work/db.db --image_path /work/images --output_path /work/sparse \
  "${MAP_ARGS[@]}"

[ -d "$JOB/sparse/0" ] || {
  echo "АЛДАА: SfM бүтсэнгүй. Зургууд хоорондоо хангалттай давхцаагүй байж магадгүй."
  echo "Зөвлөмж: тойрч авахдаа зэргэлдээ хоёр зураг 60-80% давхцаж байх ёстой."
  exit 4; }

REG=$(dcolmap model_analyzer --path /work/sparse/0 2>&1 | grep -i "Registered images" | head -1 || true)
log "${REG:-Бүртгэгдсэн зураг: тодорхойгүй}"

stage 4 "Зургийн гажилтыг арилгаж байна"
dcolmap image_undistorter \
  --image_path /work/images --input_path /work/sparse/0 \
  --output_path /work/dense --output_type COLMAP --max_image_size "$FEAT_MAX"

stage 5 "Нягт гүний зураглал (хамгийн удаан хэсэг)"
dcolmap patch_match_stereo \
  --workspace_path /work/dense --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency true \
  --PatchMatchStereo.max_image_size "$DENSE_MAX" \
  --PatchMatchStereo.cache_size "$CACHE_GB"

stage 6 "Цэгэн үүл нэгтгэж байна"
dcolmap stereo_fusion \
  --workspace_path /work/dense --workspace_format COLMAP \
  --input_type geometric --output_path /work/dense/fused.ply

NPTS=$(head -c 4000 "$JOB/dense/fused.ply" | grep -aoE 'element vertex [0-9]+' | grep -oE '[0-9]+' || echo 0)
log "Нягт цэгэн үүл: ${NPTS} цэг"
[ "${NPTS:-0}" -ge 3000 ] || log "АНХААР: цэг цөөн байна — гадаргуу муу гарах магадлалтай."

stage 7 "Гадаргуу барьж байна (Poisson · depth=$POISSON_DEPTH trim=$POISSON_TRIM)"
dcolmap poisson_mesher \
  --input_path /work/dense/fused.ply --output_path /work/dense/mesh.ply \
  --PoissonMeshing.depth "$POISSON_DEPTH" \
  --PoissonMeshing.trim "$POISSON_TRIM" || true

MV=$(head -c 2000 "$JOB/dense/mesh.ply" 2>/dev/null | grep -aoE 'element vertex [0-9]+' | grep -oE '[0-9]+' || echo 0)
if [ "${MV:-0}" -lt 1000 ]; then
  log "Poisson хоосон тор гаргалаа (${MV:-0} орой) — Delaunay-гаар оролдож байна"
  dcolmap delaunay_mesher --input_path /work/dense --output_path /work/dense/mesh.ply || true
  MV=$(head -c 2000 "$JOB/dense/mesh.ply" 2>/dev/null | grep -aoE 'element vertex [0-9]+' | grep -oE '[0-9]+' || echo 0)
fi
log "Торны орой: ${MV:-0}"
[ "${MV:-0}" -ge 1000 ] || log "АНХААР: тор бүтсэнгүй. Цэгэн үүл (fused.ply) нь ашиглах боломжтой хэвээр."

stage 8 "Дууслаа"
ls -lh "$JOB/dense/fused.ply" "$JOB/dense/mesh.ply" 2>/dev/null || true
echo "@@DONE"
