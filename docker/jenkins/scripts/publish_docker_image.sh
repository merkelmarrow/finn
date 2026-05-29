#!/bin/bash
# publish_docker_image.sh <image_dir> <tag> <build_number>
#
# Save the named Docker image to <image_dir>/finn-docker-image.tar.gz and
# write its tag to a sibling file, atomically. The flock guards a same-build
# retry: two concurrent runs of this script on the same image_dir serialise
# on .publish.lock and write to per-build .tmp- files, then rename into
# place.
set -eo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: publish_docker_image.sh <image_dir> <tag> <build_number>" >&2
  exit 2
fi

image_dir=$1
tag=$2
build=$3

tmp_img="${image_dir}/finn-docker-image.tar.gz.tmp-${build}"
tmp_tag="${image_dir}/finn-docker-tag.txt.tmp-${build}"
final_img="${image_dir}/finn-docker-image.tar.gz"
final_tag="${image_dir}/finn-docker-tag.txt"

exec 9>"${image_dir}/.publish.lock"
flock -x 9

rm -f "$tmp_img" "$tmp_tag"
if command -v pigz >/dev/null 2>&1; then
  docker save "$tag" | pigz -p "$(nproc)" > "$tmp_img"
else
  docker save "$tag" | gzip > "$tmp_img"
fi
printf '%s\n' "$tag" > "$tmp_tag"
sync "$tmp_img" "$tmp_tag"
mv "$tmp_img" "$final_img"
mv "$tmp_tag" "$final_tag"
