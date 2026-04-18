#!/bin/bash
# Copyright (c) 2020-2022, Xilinx, Inc.
# Copyright (C) 2022-2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Fail-fast: a partially-populated deps/ tree is worse than no deps/ tree at
# all. The container entrypoint (docker/finn_entrypoint.sh) does
# `pip install -e ${FINN_ROOT}/deps/qonnx` etc., and pytest later imports
# qonnx/brevitas at collection time. If a `git clone` here silently fails
# (transient github.com hiccup, DNS blip, rate-limit) and we exit 0 anyway,
# the failure surfaces hours later as "ModuleNotFoundError: qonnx" with 100+
# pytest collection errors that look unrelated to the original problem.
set -euo pipefail

QONNX_COMMIT="f5c9819bd00f01f41e70639b8461c8e4b39432f7"
FINN_EXP_COMMIT="0724be21111a21f0d81a072fccc1c446e053f851"
BREVITAS_COMMIT="aad4d5a293db6f2ec622a92a5d3278e47072453e"
CNPY_COMMIT="8c82362372ce600bbd1cf11d64661ab69d38d7de"
HLSLIB_COMMIT="a2cd3e6ce653a03e59af6bcb9fbeaa71618d160e"
OMX_COMMIT="a5d48f93309b235fdd21556d16e86e6ef5db6e2e"
AVNET_BDF_COMMIT="2d49cfc25766f07792c0b314489f21fe916b639b"
XIL_BDF_COMMIT="8cf4bb674a919ac34e3d99d8d71a9e60af93d14e"
RFSOC4x2_BDF_COMMIT="13fb6f6c02c7dfd7e4b336b18b959ad5115db696"
KV260_BDF_COMMIT="98e0d3efc901f0b974006bc4370c2a7ad8856c79"
EXP_BOARD_FILES_MD5="226ca927a16ea4ce579f1332675e9e9a"
AUPZU3_BDF_COMMIT="b595ecdf37c7204129517de1773b0895bcdcc2ed"

QONNX_URL="https://github.com/fastmachinelearning/qonnx.git"
FINN_EXP_URL="https://github.com/Xilinx/finn-experimental.git"
BREVITAS_URL="https://github.com/Xilinx/brevitas.git"
CNPY_URL="https://github.com/maltanar/cnpy.git"
HLSLIB_URL="https://github.com/Xilinx/finn-hlslib.git"
OMX_URL="https://github.com/maltanar/oh-my-xilinx.git"
AVNET_BDF_URL="https://github.com/Avnet/bdf.git"
XIL_BDF_URL="https://github.com/Xilinx/XilinxBoardStore.git"
RFSOC4x2_BDF_URL="https://github.com/RealDigitalOrg/RFSoC4x2-BSP.git"
KV260_BDF_URL="https://github.com/Xilinx/XilinxBoardStore.git"
AUPZU3_BDF_URL="https://github.com/RealDigitalOrg/aup-zu3-bsp.git"

QONNX_DIR="qonnx"
FINN_EXP_DIR="finn-experimental"
BREVITAS_DIR="brevitas"
CNPY_DIR="cnpy"
HLSLIB_DIR="finn-hlslib"
OMX_DIR="oh-my-xilinx"
AVNET_BDF_DIR="avnet-bdf"
XIL_BDF_DIR="xil-bdf"
RFSOC4x2_BDF_DIR="rfsoc4x2-bdf"
KV260_SOM_BDF_DIR="kv260-som-bdf"
AUPZU3_BDF_DIR="aupzu3-8gb-bdf"

# absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")

# Run a command up to N times, with exponential back-off, before giving up.
# Use for any command that may fail transiently due to network conditions
# (github.com 5xx, DNS blips, rate-limit). Always returns the exit status of
# the last attempt, so callers under `set -e` will still abort on permanent
# failure.
retry() {
    local n=0
    local max=5
    local delay=4
    until "$@"; do
        n=$((n+1))
        if (( n >= max )); then
            echo "fetch-repos: command failed after $n attempts: $*" >&2
            return 1
        fi
        echo "fetch-repos: attempt $n/$max failed for: $* (retrying in ${delay}s)" >&2
        sleep "$delay"
        delay=$((delay*2))
    done
}

fetch_repo() {
    # URL for git repo to be cloned
    local REPO_URL=$1
    # commit hash for repo
    local REPO_COMMIT=$2
    # directory to clone to under deps/
    local REPO_DIR=$3
    # absolute path for the repo local copy
    local CLONE_TO=$SCRIPTPATH/deps/$REPO_DIR

    # If a previous run left a partial clone (e.g. interrupted git clone),
    # the directory exists but is not a valid git repo. Detect and discard
    # so the retry below gets a clean slate instead of silently exiting.
    if [ -d "$CLONE_TO" ] && ! git -C "$CLONE_TO" rev-parse --git-dir >/dev/null 2>&1; then
        echo "fetch-repos: discarding partial clone at $CLONE_TO" >&2
        rm -rf "$CLONE_TO"
    fi

    if [ ! -d "$CLONE_TO" ]; then
        retry git clone "$REPO_URL" "$CLONE_TO"
    fi

    local CURRENT_COMMIT
    CURRENT_COMMIT=$(git -C "$CLONE_TO" rev-parse HEAD)
    if [ "$CURRENT_COMMIT" != "$REPO_COMMIT" ]; then
        # `fetch` rather than `pull` because the working copy is typically a
        # detached HEAD on $REPO_COMMIT (see explicit checkout below); pull
        # requires a tracking branch and silently no-ops in detached HEAD.
        retry git -C "$CLONE_TO" fetch --tags --force
        git -C "$CLONE_TO" checkout "$REPO_COMMIT"
    fi

    CURRENT_COMMIT=$(git -C "$CLONE_TO" rev-parse HEAD)
    if [ "$CURRENT_COMMIT" != "$REPO_COMMIT" ]; then
        echo "fetch-repos: ERROR — $REPO_DIR is at $CURRENT_COMMIT, expected $REPO_COMMIT" >&2
        return 1
    fi
    echo "Successfully checked out $REPO_DIR at commit $CURRENT_COMMIT"
}

fetch_board_files() {
    echo "Downloading and extracting board files..."
    mkdir -p "$SCRIPTPATH/deps/board_files"
    OLD_PWD=$(pwd)
    cd "$SCRIPTPATH/deps/board_files"
    retry wget -q https://github.com/cathalmccabe/pynq-z1_board_files/raw/master/pynq-z1.zip
    retry wget -q https://dpoauwgwqsy2x.cloudfront.net/Download/pynq-z2.zip
    unzip -q pynq-z1.zip
    unzip -q pynq-z2.zip
    cp -r $SCRIPTPATH/deps/$AVNET_BDF_DIR/* $SCRIPTPATH/deps/board_files/
    cp -r $SCRIPTPATH/deps/$XIL_BDF_DIR/boards/Xilinx/rfsoc2x2 $SCRIPTPATH/deps/board_files/;
    cp -r $SCRIPTPATH/deps/$RFSOC4x2_BDF_DIR/board_files/rfsoc4x2 $SCRIPTPATH/deps/board_files/;
    cp -r $SCRIPTPATH/deps/$KV260_SOM_BDF_DIR/boards/Xilinx/kv260_som $SCRIPTPATH/deps/board_files/;
    cp -r $SCRIPTPATH/deps/$AUPZU3_BDF_DIR/board-files/aup-zu3-8gb $SCRIPTPATH/deps/board_files/;
    cd $OLD_PWD
}

fetch_repo $QONNX_URL $QONNX_COMMIT $QONNX_DIR
fetch_repo $FINN_EXP_URL $FINN_EXP_COMMIT $FINN_EXP_DIR
fetch_repo $BREVITAS_URL $BREVITAS_COMMIT $BREVITAS_DIR
fetch_repo $CNPY_URL $CNPY_COMMIT $CNPY_DIR
fetch_repo $HLSLIB_URL $HLSLIB_COMMIT $HLSLIB_DIR
fetch_repo $OMX_URL $OMX_COMMIT $OMX_DIR
fetch_repo $AVNET_BDF_URL $AVNET_BDF_COMMIT $AVNET_BDF_DIR
fetch_repo $XIL_BDF_URL $XIL_BDF_COMMIT $XIL_BDF_DIR
fetch_repo $RFSOC4x2_BDF_URL $RFSOC4x2_BDF_COMMIT $RFSOC4x2_BDF_DIR
fetch_repo $KV260_BDF_URL $KV260_BDF_COMMIT $KV260_SOM_BDF_DIR
fetch_repo $AUPZU3_BDF_URL $AUPZU3_BDF_COMMIT $AUPZU3_BDF_DIR

# Can skip downloading of board files entirely if desired
if [ "$FINN_SKIP_BOARD_FILES" = "1" ]; then
    echo "Skipping download and verification of board files"
else
    # download extra board files and extract if needed
    if [ ! -d "$SCRIPTPATH/deps/board_files" ]; then
        fetch_board_files
    else
        cd $SCRIPTPATH
        BOARD_FILES_MD5=$(find deps/board_files/ -type f -exec md5sum {} \; | sort -k 2 | md5sum | cut -d' ' -f 1)
        if [ "$BOARD_FILES_MD5" = "$EXP_BOARD_FILES_MD5" ]; then
            echo "Verified board files folder content md5: $BOARD_FILES_MD5"
        else
            echo "Board files folder md5: expected $BOARD_FILES_MD5 found $EXP_BOARD_FILES_MD5"
            echo "Board files folder content mismatch, removing and re-downloading"
            rm -rf deps/board_files/
            fetch_board_files
        fi
    fi
fi
