#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FILE_ID="1VDGWnEC1y8BGDh_N8U5pxeu2pJLp-BuQ"
ZIP_PATH="db.zip"

# Prefer the macOS system curl: it uses the system keychain, whereas
# anaconda's bundled curl can ship a stale CA bundle that fails on
# drive.google.com.
if [[ -x /usr/bin/curl ]]; then
    CURL=/usr/bin/curl
else
    CURL=curl
fi

download_with_curl() {
    local cookie_jar
    cookie_jar="$(mktemp)"
    trap 'rm -f "$cookie_jar"' RETURN

    # First request follows redirects and either returns the file directly
    # (small files) or lands on the virus-scan interstitial (large files).
    "$CURL" -L -sc "$cookie_jar" \
        "https://drive.google.com/uc?export=download&id=${FILE_ID}" \
        -o "$ZIP_PATH"

    # If the response is HTML, it's the interstitial — extract the confirm
    # token and retry.
    if file "$ZIP_PATH" | grep -qi 'html'; then
        local confirm
        confirm="$(grep -oE 'confirm=[0-9A-Za-z_-]+' "$ZIP_PATH" | head -n1 | cut -d= -f2 || true)"
        if [[ -z "$confirm" ]]; then
            confirm="$(awk '/download_warning/ {print $NF}' "$cookie_jar" | tail -n1)"
        fi
        if [[ -z "$confirm" ]]; then
            echo "Could not find Google Drive confirm token." >&2
            return 1
        fi
        "$CURL" -L -b "$cookie_jar" -o "$ZIP_PATH" \
            "https://drive.google.com/uc?export=download&confirm=${confirm}&id=${FILE_ID}"
    fi
}

echo "==> Downloading db.zip from Google Drive"
download_with_curl

if [[ ! -s "$ZIP_PATH" ]]; then
    echo "Download failed: $ZIP_PATH is empty or missing." >&2
    exit 1
fi

echo "==> Unzipping $ZIP_PATH into db/"
mkdir -p db
unzip -o -q "$ZIP_PATH" -d db -x '__MACOSX/*' '*/.DS_Store'

echo "==> Cleaning up"
rm -f "$ZIP_PATH"

echo "==> Done. Contents of db/:"
ls db
