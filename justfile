set shell := ["bash", "-euo", "pipefail", "-c"]

_plist_dst := env_var('HOME') / "Library/LaunchAgents/ai.omlx.server.plist"
_plist_src := "packaging/launchd/ai.omlx.server.plist.template"

# shared shell logging helpers, eval'd at the top of each recipe (see `eval "{{_log_lib}}"`)
_log_lib := '''
log_info()  { printf '\033[32m→\033[0m %s\n' "$*"; }
log_warn()  { printf '\033[33m⚠ %s\033[0m\n' "$*" >&2; }
log_error() { printf '\033[31m✗ %s\033[0m\n' "$*" >&2; }
log_done()  { printf '\033[32m✓\033[0m %s\n' "$*"; }
_BUILD_LOG=$(mktemp)
trap 'rm -f "$_BUILD_LOG"' EXIT
run_quiet() {
    local label="$1"; shift
    log_info "$label"
    if "$@" > "$_BUILD_LOG" 2>&1; then return 0
    else local rc=$?; log_error "$label failed. Log:"; cat "$_BUILD_LOG" >&2; exit $rc; fi
}
'''

_default:
    @just --list --unsorted --list-heading '' --list-prefix='- '

# sync fork with upstream, then rebase local changes on top
pull:
    #!/usr/bin/env bash
    set -euo pipefail
    {{_log_lib}}
    log_info "fetching upstream"
    git fetch upstream
    log_info "rebasing local changes onto upstream/main"
    git rebase --autostash upstream/main
    log_info "pushing to fork"
    git push --force-with-lease origin main
    log_done "pull complete"

# create venv and install all deps
setup:
    #!/usr/bin/env bash
    set -euo pipefail
    {{_log_lib}}
    log_info "syncing dependencies"
    uv sync --dev
    log_done "setup complete"

# editable install into venv
install:
    #!/usr/bin/env bash
    set -euo pipefail
    {{_log_lib}}
    run_quiet "installing omlx (editable)" uv pip install -e ".[dev]"
    log_done "install complete"

# remove venv and build artifacts
uninstall:
    #!/usr/bin/env bash
    set -euo pipefail
    {{_log_lib}}
    [[ -d .venv ]] && { log_info "removing .venv"; rm -rf .venv; }
    [[ -d build ]] && { log_info "removing build/"; rm -rf build; }
    [[ -d dist ]] && { log_info "removing dist/"; rm -rf dist; }
    find . -name '*.egg-info' -type d -exec rm -rf {} + 2>/dev/null || true
    log_done "uninstall complete"

# run server
run *args:
    uv run omlx serve {{args}}

# run tests (default: fast only)
test *args:
    uv run pytest {{args}}

# run slow tests (model-loading)
test-slow:
    uv run pytest -m slow

# run integration tests (requires running server)
test-integration:
    uv run pytest -m integration

# lint and type check
lint:
    #!/usr/bin/env bash
    set -euo pipefail
    {{_log_lib}}
    log_info "ruff"
    uv run ruff check omlx/
    log_info "black"
    uv run black --check omlx/
    log_info "mypy"
    uv run mypy omlx/
    log_done "all checks passed"

# run as a persistent launchd service: just service <install|uninstall|status|logs|restart>
service verb:
    @just _service-{{verb}}

# install LaunchAgent (start at login, restart on crash, binds 0.0.0.0, port from settings.json)
[private]
_service-install:
    #!/usr/bin/env bash
    set -euo pipefail
    {{_log_lib}}
    uv_bin="$(command -v uv)" || { log_error "uv not found on PATH"; exit 1; }
    mkdir -p "$HOME/.omlx/logs" "$(dirname "{{_plist_dst}}")"
    log_info "rendering plist for this machine"
    sed -e "s|__UV__|$uv_bin|g" \
        -e "s|__PROJECT__|$PWD|g" \
        -e "s|__HOME__|$HOME|g" \
        "{{_plist_src}}" > "{{_plist_dst}}"
    launchctl unload "{{_plist_dst}}" 2>/dev/null || true
    launchctl load "{{_plist_dst}}"
    log_done "service loaded — bound to 0.0.0.0, port from settings.json (logs: ~/.omlx/logs/)"

# stop and remove the LaunchAgent
[private]
_service-uninstall:
    #!/usr/bin/env bash
    set -euo pipefail
    {{_log_lib}}
    launchctl unload "{{_plist_dst}}" 2>/dev/null || true
    rm -f "{{_plist_dst}}"
    log_done "service removed"

# show service status
[private]
_service-status:
    @launchctl list | grep ai.omlx.server || echo "not loaded"

# tail service logs
[private]
_service-logs:
    @tail -f "$HOME/.omlx/logs/server.out.log" "$HOME/.omlx/logs/server.err.log"

# reload after code/config changes
[private]
_service-restart:
    #!/usr/bin/env bash
    set -euo pipefail
    {{_log_lib}}
    launchctl unload "{{_plist_dst}}" 2>/dev/null || true
    launchctl load "{{_plist_dst}}"
    log_done "service restarted"
