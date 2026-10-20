// Active Models card: vendor logo, spoken short name, plain meaning of each id part, one-line status.
(function (root) {
    'use strict';

    const LOGO_DIR = '/admin/static/img/vendors/';
    const TERM = 'status.active_models.term.';

    // matched against the first part of the id; logos are listed in vendor_deps.py
    const VENDORS = [
        { pattern: /^qwen/i, name: 'Qwen', logo: 'qwen-color.svg', about: 'vendor_qwen' },
        { pattern: /^(baai|bge)$/i, name: 'BAAI', logo: 'baai.svg', mono: true, about: 'vendor_baai' },
        { pattern: /^jina$/i, name: 'Jina AI', logo: 'jina.svg', mono: true },
        { pattern: /^mxbai$/i, name: 'Mixedbread' },
        { pattern: /^whisper$/i, name: 'OpenAI', logo: 'openai.svg', mono: true },
        { pattern: /^kokoro$/i, name: 'hexgrad' },
        { pattern: /^lettucedect$/i, name: 'KRLabs' },
        { pattern: /^clef$/i, name: 'Cloudflare', logo: 'cloudflare-color.svg' },
    ];
    const FAMILIES = new Set(['bge', 'whisper', 'kokoro', 'lettucedect', 'clef']);
    const SERIES = new Set(['jina', 'mxbai']);
    // a bare family name after the first part names the base model, e.g. lettucedect-v2-qwen-2b
    const BASE_FAMILIES = { qwen: 'Qwen', llama: 'Llama', gemma: 'Gemma', mistral: 'Mistral', phi: 'Phi' };
    // midpoints of the bits-per-weight ranges in docs/oQ_Quantization.md
    const OQ_BITS_PER_WEIGHT = { 2: 2.9, 2.5: 3.2, 2.7: 3.3, 3: 3.6, 3.5: 3.9, 4: 4.65, 5: 5.6, 6: 6.6, 8: 8.5 };
    const STATUS_CLASSES = {
        prefill: { dot: 'bg-blue-500', text: 'text-blue-700 font-medium' },
        active: { dot: 'bg-green-500', text: 'text-green-700 font-medium' },
        queued: { dot: 'bg-amber-300', text: 'text-amber-700 font-medium' },
        idle: { dot: 'bg-neutral-300', text: 'text-neutral-400' },
    };
    const cache = new Map();

    function trim(value, digits = 2) {
        return String(Number(value.toFixed(digits)));
    }

    function paramsPart(token, amount, unit) {
        const billions = unit === 'b' ? amount : amount / 1000;
        const label = token.slice(0, -1) + unit.toUpperCase();
        if (billions >= 0.1) {
            return { token: label, kind: 'params', key: 'params_b', billions,
                vars: { billions: trim(billions), yi: trim(billions * 10) } };
        }
        return { token: label, kind: 'params', key: 'params_m', billions,
            vars: { millions: trim(billions * 1000), wan: trim(billions * 1e5) } };
    }

    function classify(token, index, ctx) {
        const lower = token.toLowerCase();
        let m;
        if (index === 0 && ctx.vendor && ctx.vendor.about && lower === ctx.vendor.name.toLowerCase() && ctx.count > 1) {
            return { kind: 'vendor', key: ctx.vendor.about };
        }
        if ((m = /^(\d+(?:\.\d+)?)([bm])$/i.exec(token))) {
            return paramsPart(token, Number(m[1]), m[2].toLowerCase());
        }
        if (ctx.params && ctx.params.key === 'params_b' && (m = /^a(\d+(?:\.\d+)?)b$/i.exec(token))) {
            const active = Number(m[1]);
            return { kind: 'moe', key: 'moe', vars: {
                total: ctx.params.token, active: trim(active) + 'B',
                total_yi: ctx.params.vars.yi, active_yi: trim(active * 10),
            } };
        }
        if ((m = /^oq(\d+(?:\.\d+)?)(e?)$/i.exec(token))) {
            const bpw = OQ_BITS_PER_WEIGHT[Number(m[1])] || Number(m[1]) + 0.6;
            return { hidden: true, key: m[2] ? 'oqe' : 'oq', vars: { bits: m[1], ratio: trim(16 / bpw, 1) } };
        }
        if ((m = /^(\d+)bit$/i.exec(token))) {
            // mlx affine quantization stores a scale and bias per 64 weights: about +0.5 bit
            return { hidden: true, key: 'bits', vars: { bits: m[1], ratio: trim(16 / (Number(m[1]) + 0.5), 1) } };
        }
        if (lower === 'dwq') return { hidden: true, key: 'dwq' };
        if (lower === 'bf16' || lower === 'fp16') return { hidden: true, key: 'fp16' };
        if (lower === 'mlx') return { hidden: true, key: 'mlx' };
        if (lower === 'mtp') return { hidden: true, key: 'mtp' };
        if ((m = /^v(\d+(?:\.\d+)*)$/i.exec(token))) return { key: 'version', vars: { version: m[1] } };
        if (lower === 'asr' || lower === 'tts' || lower === 'embedding' || lower === 'decision') return { key: lower };
        if (lower === 'reranker' || lower === 'rerank') return { key: 'reranker' };
        if (lower === 'm3' && ctx.family === 'bge') return { key: 'm3' };
        if (/^(large|base|small|medium|tiny|mini)$/.test(lower)) return { key: 'size_tier' };
        if (index > 0 && BASE_FAMILIES[lower]) return { hidden: true, key: 'based_on', vars: { family: BASE_FAMILIES[lower] } };
        if (index === ctx.nameIndex) {
            if ((m = /^([a-z]+)(\d+(?:\.\d+)*)$/i.exec(token))) {
                return { key: 'family_version', vars: { family: m[1], version: m[2] } };
            }
            if (FAMILIES.has(lower)) return { key: lower };
            if (SERIES.has(lower) && ctx.vendor) return { key: 'series', vars: { vendor: ctx.vendor.name } };
        }
        return { key: null };
    }

    function vendorFor(first) {
        const rule = VENDORS.find(v => v.pattern.test(first));
        if (!rule) return null;
        const logo = rule.logo ? LOGO_DIR + rule.logo : null;
        return {
            name: rule.name,
            about: rule.about || null,
            logo,
            mono: !!rule.mono,
            // mono logos draw in the text color so they follow light and dark themes
            maskStyle: logo && rule.mono
                ? `-webkit-mask: url("${logo}") center / contain no-repeat; mask: url("${logo}") center / contain no-repeat; background-color: currentColor;`
                : '',
        };
    }

    function describe(id) {
        const key = String(id || '');
        if (cache.has(key)) return cache.get(key);
        const base = key.split('/').pop();
        const tokens = base.split('-').filter(Boolean);
        const vendor = tokens.length ? vendorFor(tokens[0]) : null;
        const ctx = { vendor, count: tokens.length, params: null, nameIndex: 0, family: null };
        const parts = [];
        tokens.forEach((token, index) => {
            const part = Object.assign({ token }, classify(token, index, ctx));
            if (part.kind === 'vendor') ctx.nameIndex = index + 1;
            if (index === ctx.nameIndex) ctx.family = token.toLowerCase();
            if (part.kind === 'params') ctx.params = part;
            parts.push(part);
        });

        const head = [];
        const tail = [];
        let size = '';
        for (const part of parts) {
            if (part.hidden || part.kind === 'vendor') continue;
            if (part.kind === 'params' && !size) size = part.token;
            else if (part.kind === 'moe' && size) size += '-' + part.token;
            else if (size) tail.push(part.token);
            else head.push(part.token);
        }
        const shortName = [head.join('-'), size, tail.join(' ')].filter(Boolean).join(' ') || base || key;
        const info = {
            id: key,
            vendor,
            initial: ((vendor && vendor.name) || base || '?').charAt(0).toUpperCase(),
            shortName,
            parts: parts.map(p => ({ token: p.token, key: p.key, vars: p.vars || {} })),
        };
        cache.set(key, info);
        return info;
    }

    function fill(text, vars) {
        return Object.entries(vars || {}).reduce((out, [name, value]) => out.split('{' + name + '}').join(String(value)), text);
    }

    function explain(part, t) {
        return part && part.key ? fill(t(TERM + part.key), part.vars) : '';
    }

    function vendorAbout(vendor, t) {
        return vendor && vendor.about ? t(TERM + vendor.about) : '';
    }

    // the id people see in the model folder, e.g. Jundot/Qwen3.6-35B-A3B-oQ4e-mtp; null when it adds nothing
    function sourceLabel(id, models, modelDirs) {
        const entry = (models || []).find(item => item && item.id === id);
        if (!entry) return null;
        if (entry.source_repo_id && entry.source_repo_id !== id) return entry.source_repo_id;
        const path = String(entry.model_path || '');
        for (const dir of modelDirs || []) {
            const root = String(dir || '').trim().replace(/\/+$/, '');
            if (!root || !path.startsWith(root + '/')) continue;
            const rel = path.slice(root.length + 1);
            return rel && rel !== id ? rel : null;
        }
        return null;
    }

    function status(model, t, formatDuration) {
        const active = model.active_requests || 0;
        const queued = model.waiting_requests || 0;
        const prefill = (model.prefilling || [])[0];
        const pieces = [];
        let tone = 'idle';
        if (active > 0) {
            tone = 'active';
            pieces.push(t('status.active_models.requests_count').replace('{count}', active));
        }
        if (prefill) {
            tone = 'prefill';
            const percent = prefill.total > 0 ? Math.min(100, Math.round(prefill.processed / prefill.total * 100)) : 0;
            pieces.push(t('status.active_models.prefill_progress').replace('{percent}', percent));
        } else {
            const speed = (model.generating || []).reduce((sum, gen) => sum + (gen.tokens_per_second || 0), 0);
            if (speed > 0) pieces.push(Math.round(speed) + ' tok/s');
        }
        if (queued > 0) {
            if (tone === 'idle') tone = 'queued';
            pieces.push(t('status.active_models.queued_count').replace('{count}', queued));
        }
        if (tone === 'idle') {
            pieces.push(t('status.active_models.idle') + (model.idle_seconds != null ? ' ' + formatDuration(model.idle_seconds) : ''));
        }
        const avg = model.avg_generation_tps > 0
            ? t('status.active_models.avg_speed').replace('{speed}', Math.round(model.avg_generation_tps))
            : '';
        return { tone, text: pieces.join(' · '), avg, dot: STATUS_CLASSES[tone].dot, textClass: STATUS_CLASSES[tone].text };
    }

    function popoverId(id) {
        return 'am-detail-' + String(id || '').replace(/[^A-Za-z0-9_-]/g, '-');
    }

    // open the popover upward when the row sits too close to the bottom of the window
    function opensUp(el) {
        const rect = el.getBoundingClientRect();
        const below = window.innerHeight - rect.bottom;
        return below < 320 && rect.top > below;
    }

    // the usage table's own `models` (plain ids) shadows the dashboard's model list inside its Alpine scope
    function dashboardModels() {
        const el = document.querySelector('[x-data="dashboard()"]');
        return (el && root.Alpine && root.Alpine.$data(el).models) || [];
    }

    // fixed coordinates so a popover escapes overflow-clipping containers such as the usage table
    function fixedPopoverStyle(el) {
        const rect = el.getBoundingClientRect();
        const width = Math.min(544, window.innerWidth - 32);
        const left = Math.max(16, Math.min(rect.left, window.innerWidth - width - 16));
        const vertical = opensUp(el)
            ? 'bottom: ' + (window.innerHeight - rect.top + 6) + 'px;'
            : 'top: ' + (rect.bottom + 6) + 'px;';
        return 'left: ' + left + 'px; width: ' + width + 'px; ' + vertical;
    }

    root.ActiveModels = { describe, explain, vendorAbout, sourceLabel, status, popoverId, opensUp, dashboardModels, fixedPopoverStyle };
})(typeof window !== 'undefined' ? window : globalThis);
