// Run with: node --test apps/omlx-web/tests/active_models_ui.test.cjs
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.join(__dirname, '..', 'omlx_web');
const source = fs.readFileSync(path.join(root, 'static/js/active_models.js'), 'utf8');
const en = JSON.parse(fs.readFileSync(path.join(root, 'i18n/en.json'), 'utf8'));
const zh = JSON.parse(fs.readFileSync(path.join(root, 'i18n/zh.json'), 'utf8'));

function load() {
    const context = vm.createContext({});
    vm.runInContext(source, context);
    return context.ActiveModels;
}
const lib = load();
const tEn = key => (key in en ? en[key] : key);
const tZh = key => (key in zh ? zh[key] : key);
const parts = id => lib.describe(id).parts.filter(p => p.key).map(p => p.token + ':' + p.key).join(' ');

test('short names keep what people say and hide format suffixes', () => {
    const cases = [
        ['BAAI-bge-reranker-v2-m3-mlx-fp16', 'BAAI', 'bge-reranker-v2-m3'],
        ['Qwen3-ASR-1.7B-bf16', 'Qwen', 'Qwen3-ASR 1.7B'],
        ['Qwen3.6-35B-A3B-oQ4e-mtp', 'Qwen', 'Qwen3.6 35B-A3B'],
        ['bge-m3-mlx-8bit', 'BAAI', 'bge-m3'],
        ['lettucedect-v2-qwen-2b', 'KRLabs', 'lettucedect-v2 2B'],
        ['jina-reranker-v3.5-mlx', 'Jina AI', 'jina-reranker-v3.5'],
        ['clef-flash-oQ4e', 'Cloudflare', 'clef-flash'],
        ['Qwen3.8-27B-oQ4e-mtp', 'Qwen', 'Qwen3.8 27B'],
        ['Kokoro-82M-bf16', 'hexgrad', 'Kokoro 82M'],
        ['mxbai-rerank-large-v2', 'Mixedbread', 'mxbai-rerank-large-v2'],
        ['Qwen3-Embedding-0.6B-4bit-DWQ', 'Qwen', 'Qwen3-Embedding 0.6B'],
        ['Qwen3-Reranker-0.6B-4bit', 'Qwen', 'Qwen3-Reranker 0.6B'],
        ['Qwen3.5-0.8B-decision', 'Qwen', 'Qwen3.5 0.8B decision'],
        ['whisper-large-v3-asr-fp16', 'OpenAI', 'whisper-large-v3-asr'],
    ];
    for (const [id, vendor, shortName] of cases) {
        const info = lib.describe(id);
        assert.equal(info.vendor && info.vendor.name, vendor, id);
        assert.equal(info.shortName, shortName, id);
    }
});

test('every name part people ask about gets an explanation key', () => {
    assert.equal(parts('Qwen3.6-35B-A3B-oQ4e-mtp'),
        'Qwen3.6:family_version 35B:params_b A3B:moe oQ4e:oqe mtp:mtp');
    assert.equal(parts('BAAI-bge-reranker-v2-m3-mlx-fp16'),
        'BAAI:vendor_baai bge:bge reranker:reranker v2:version m3:m3 mlx:mlx fp16:fp16');
    assert.equal(parts('Qwen3-Embedding-0.6B-4bit-DWQ'),
        'Qwen3:family_version Embedding:embedding 0.6B:params_b 4bit:bits DWQ:dwq');
    assert.equal(parts('lettucedect-v2-qwen-2b'), 'lettucedect:lettucedect v2:version qwen:based_on 2B:params_b');
    // unknown parts stay in the name without an explanation
    assert.equal(parts('clef-flash-oQ4e'), 'clef:clef oQ4e:oqe');
});

test('explanations fill numbers in both English and Chinese', () => {
    const info = lib.describe('Qwen3.6-35B-A3B-oQ4e-mtp');
    const byToken = Object.fromEntries(info.parts.map(p => [p.token, p]));
    assert.match(lib.explain(byToken['35B'], tZh), /350 亿/);
    assert.match(lib.explain(byToken['35B'], tEn), /35 billion/);
    assert.match(lib.explain(byToken.A3B, tZh), /350 亿.*30 亿.*3B/);
    assert.match(lib.explain(byToken.oQ4e, tZh), /1\/3\.4/);
    assert.match(lib.explain(lib.describe('Kokoro-82M-bf16').parts[1], tZh), /8200 万/);
    for (const id of ['Qwen3.6-35B-A3B-oQ4e-mtp', 'BAAI-bge-reranker-v2-m3-mlx-fp16', 'Qwen3-Embedding-0.6B-4bit-DWQ',
        'lettucedect-v2-qwen-2b', 'whisper-large-v3-asr-fp16', 'Qwen3.5-0.8B-decision', 'mxbai-rerank-large-v2']) {
        for (const part of lib.describe(id).parts.filter(p => p.key)) {
            for (const t of [tEn, tZh]) {
                const text = lib.explain(part, t);
                assert.ok(!text.startsWith('status.active_models.'), `${id} ${part.token}: missing ${part.key}`);
                assert.ok(!/\{\w+\}/.test(text), `${id} ${part.token}: unfilled ${text}`);
            }
        }
    }
});

test('unknown vendors fall back to a letter avatar', () => {
    const info = lib.describe('mlx-community/gemma-3-27b-it-4bit');
    assert.equal(info.vendor, null);
    assert.equal(info.initial, 'G');
    assert.equal(info.shortName, 'gemma-3 27B it');
});

test('mono logos draw in the text color', () => {
    assert.equal(lib.describe('bge-m3-mlx-8bit').vendor.logo, '/admin/static/img/vendors/baai.svg');
    assert.match(lib.describe('bge-m3-mlx-8bit').vendor.maskStyle, /background-color: currentColor/);
    assert.equal(lib.describe('Qwen3-ASR-1.7B-bf16').vendor.maskStyle, '');
    for (const id of ['Qwen3-ASR-1.7B-bf16', 'bge-m3-mlx-8bit', 'jina-reranker-v3.5-mlx', 'whisper-large-v3-asr-fp16', 'clef-flash-oQ4e']) {
        const file = lib.describe(id).vendor.logo.replace('/admin/static/', '');
        assert.ok(fs.existsSync(path.join(root, 'static', file)), file);
    }
});

test('row status shows request count and summed speed, or idle time', () => {
    const fmt = s => s + 's';
    const generating = {active_requests: 2, generating: [{tokens_per_second: 100.4}, {tokens_per_second: 63.9}]};
    assert.deepEqual([lib.status(generating, tZh, fmt).tone, lib.status(generating, tZh, fmt).text],
        ['active', '2 个请求 · 164 tok/s']);
    const prefill = {active_requests: 1, prefilling: [{processed: 45, total: 100}], waiting_requests: 2};
    assert.equal(lib.status(prefill, tEn, fmt).text, '1 req · reading prompt 45% · 2 queued');
    assert.equal(lib.status(prefill, tEn, fmt).tone, 'prefill');
    assert.equal(lib.status({active_requests: 0, idle_seconds: 6}, tEn, fmt).text, 'idle 6s');
    const idleWithAvg = lib.status({active_requests: 0, idle_seconds: 6, avg_generation_tps: 121.6}, tZh, fmt);
    assert.deepEqual([idleWithAvg.text, idleWithAvg.avg], ['空闲 6s', '平均 122 tok/s']);
    assert.equal(lib.status({...prefill, avg_generation_tps: 98.2}, tEn, fmt).avg, 'avg 98 tok/s');
    assert.equal(lib.status({active_requests: 0, idle_seconds: 6}, tEn, fmt).avg, '');
    assert.equal(lib.status({active_requests: 0, waiting_requests: 1}, tEn, fmt).tone, 'queued');
});

test('source label shows the folder path under a model directory', () => {
    const models = [
        {id: 'Qwen3.6-35B-A3B-oQ4e-mtp', model_path: '/Volumes/WD/models/Jundot/Qwen3.6-35B-A3B-oQ4e-mtp'},
        {id: 'flat-model', model_path: '/Volumes/WD/models/flat-model'},
        {id: 'cached', model_path: '/hf/snapshots/abc', source_repo_id: 'org/cached'},
    ];
    assert.equal(lib.sourceLabel('Qwen3.6-35B-A3B-oQ4e-mtp', models, ['/Volumes/WD/models/']), 'Jundot/Qwen3.6-35B-A3B-oQ4e-mtp');
    assert.equal(lib.sourceLabel('flat-model', models, ['/Volumes/WD/models']), null);
    assert.equal(lib.sourceLabel('cached', models, []), 'org/cached');
    assert.equal(lib.sourceLabel('missing', models, ['/Volumes/WD/models']), null);
    assert.equal(lib.popoverId('a/b c.1'), 'am-detail-a-b-c-1');
});

test('fixed popover stays inside the window and flips up near the bottom', () => {
    const context = vm.createContext({window: {innerWidth: 1000, innerHeight: 800}});
    vm.runInContext(source, context);
    const {fixedPopoverStyle} = context.window.ActiveModels;
    const at = (left, top, bottom) => ({getBoundingClientRect: () => ({left, top, bottom})});
    assert.equal(fixedPopoverStyle(at(100, 100, 120)), 'left: 100px; width: 544px; top: 126px;');
    assert.equal(fixedPopoverStyle(at(900, 700, 720)), 'left: 440px; width: 544px; bottom: 106px;');
});

test('unknown ids fall back to the bare id with a letter avatar', () => {
    const info = lib.describe('markitdown');
    assert.equal(info.shortName, 'markitdown');
    assert.equal(info.vendor, null);
    assert.equal(info.initial, 'M');
    assert.equal(info.parts.filter(p => p.key).length, 0);
});
