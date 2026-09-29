import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { markerSplit, normalizeExtractedLines, analyzePages } from '../src/core.js';
import * as ort from 'onnxruntime-web';

const main = await fs.readFile(new URL('../src/main.js', import.meta.url), 'utf8');

const tensor = new ort.Tensor('int64', BigInt64Array.from([4, 17, 0], BigInt), [1, 3]);
assert.equal(tensor.type, 'int64');
assert.deepEqual(Array.from(tensor.data, Number), [4, 17, 0]);

const normalized = normalizeExtractedLines([
  'Intro text Q114. First question text Q115. Second question text',
  'Q1. What is India?'
]);
assert.deepEqual(normalized, [
  'Intro text',
  'Q114. First question text',
  'Q115. Second question text',
  'Q1. What is India?'
]);

const pages = Array.from({length:63}, (_, i) => ({page:i+1, lines:[]}));
for(let q=1;q<=115;q++){
  const page = pages[Math.min(62, Math.floor((q-1)*63/115))];
  page.lines.push(`Q${q}. Sample question ${q}`);
  page.lines.push('A. Option A');
  page.lines.push('B. Option B');
  page.lines.push('C. Option C');
  page.lines.push('D. Option D');
  page.lines.push('Answer: A');
  page.lines.push('Explanation: Sample');
}
const stats = analyzePages(pages);
assert.equal(stats.pages, 63);
assert.equal(stats.questions, 115);
assert.equal(stats.options, 460);
assert.equal(stats.nonempty, 63);
assert.equal(stats.answerMarkers, 230);

assert.match(main, /BigInt64Array\.from\(a,BigInt\)/);
assert.match(main, /createSession\(modelBlob,dataBlob,dataName\)/);
assert.match(main, /externalData:\[\{path:dataName,data:dataBlob\}\]/);
assert.match(main, /modelAsset\(F\.encData/);
assert.match(main, /modelAsset\(F\.decData/);
assert.doesNotMatch(main, /createSession\(MODEL\+F\.enc/);
assert.doesNotMatch(main, /createSession\(MODEL\+F\.dec/);

const modelFiles = [
  'encoder_model.onnx',
  'encoder_model.onnx.data',
  'decoder_model.onnx',
  'decoder_shared.onnx.data',
  'tokenizer_src.json',
  'tokenizer_tgt.json',
  'tokenizer_meta.json',
  'generation_config.json'
];
const base='https://huggingface.co/hari31416/indictrans2-en-indic-dist-200M-ONNX-int8/resolve/main/';
for(const name of modelFiles){
  const r=await fetch(base+name,{method:'HEAD'});
  assert.equal(r.ok,true,`Model asset unavailable: ${name} (HTTP ${r.status})`);
}
console.log('TeluguSetu mobile preflight: PASS');
console.log('BigInt int64 tensor: PASS');
console.log('Question/option parser: PASS (115 questions / 460 options)');
console.log('Model asset availability: PASS (8 files reachable)');
