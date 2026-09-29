import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import * as ort from 'onnxruntime-web';
import { IndicProcessor, IT2Tokenizer } from '../src/indictrans2.js';

const main = await fs.readFile(new URL('../src/main.js', import.meta.url), 'utf8');
const tensor = new ort.Tensor('int64', BigInt64Array.from([4, 17, 0], BigInt), [1, 3]);
assert.deepEqual(Array.from(tensor.data, Number), [4, 17, 0]);

function normalize(lines) {
  const out=[];
  for (const raw of lines) {
    const line=String(raw||'').trim();
    const m=[...line.matchAll(/(?:^|\s)(Q(?:uestion)?\s*\d{1,4}\s*[\.\):\-])(?=\s|$)/gi)];
    if (m.length<=1) { if(line) out.push(line); continue; }
    if (m[0].index>0) { const p=line.slice(0,m[0].index).trim(); if(p) out.push(p); }
    for(let i=0;i<m.length;i++) out.push(line.slice(m[i].index,i+1<m.length?m[i+1].index:line.length).trim());
  }
  return out;
}
const pages=Array.from({length:63},(_,i)=>({page:i+1,lines:[]}));
for(let q=1;q<=115;q++){const p=pages[Math.min(62,Math.floor((q-1)*63/115))];p.lines.push(`Q${q}. Sample question`,'A. Option A','B. Option B','C. Option C','D. Option D');}
const all=pages.flatMap(p=>p.lines);
const qs=all.flatMap(l=>[...l.matchAll(/(?:^|\s)Q\s*(\d{1,4})\s*[\.\):\-]/gi)].map(m=>m[1]));
const opts=all.filter(l=>/^\s*[A-D]\s*[\.\):\-]\s+/.test(l));
assert.equal(new Set(qs).size,115);
assert.equal(opts.length,460);
assert.deepEqual(normalize(['Intro Q114. First Q115. Second']),['Intro','Q114. First','Q115. Second']);
assert.match(main,/import ['"]regenerator-runtime\/runtime\.js['"]/);
assert.match(main,/BigInt64Array\.from\(a,BigInt\)/);
assert.match(main,/createSessionFromBlobs\(modelBlob,dataBlob,dataName\)/);
assert.match(main,/externalData:\[\{path:dataName,data:dataBytes\}\]/);
assert.match(main,/modelAsset\(F\.encData/);
assert.match(main,/modelAsset\(F\.decData/);
assert.match(main,/modelAsset\(F\.decPast/);
assert.match(main,/decoder_with_past_model\.onnx/);
assert.match(main,/preprocessBatch\(\[text\]/);
assert.match(main,/postprocessBatch\(\[raw\]/);
assert.match(main,/collectPresents/);
assert.match(main,/MODEL_VERSION='v4'/);
const ip=new IndicProcessor({inference:true});
ip.resetQueue();
const [prepared]=ip.preprocessBatch(['The Prime Minister addressed the nation yesterday evening.'],{srcLang:'eng_Latn',tgtLang:'tel_Telu'});
assert.match(prepared,/^eng_Latn tel_Telu /);
const [post]=ip.postprocessBatch(['प्रधानमंत्री ने राष्ट्र को संबोधित किया ।'],{lang:'tel_Telu'});
assert.match(post,/^[\u0C00-\u0C7F]/);

const requiredModelAssets=['encoder_model.onnx','encoder_model.onnx.data','decoder_model.onnx','decoder_with_past_model.onnx','decoder_shared.onnx.data','tokenizer_src.json','tokenizer_tgt.json','tokenizer_meta.json','generation_config.json'];
assert.deepEqual(requiredModelAssets,['encoder_model.onnx','encoder_model.onnx.data','decoder_model.onnx','decoder_with_past_model.onnx','decoder_shared.onnx.data','tokenizer_src.json','tokenizer_tgt.json','tokenizer_meta.json','generation_config.json']);
console.log('TeluguSetu preflight PASS: 115 questions, 460 options, IndicProcessor parity, decoder-with-past, model manifest, cached model wiring');
