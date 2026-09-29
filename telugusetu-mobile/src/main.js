import 'regenerator-runtime/runtime.js';
import './style.css';
import * as ort from 'onnxruntime-web';
import { IndicProcessor, IT2Tokenizer } from './indictrans2.js';
import * as pdfjsLib from 'pdfjs-dist/legacy/build/pdf.mjs';
import pdfWorkerUrl from 'pdfjs-dist/legacy/build/pdf.worker.min.mjs?url';
pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;
import { PDFDocument, rgb } from 'pdf-lib';
import fontkit from '@pdf-lib/fontkit';
import { Document, Packer, Paragraph, TextRun } from 'docx';
import { Filesystem, Directory } from '@capacitor/filesystem';
import { Share } from '@capacitor/share';

const MODEL = 'https://huggingface.co/hari31416/indictrans2-en-indic-dist-200M-ONNX-int8/resolve/main/';
const F = {
  enc:'encoder_model.onnx', encData:'encoder_model.onnx.data',
  dec:'decoder_model.onnx', decData:'decoder_shared.onnx.data', decPast:'decoder_with_past_model.onnx',
  srcTok:'tokenizer_src.json', tgtTok:'tokenizer_tgt.json',
  meta:'tokenizer_meta.json', gen:'generation_config.json'
};
const DB_NAME='telugusetu-local-v1', DB_VERSION=1, STORE='jobs';
const MODEL_DB='telugusetu-model-v1', MODEL_STORE='assets', MODEL_VERSION='v4';
ort.env.wasm.numThreads=1;
ort.env.wasm.proxy=false;
ort.env.logLevel='error';

const app=document.querySelector('#app');
app.innerHTML=
'<main><section class="card">'+
'<h1>🇮🇳 TeluguSetu</h1><p class="subtitle">Offline English PDF → Telugu PDF / DOCX</p>'+
'<label class="drop" id="drop"><span id="fileLabel">Tap to select PDF</span><input id="file" type="file" accept=".pdf,application/pdf" hidden></label>'+
'<button id="analyze" class="primary">Analyze PDF</button>'+
'<div id="analysis" class="hidden"></div>'+
'<div id="actions" class="hidden"><button id="translate" class="primary">Translate to Telugu</button><button id="resume" class="secondary hidden">Resume Translation</button><button id="partial" class="secondary hidden">Download Partial PDF</button><button id="partialDocx" class="secondary hidden">Download Partial DOCX</button></div>'+
'<div id="progressWrap" class="hidden"><div class="progress"><div id="bar"></div></div><div id="progressText">0%</div><div id="detail"></div></div>'+
'<div id="downloads"></div><div id="message" class="small"></div></section></main>';

const $=id=>document.getElementById(id);
let selectedFile=null,currentJob=null,running=false,cancelled=false,model=null;

function msg(s){$('message').textContent=s||'';}
function openDB(){return new Promise((resolve,reject)=>{const r=indexedDB.open(DB_NAME,DB_VERSION);r.onupgradeneeded=()=>r.result.createObjectStore(STORE,{keyPath:'id'});r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);});}
function openModelDB(){return new Promise((resolve,reject)=>{const r=indexedDB.open(MODEL_DB,1);r.onupgradeneeded=()=>r.result.createObjectStore(MODEL_STORE);r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);});}
async function getModelAsset(key){const db=await openModelDB();return new Promise((res,rej)=>{const t=db.transaction(MODEL_STORE,'readonly'),r=t.objectStore(MODEL_STORE).get(key);r.onsuccess=()=>res(r.result||null);r.onerror=()=>rej(r.error);});}
async function putModelAsset(key,blob){const db=await openModelDB();return new Promise((res,rej)=>{const t=db.transaction(MODEL_STORE,'readwrite');t.objectStore(MODEL_STORE).put(blob,key);t.oncomplete=res;t.onerror=()=>rej(t.error);});}
async function cleanupOldModelAssets(){
  const db=await openModelDB();
  return new Promise((res,rej)=>{
    const t=db.transaction(MODEL_STORE,'readwrite'),store=t.objectStore(MODEL_STORE),r=store.openCursor();
    r.onsuccess=()=>{const cur=r.result;if(!cur){return;}if(typeof cur.key==='string'&&!cur.key.startsWith(MODEL_VERSION+':'))cur.delete();cur.continue();};
    t.oncomplete=res;t.onerror=()=>rej(t.error);
  });
}
function modelProgress(p,text){$('bar').style.width=p+'%';$('progressText').textContent=p+'%';$('detail').textContent=text;}
async function modelAsset(fileName,label,start,end){const key=MODEL_VERSION+':'+fileName;const cached=await getModelAsset(key).catch(()=>null);if(cached){modelProgress(end,label+' already saved on this phone.');return cached;}modelProgress(start,label+' downloading…');const response=await fetch(MODEL+fileName);if(!response.ok)throw new Error('Model download failed for '+fileName+' (HTTP '+response.status+').');const blob=await response.blob();try{await putModelAsset(key,blob);}catch(e){console.warn('Model cache write failed:',e);}modelProgress(end,label+' downloaded.');return blob;}
async function saveJob(job){const db=await openDB();return new Promise((res,rej)=>{const t=db.transaction(STORE,'readwrite');t.objectStore(STORE).put(job);t.oncomplete=res;t.onerror=()=>rej(t.error);});}
async function loadLatestJob(){const db=await openDB();return new Promise((resolve,reject)=>{const t=db.transaction(STORE,'readonly'),r=t.objectStore(STORE).getAll();r.onsuccess=()=>resolve(r.result.sort((a,b)=>b.updatedAt-a.updatedAt)[0]||null);r.onerror=()=>reject(r.error);});}

$('drop').onclick=()=>$('file').click();
$('file').onchange=()=>{selectedFile=$('file').files[0]||null;$('fileLabel').textContent=selectedFile?selectedFile.name:'Tap to select PDF';};

function markerSplit(line){
  const m=line.match(/^\s*((?:(?:Q(?:uestion)?\s*)?\d{1,4}[\).:\-]|[A-D][\).:\-]|(?:Correct\s+Answer|Answer|Ans\.?|Explanation)\s*[:\-]))\s*(.*)$/i);
  return m?{marker:m[1],body:m[2]}:{marker:'',body:line};
}

function normalizeExtractedLines(lines){
  const out=[];
  for(const raw of lines){
    const line=raw.trim();
    if(!line)continue;
    const matches=[...line.matchAll(/(?:^|\\s)(Q(?:uestion)?\\s*\\d{1,4}\\s*[\\.\\):\\-])(?=\\s|$)/gi)];
    if(matches.length<=1){out.push(line);continue;}
    let first=matches[0];
    if(first.index>0){
      const prefix=line.slice(0,first.index).trim();
      if(prefix)out.push(prefix);
    }
    for(let i=0;i<matches.length;i++){
      const start=matches[i].index;
      const endPos=i+1<matches.length?matches[i+1].index:line.length;
      const chunk=line.slice(start,endPos).trim();
      if(chunk)out.push(chunk);
    }
  }
  return out;
}

async function extractPdf(file){
  const buf=await file.arrayBuffer();
  const pdf=await pdfjsLib.getDocument({data:buf,useWorkerFetch:false,isEvalSupported:true}).promise;
  const pages=[];
  for(let p=1;p<=pdf.numPages;p++){
    const page=await pdf.getPage(p),c=await page.getTextContent();
    const items=c.items.filter(x=>x.str&&x.str.trim()).sort((a,b)=>(b.transform[5]-a.transform[5])||(a.transform[4]-b.transform[4]));
    const lines=[];let lastY=null,line='';
    for(const it of items){
      const y=Math.round(it.transform[5]);
      if(lastY!==null&&Math.abs(y-lastY)>3){if(line.trim())lines.push(line.trim());line='';}
      line+=(line?' ':'')+it.str;lastY=y;
    }
    if(line.trim())lines.push(line.trim());
    pages.push({page:p,lines:normalizeExtractedLines(lines)});
  }
  return pages;
}
function analyzePages(pages){
  const all=pages.flatMap(p=>p.lines);
  const qs=all.flatMap(l=>[...l.matchAll(/(?:^|\s)Q(?:uestion)?\s*(\d{1,4})\s*[\.\):\-]/gi)].map(m=>m[1]));
  const opts=all.filter(l=>/^\s*[A-D]\s*[\.\):\-]\s+/i.test(l));
  const ans=all.filter(l=>/^\s*(answer|ans\.?|correct\s+answer|explanation)\s*[:\-]/i.test(l));
  return {pages:pages.length,questions:new Set(qs).size,options:opts.length,answerMarkers:ans.length,nonempty:pages.filter(p=>p.lines.length).length};
}

$('analyze').onclick=async()=>{
  if(!selectedFile){msg('Select a PDF first.');return;}
  try{
    msg('Reading PDF on this phone…');
    const pages=await extractPdf(selectedFile),stats=analyzePages(pages),items=[];let q=0;
    for(const page of pages)for(const line of page.lines){const s=markerSplit(line);if(/^Q\s*\d+/i.test(s.marker))q++;items.push({page:page.page,q,marker:s.marker,body:s.body,translated:null});}
    currentJob={id:crypto.randomUUID(),name:selectedFile.name,pages,items,stats,createdAt:Date.now(),updatedAt:Date.now(),status:'analyzed'};
    await saveJob(currentJob);
    $('analysis').innerHTML='<h3>Analysis</h3><div class="stats"><div><b>Pages</b><span>'+stats.pages+'</span></div><div><b>Questions</b><span>'+stats.questions+'</span></div><div><b>Options</b><span>'+stats.options+'</span></div><div><b>Nonempty pages</b><span>'+stats.nonempty+'</span></div><div><b>Answer markers</b><span>'+stats.answerMarkers+'</span></div></div>';
    $('analysis').classList.remove('hidden');$('actions').classList.remove('hidden');$('translate').classList.remove('hidden');$('resume').classList.add('hidden');$('partial').classList.add('hidden');$('partialDocx').classList.add('hidden');
    msg('Analysis complete. Translation will run locally on this phone.');
  }catch(e){console.error(e);msg('PDF analysis failed: '+e.message);}
};

async function loadTokenizerPair(){
  const srcBlob=await modelAsset(F.srcTok,'English tokenizer',0,2);
  const tgtBlob=await modelAsset(F.tgtTok,'Telugu tokenizer',2,4);
  const metaBlob=await modelAsset(F.meta,'Tokenizer metadata',4,5);
  const srcJson=JSON.parse(await srcBlob.text());
  const tgtJson=JSON.parse(await tgtBlob.text());
  const meta=JSON.parse(await metaBlob.text());
  if(Number(meta.src_dict_size)!==32322||Number(meta.tgt_dict_size)!==122672||Number(meta.unk_id)!==3){
    throw new Error('Unexpected IndicTrans2 tokenizer metadata.');
  }
  return {tok:new IT2Tokenizer(srcJson,tgtJson,meta),meta};
}
async function loadGenerationConfig(){
  const genBlob=await modelAsset(F.gen,'Generation config',5,6);
  return JSON.parse(await genBlob.text());
}
async function createSessionFromBlobs(modelBlob,dataBlob,dataName){
  const modelBytes=new Uint8Array(await modelBlob.arrayBuffer());
  const dataBytes=new Uint8Array(await dataBlob.arrayBuffer());
  return ort.InferenceSession.create(modelBytes,{
    executionProviders:['wasm'],
    executionMode:'sequential',
    graphOptimizationLevel:'disabled',
    enableCpuMemArena:false,
    enableMemPattern:false,
    externalData:[{path:dataName,data:dataBytes}]
  });
}
async function ensureModel(){
  if(model)return model;
  msg('Preparing the Telugu translation engine (~318 MB first time only)…');
  $('progressWrap').classList.remove('hidden');
  modelProgress(1,'Checking saved model files on this phone…');
  try{await navigator.storage?.persist?.();}catch{}
  try{await cleanupOldModelAssets();}catch(e){console.warn('Old model cache cleanup failed:',e);}
  const pair=await loadTokenizerPair();
  const gen=await loadGenerationConfig();
  const ip=new IndicProcessor({inference:true});
  const encBlob=await modelAsset(F.enc,'Encoder graph',8,10);
  const encData=await modelAsset(F.encData,'Encoder weights',10,34);
  modelProgress(36,'Opening encoder…');
  const enc=await createSessionFromBlobs(encBlob,encData,F.encData);
  const decBlob=await modelAsset(F.dec,'Decoder graph',38,40);
  const decPastBlob=await modelAsset(F.decPast,'Decoder-with-past graph',40,42);
  const decData=await modelAsset(F.decData,'Shared decoder weights',42,84);
  modelProgress(86,'Opening decoder…');
  const dec=await createSessionFromBlobs(decBlob,decData,F.decData);
  modelProgress(90,'Opening decoder-with-past…');
  const decPast=await createSessionFromBlobs(decPastBlob,decData,F.decData);
  const numLayers=(dec.outputNames.length-1)/4;
  if(!Number.isInteger(numLayers)||numLayers<=0)throw new Error('Unexpected IndicTrans2 decoder output layout.');
  model={tok:pair.tok,meta:pair.meta,ip,startId:Number(gen.decoder_start_token_id??2),eosId:Number(gen.eos_token_id??2),numLayers,enc,dec,decPast};
  modelProgress(100,'Translation engine ready on this phone.');
  msg('Translation engine ready. Future runs reuse the saved model files.');
  return model;
}
function i64(a){return new ort.Tensor('int64',BigInt64Array.from(a,BigInt),[1,a.length]);}

function greedyArgmax(data,dims){
  const [B,T,V]=dims;
  const lastOff=(B-1)*T*V+(T-1)*V;
  let best=-Infinity,bestIdx=0;
  for(let i=0;i<V;i++){const v=data[lastOff+i];if(v>best){best=v;bestIdx=i;}}
  return bestIdx;
}
function collectPresents(out,numLayers){
  const kinds=['decoder.key','decoder.value','encoder.key','encoder.value'];
  const arr=new Array(numLayers*4);
  for(let i=0;i<numLayers;i++)for(let j=0;j<4;j++)arr[i*4+j]=out[`present.${i}.${kinds[j]}`];
  if(arr.some(x=>!x))throw new Error('IndicTrans2 decoder cache outputs are incomplete.');
  return arr;
}
async function translateText(text){
  const m=await ensureModel();
  m.ip.resetQueue();
  const [prepared]=m.ip.preprocessBatch([text],{srcLang:'eng_Latn',tgtLang:'tel_Telu'});
  const encoded=m.tok.encodeSrc(prepared);
  let ids=encoded.input_ids;
  if(ids.length>256)ids=ids.slice(0,255).concat(ids[ids.length-1]);
  const mask=encoded.attention_mask.slice(0,ids.length);
  const encOut=await m.enc.run({input_ids:i64(ids),attention_mask:i64(mask)});
  const hidden=encOut.last_hidden_state;
  let out=await m.dec.run({input_ids:i64([m.startId]),encoder_hidden_states:hidden,encoder_attention_mask:i64(mask)});
  let next=greedyArgmax(out.logits.data,out.logits.dims);
  const generated=[m.startId,next];
  let presents=collectPresents(out,m.numLayers);
  for(let step=1;step<128&&next!==m.eosId;step++){
    const feeds={input_ids:i64([next]),encoder_attention_mask:i64(mask)};
    const kinds=['decoder.key','decoder.value','encoder.key','encoder.value'];
    for(let i=0;i<presents.length;i++){
      const layer=Math.floor(i/4),kind=kinds[i%4];
      feeds[`past_key_values.${layer}.${kind}`]=presents[i];
    }
    out=await m.decPast.run(feeds);
    next=greedyArgmax(out.logits.data,out.logits.dims);
    generated.push(next);
    presents=collectPresents(out,m.numLayers);
    if(step%8===0)await new Promise(r=>setTimeout(r,0));
  }
  const raw=m.tok.decodeTgt(generated);
  const [final]=m.ip.postprocessBatch([raw],{lang:'tel_Telu'});
  return final.trim();
}
function translatable(item){return item.body.trim()&&!/^\d[\d\s|/_.:\-]*$/.test(item.body.trim());}
function completedCount(){return currentJob.items.filter(x=>translatable(x)&&x.translated!==null).length;}
function totalCount(){return currentJob.items.filter(translatable).length;}

async function translateLoop(){
  if(!currentJob||running)return;
  running=true;cancelled=false;$('progressWrap').classList.remove('hidden');$('translate').disabled=true;$('resume').disabled=true;
  try{
    const total=totalCount();let done=completedCount();
    for(const item of currentJob.items){
      if(cancelled)break;
      if(!translatable(item)){item.translated=item.body;continue;}
      if(item.translated!==null)continue;
      $('detail').textContent='Translating question '+(item.q||'?')+' • '+(done+1)+' of '+total;
      item.translated=await translateText(item.body);done++;
      currentJob.updatedAt=Date.now();currentJob.status=done===total?'complete':'paused';await saveJob(currentJob);
      const pct=Math.round(done*100/Math.max(total,1));$('bar').style.width=pct+'%';$('progressText').textContent=pct+'%';$('partial').classList.remove('hidden');$('partialDocx').classList.remove('hidden');
    }
    if(done===total){currentJob.status='complete';await saveJob(currentJob);$('bar').style.width='100%';$('progressText').textContent='100%';$('detail').textContent='All translatable lines completed.';msg('Translation complete.');await makeCompleteOutputs();}
    else{currentJob.status='paused';await saveJob(currentJob);$('resume').classList.remove('hidden');msg('Translation paused safely. You can resume later.');}
  }catch(e){console.error(e);currentJob.status='paused';currentJob.updatedAt=Date.now();await saveJob(currentJob);$('resume').classList.remove('hidden');msg('Translation stopped safely at the last saved item: '+e.message);}
  finally{running=false;$('translate').disabled=false;$('resume').disabled=false;}
}

function wrapLines(font,text,size,max){
  const words=text.split(/\s+/),out=[];let line='';
  for(const w of words){const test=line?line+' '+w:w;if(font.widthOfTextAtSize(test,size)>max){if(line)out.push(line);line=w;}else line=test;}
  if(line)out.push(line);return out;
}
async function buildPdf(job){
  const pdf=await PDFDocument.create();pdf.registerFontkit(fontkit);
  const fontBytes=await fetch('./fonts/NotoSansTelugu-Regular.ttf').then(r=>{if(!r.ok)throw new Error('Bundled Noto Sans Telugu Regular font is missing.');return r.arrayBuffer();});
  const boldBytes=await fetch('./fonts/NotoSansTelugu-Bold.ttf').then(r=>{if(!r.ok)throw new Error('Bundled Noto Sans Telugu Bold font is missing.');return r.arrayBuffer();});
  const font=await pdf.embedFont(fontBytes,{subset:true});
  const boldFont=await pdf.embedFont(boldBytes,{subset:true});
  for(const page of job.pages){
    const p=pdf.addPage([595.28,841.89]);let y=805;
    const pageItems=job.items.filter(x=>x.page===page.page);
    for(const item of pageItems){
      const text=(item.marker?item.marker+' ':'')+(item.translated??item.body);if(!text.trim())continue;
      const drawFont=/^Q(?:uestion)?\s*\d+/i.test(item.marker)?boldFont:font; for(const ln of wrapLines(drawFont,text,10.5,520)){if(y<45)break;p.drawText(ln,{x:36,y,size:10.5,font:drawFont,color:rgb(.08,.1,.14)});y-=15;}
    }
  }
  return pdf.save();
}
async function buildDocx(job){
  const children=[];
  for(const page of job.pages){for(const item of job.items.filter(x=>x.page===page.page)){const text=(item.marker?item.marker+' ':'')+(item.translated??item.body);if(text.trim())children.push(new Paragraph({children:[new TextRun({text,font:'Noto Sans Telugu',bold:/^Q(?:uestion)?\s*\d+/i.test(item.marker),size:21})]}));}children.push(new Paragraph({text:''}));}
  return Packer.toBlob(new Document({sections:[{children}]}));
}
function blobToBase64(blob){return new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(String(r.result).split(',')[1]);r.onerror=reject;r.readAsDataURL(blob);});}
async function saveAndShare(blob,name){
  const data=await blobToBase64(blob),r=await Filesystem.writeFile({path:name,data,directory:Directory.Documents,recursive:true});
  try{await Share.share({title:name,text:'TeluguSetu output',url:r.uri,dialogTitle:'Share TeluguSetu file'});}catch{}
  return r.uri;
}
async function makeCompleteOutputs(){
  $('downloads').innerHTML='<p class="small">Creating final PDF and DOCX…</p>';
  try{
    const pdf=await buildPdf(currentJob),docx=await buildDocx(currentJob);
    const base=currentJob.name.replace(/\.pdf$/i,'');
    await saveAndShare(new Blob([pdf],{type:'application/pdf'}),base+'_Telugu.pdf');
    await saveAndShare(docx,base+'_Telugu.docx');
    $('downloads').innerHTML='<p class="small">Final files saved to Android Documents. You can share them from the system share dialog.</p>';
  }catch(e){$('downloads').textContent='Output generation failed: '+e.message;}
}
$('partial').onclick=async()=>{
  if(!currentJob)return;
  try{msg('Creating partial translated PDF…');const bytes=await buildPdf(currentJob);await saveAndShare(new Blob([bytes],{type:'application/pdf'}),currentJob.name.replace(/\.pdf$/i,'')+'_Telugu_Partial.pdf');msg('Partial PDF saved and ready to share.');}
  catch(e){msg('Partial PDF failed: '+e.message);}
};
$('partialDocx').onclick=async()=>{
  if(!currentJob)return;
  try{msg('Creating partial translated DOCX…');const blob=await buildDocx(currentJob);await saveAndShare(blob,currentJob.name.replace(/\.pdf$/i,'')+'_Telugu_Partial.docx');msg('Partial DOCX saved and ready to share.');}
  catch(e){msg('Partial DOCX failed: '+e.message);}
};
$('translate').onclick=translateLoop;
$('resume').onclick=translateLoop;

(async()=>{
  try{
    const latest=await loadLatestJob();
    if(latest){
      currentJob=latest;$('fileLabel').textContent=latest.name;
      const done=completedCount(),total=totalCount();
      $('analysis').innerHTML='<h3>Saved translation</h3><div class="stats"><div><b>Pages</b><span>'+latest.stats.pages+'</span></div><div><b>Questions</b><span>'+latest.stats.questions+'</span></div><div><b>Saved progress</b><span>'+done+' / '+total+'</span></div></div>';
      $('analysis').classList.remove('hidden');$('actions').classList.remove('hidden');$('partial').classList.remove('hidden');$('partialDocx').classList.remove('hidden');
      if(done<total)$('resume').classList.remove('hidden');else{ $('translate').classList.add('hidden');$('bar').style.width='100%';$('progressText').textContent='100%';}
      $('progressWrap').classList.remove('hidden');msg('A saved translation is available. Resume whenever you are ready.');
    }
  }catch(e){console.warn(e);}
})();