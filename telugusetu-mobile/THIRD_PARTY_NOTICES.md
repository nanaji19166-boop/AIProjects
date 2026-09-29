# TeluguSetu third-party notices

## IndicTrans2

TeluguSetu Mobile uses the INT8 ONNX export:

- `hari31416/indictrans2-en-indic-dist-200M-ONNX-int8`
- Upstream base model: AI4Bharat IndicTrans2 En→Indic Distilled 200M
- License: MIT

The app uses the model's BPE tokenizer, tokenizer metadata, encoder, first-step decoder, decoder-with-past graph, and shared decoder external weights. IndicTrans2 requires its IndicProcessor preprocessing/postprocessing pipeline; TeluguSetu bundles a browser-compatible JavaScript port derived from the validated Anuvaad/prashnam implementation.

## Noto Sans Telugu

The generated PDF embeds Noto Sans Telugu Regular and Bold. Noto Telugu is licensed under the SIL Open Font License 1.1. The APK build includes the OFL text at `fonts/OFL.txt`.

## ONNX Runtime Web

- Package: `onnxruntime-web`
- License: MIT

## Capacitor

Capacitor and its official plugins are open-source packages with their respective licenses retained in npm-installed dependencies.

This file is an attribution notice; consult the upstream licenses for the complete terms.
