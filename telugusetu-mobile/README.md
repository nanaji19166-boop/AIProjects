# TeluguSetu Android

Local-first English PDF → Telugu PDF/DOCX Android app.

## Translation engine

TeluguSetu uses the INT8 ONNX export of IndicTrans2 200M. The mobile pipeline is:

1. PDF.js extracts text while retaining page boundaries.
2. TeluguSetu detects questions/options/answer/explanation markers.
3. IndicProcessor performs the model-required preprocessing, including language tags, punctuation/entity normalization and shared-script normalization.
4. The validated BPE tokenizer encodes the source.
5. Source IDs at or above the model's canonical encoder vocabulary size are remapped to `<unk>` (ID 3).
6. ONNX Runtime Web runs the encoder.
7. The first decoder step uses `decoder_model.onnx`.
8. Later steps use `decoder_with_past_model.onnx` and its KV cache.
9. Target IDs are safely decoded and IndicProcessor postprocesses the result back into Telugu.
10. Translation checkpoints are stored locally so interrupted work can resume.

The model export documents a maximum source/target position of 256 tokens. TeluguSetu therefore caps an individual encoded source at 256 tokens rather than sending an oversized sequence to the ONNX graph.

## Model files

The app downloads and caches these files from Hugging Face on first use:

- `encoder_model.onnx`
- `encoder_model.onnx.data`
- `decoder_model.onnx`
- `decoder_with_past_model.onnx`
- `decoder_shared.onnx.data`
- `tokenizer_src.json`
- `tokenizer_tgt.json`
- `tokenizer_meta.json`
- `generation_config.json`

The model is not bundled into the APK. This keeps the APK small; the translation model is downloaded once and cached on the device.

## APK build

GitHub Actions:

- Node.js 22
- Capacitor 7
- Java 21
- Android SDK platform 35 / build tools 35.0.0
- Gradle `assembleDebug`
- Noto Sans Telugu Regular + Bold + OFL license bundled into the app assets
- APK uploaded as a GitHub Actions artifact and prerelease

The debug APK is signed by the Android SDK's debug key and is suitable for direct testing/install. For a long-lived distributable APK where updates must install over the previous build, configure a stable private release keystore in GitHub Actions; never commit that keystore to the repository.

## Important device requirements

- First translation requires internet access to download the model.
- After all model files are cached, inference is local; PDF text and translations are not sent to a translation API.
- Keep adequate free storage for the model cache and generated files.
- Large PDFs are processed incrementally, but Android may suspend WebView JavaScript when the app is backgrounded. Checkpoint/resume protects progress; it is not a background-service guarantee.

## Output

The app creates a structurally reconstructed Telugu PDF and DOCX. The current v0.x exporter preserves page boundaries and question/option/answer order but does not yet reproduce every original visual element, image, table or typography exactly.

See `THIRD_PARTY_NOTICES.md` for attribution and licenses.
