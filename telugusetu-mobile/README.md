# TeluguSetu Android
Local-first English PDF → Telugu PDF/DOCX Android app.

The app uses Capacitor, PDF.js, ONNX Runtime Web and the IndicTrans2 INT8 200M ONNX bundle. Translation runs on the Android device. Each completed line is checkpointed in IndexedDB so a stopped job can resume. A partial translated PDF can be exported at any time.

The model is downloaded from Hugging Face on first use rather than bundled into the APK because the current ONNX bundle is about 318 MB and uses external weight files. Future builds can package/cache the model more aggressively.

No Play Store publication is required. The generated debug APK can be installed directly on Android.