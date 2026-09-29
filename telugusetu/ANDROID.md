# TeluguSetu — run on Android via Chrome

## Recommended first run
The translator engine runs on a Windows/Linux/macOS computer. Your Android phone is the mobile UI.

1. On the computer, install Python 3.10/3.11.
2. Clone this repository and enter `telugusetu`.
3. Create a virtual environment and install `requirements.txt`.
4. Start the server with:
   `uvicorn app:app --host 0.0.0.0 --port 8000`
5. Connect the Android phone and computer to the same Wi-Fi.
6. On Android Chrome, open the computer's LAN address, for example:
   `http://192.168.1.20:8000`

The first translation run downloads the official IndicTrans2 distilled 200M model from Hugging Face. After it is downloaded, inference is local and does not require a Google/paid translation API key.

## Why this design
IndicTrans2 officially supports English (`eng_Latn`) to Telugu (`tel_Telu`). The 200M distilled checkpoint is the smaller official HF checkpoint and is suitable for a practical CPU-first prototype. The model itself is released under MIT by AI4Bharat.

## Current validation
The uploaded February 2026 MCQ PDF was structurally checked:
- 63 pages
- 115 questions, numbered Q1–Q115
- 460 A/B/C/D option lines
- 116 answer labels

The source analysis test passes.

## Important
A complete translation test requires downloading the model weights. The current execution environment has no outbound network access, so the model weights could not be downloaded here. Do not treat the project as end-to-end tested until the first local machine run completes with the model.
