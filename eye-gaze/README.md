# Eye Gaze Detection & Change Point Detection

A modular project demonstrating:
- Eye/pupil center extraction from face images using OpenCV (with optional dlib landmarks)
- Dataset construction and anomaly injection
- Training Decision Tree and Random Forest regressors
- Training a Keras neural network classifier (with L2 regularization) to detect anomaly-displaced points
- Two change point detection algorithms (Adaptive CUSUM-like and rolling statistical)
- Evaluation scripts and pipeline orchestration

## Quickstart
1. Create a virtual environment (Python 3.8+)
```bash
python -m venv venv
source venv/bin/activate    # linux/mac
venv\Scripts\activate     # windows
```
2. Install requirements
```bash
pip install -r requirements.txt
```
3. Prepare an `images/` folder with face images (jpg/png).
4. Run pipeline
```bash
python scripts/run_pipeline.py --img_dir ./images --out_dir ./outputs --anomaly_rate 0.05
```

## Project layout
See the repository tree in this README. Each module is designed to be extensible and testable.

## Notes
- dlib-based landmarking is *optional* — if you want dlib, download `shape_predictor_68_face_landmarks.dat` and place it in the repo root or give the path in config.
- The CPD implementations are intentionally simple and educational. For production, consider packages like `ruptures` or Bayesian online CPD implementations.
