from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import librosa
import tempfile
import parselmouth
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

latest_data = {}

@app.get("/")
def home():
    return {
        "message": "Acoustic Speech Analysis API Running"
    }


@app.post("/analyze")
async def analyze_audio(audio_file: UploadFile = File(...)):

    global latest_data

    # ==========================
    # Save uploaded WAV file
    # ==========================
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(await audio_file.read())
        temp_path = temp_audio.name

    # ==========================
    # Load Audio
    # ==========================
    y, sr = librosa.load(temp_path, sr=None)

    duration = len(y) / sr

    # ==========================
    # Praat Sound Object
    # ==========================
    sound = parselmouth.Sound(temp_path)

    # ==========================
    # Voice Quality
    # ==========================
    point_process = parselmouth.praat.call(
        sound,
        "To PointProcess (periodic, cc)",
        75,
        500
    )

    jitter_local = parselmouth.praat.call(
        point_process,
        "Get jitter (local)",
        0,
        0,
        0.0001,
        0.02,
        1.3
    )

    shimmer_local = parselmouth.praat.call(
        [sound, point_process],
        "Get shimmer (local)",
        0,
        0,
        0.0001,
        0.02,
        1.3,
        1.6
    )

    harmonicity = parselmouth.praat.call(
        sound,
        "To Harmonicity (cc)",
        0.01,
        75,
        0.1,
        1.0
    )

    mean_hnr = parselmouth.praat.call(
        harmonicity,
        "Get mean",
        0,
        0
    )

    # ==========================
    # Pitch Analysis
    # ==========================
    pitch = sound.to_pitch()

    frequencies = pitch.selected_array["frequency"]

    voiced = frequencies[frequencies > 0]

    if len(voiced) > 0:
        mean_pitch = float(np.mean(voiced))
        min_pitch = float(np.min(voiced))
        max_pitch = float(np.max(voiced))
    else:
        mean_pitch = 0
        min_pitch = 0
        max_pitch = 0

    pitch_times = pitch.xs()

    pitch_values = [
        None if x == 0 else float(x)
        for x in frequencies
    ]

    # ==========================
    # Formants
    # ==========================
    formant = sound.to_formant_burg()

    duration_sound = sound.get_total_duration()

    formant_times = np.arange(
        0.05,
        duration_sound,
        0.05
    )

    f1 = []
    f2 = []
    f3 = []

    for t in formant_times:

        f1_value = formant.get_value_at_time(1, t)
        f2_value = formant.get_value_at_time(2, t)
        f3_value = formant.get_value_at_time(3, t)

        f1.append(
            None if np.isnan(f1_value)
            else float(f1_value)
        )

        f2.append(
            None if np.isnan(f2_value)
            else float(f2_value)
        )

        f3.append(
            None if np.isnan(f3_value)
            else float(f3_value)
        )

    # ==========================
    # Waveform
    # ==========================
    waveform_samples = min(10000, len(y))

    indices = np.linspace(
        0,
        len(y) - 1,
        waveform_samples,
        dtype=int
    )

    waveform_time = indices / sr

    waveform_amplitude = y[indices]

    # ==========================
    # MFCC
    # ==========================
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    # ==========================
    # Spectrogram
    # ==========================
    spectrogram = np.abs(
        librosa.stft(y)
    )

    spectrogram_db = librosa.amplitude_to_db(
        spectrogram,
        ref=np.max
    )

    # ==========================
    # Audio Statistics
    # ==========================
    rms = librosa.feature.rms(y=y)

    zcr = librosa.feature.zero_crossing_rate(y)

    spectral_centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )
    # ==========================
    # Stress Analysis
    # ==========================

    # Pitch Score
    if mean_pitch < 160:
        pitch_score = 0
    elif mean_pitch < 180:
        pitch_score = 25
    elif mean_pitch < 200:
        pitch_score = 50
    elif mean_pitch < 220:
        pitch_score = 75
    else:
        pitch_score = 100

    # Jitter Score
    jitter_percent = float(jitter_local * 100)

    if jitter_percent < 1:
        jitter_score = 0
    elif jitter_percent < 1.5:
        jitter_score = 25
    elif jitter_percent < 2:
        jitter_score = 50
    elif jitter_percent < 3:
        jitter_score = 75
    else:
        jitter_score = 100

    # Shimmer Score
    shimmer_percent = float(shimmer_local * 100)

    if shimmer_percent < 5:
        shimmer_score = 0
    elif shimmer_percent < 8:
        shimmer_score = 25
    elif shimmer_percent < 10:
        shimmer_score = 50
    elif shimmer_percent < 15:
        shimmer_score = 75
    else:
        shimmer_score = 100

    # HNR Score
    if mean_hnr > 20:
        hnr_score = 0
    elif mean_hnr > 15:
        hnr_score = 25
    elif mean_hnr > 10:
        hnr_score = 50
    elif mean_hnr > 5:
        hnr_score = 75
    else:
        hnr_score = 100

    # Spectral Centroid Score
    centroid = float(np.mean(spectral_centroid))

    if centroid < 3000:
        centroid_score = 0
    elif centroid < 4000:
        centroid_score = 25
    elif centroid < 5000:
        centroid_score = 50
    elif centroid < 6000:
        centroid_score = 75
    else:
        centroid_score = 100

    # Final Weighted Stress Score
    stress_score = round(
        pitch_score * 0.25 +
        jitter_score * 0.20 +
        shimmer_score * 0.20 +
        hnr_score * 0.20 +
        centroid_score * 0.15,
        2
    )

    if stress_score < 20:
        stress_level = "Relaxed"
    elif stress_score < 40:
        stress_level = "Mild Stress"
    elif stress_score < 60:
        stress_level = "Moderate Stress"
    elif stress_score < 80:
         stress_level = "High Stress"
    else:
        stress_level = "Severe Stress"
    # ==========================
    # Store Data
    # ==========================
    latest_data = {
        "stress_analysis": {
            "stress_score": stress_score,
            "stress_level": stress_level,
            "pitch_score": pitch_score,
            "jitter_score": jitter_score,
            "shimmer_score": shimmer_score,
            "hnr_score": hnr_score,
            "centroid_score": centroid_score
        },

        "audio_info": {
            "sample_rate": sr,
            "duration_seconds": round(duration, 2),
            "total_samples": len(y)
        },

        "pitch_statistics": {
            "mean_pitch_hz": round(mean_pitch, 2),
            "min_pitch_hz": round(min_pitch, 2),
            "max_pitch_hz": round(max_pitch, 2)
        },

        "audio_statistics": {
            "mean_rms_energy":
                float(np.mean(rms)),

            "mean_zero_crossing_rate":
                float(np.mean(zcr)),

            "mean_spectral_centroid":
                float(np.mean(spectral_centroid))
        },

        "voice_quality": {
            "jitter_local_percent":
                round(float(jitter_local * 100), 4),

            "shimmer_local_percent":
                round(float(shimmer_local * 100), 4),

            "harmonic_to_noise_ratio_db":
                round(float(mean_hnr), 2)
        },

        "waveform": {
            "time": waveform_time.tolist(),
            "amplitude": waveform_amplitude.tolist()
        },

        "pitch_contour": {
            "time": pitch_times.tolist(),
            "frequency": pitch_values
        },

        "formants": {
            "time": formant_times.tolist(),
            "F1": f1,
            "F2": f2,
            "F3": f3
        },

        "mfcc": {
            "coefficients": mfcc.tolist()
        },

        "spectrogram": {
            "data": spectrogram_db[:64, :50].tolist()
        }
    }

    return {
        "status": "Analysis Complete",
        "audio_info": latest_data["audio_info"],
        "pitch_statistics": latest_data["pitch_statistics"],
        "audio_statistics": latest_data["audio_statistics"],
        "voice_quality": latest_data["voice_quality"],
        "stress_analysis": latest_data["stress_analysis"]
    }


@app.get("/waveform")
def get_waveform():
    if not latest_data:
        raise HTTPException(status_code=404, detail="No analysis available")
    return latest_data["waveform"]


@app.get("/pitch")
def get_pitch():
    if not latest_data:
        raise HTTPException(status_code=404, detail="No analysis available")
    return {
        "pitch_statistics": latest_data["pitch_statistics"],
        "pitch_contour": latest_data["pitch_contour"]
    }


@app.get("/formants")
def get_formants():
    if not latest_data:
        raise HTTPException(status_code=404, detail="No analysis available")
    return latest_data["formants"]


@app.get("/mfcc")
def get_mfcc():
    if not latest_data:
        raise HTTPException(status_code=404, detail="No analysis available")
    return latest_data["mfcc"]


@app.get("/spectrogram")
def get_spectrogram():
    if not latest_data:
        raise HTTPException(status_code=404, detail="No analysis available")
    return latest_data["spectrogram"]


@app.get("/voice_quality")
def get_voice_quality():
    if not latest_data:
        raise HTTPException(status_code=404, detail="No analysis available")
    return latest_data["voice_quality"]


@app.get("/stress_analysis")
def get_stress_analysis():
    if not latest_data:
        raise HTTPException(
            status_code=404,
            detail="No analysis available"
        )
    return latest_data["stress_analysis"]