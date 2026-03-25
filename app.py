import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from collections import Counter

st.set_page_config(page_title="🔑 Détecteur de Tonalité DUAL", layout="wide")
st.title("🔑 Détecteur de Tonalité Musicale – Analyse Scindée")
st.markdown("**Analyse différenciée : 40 premières secondes VS Reste du morceau**")

# --- Configuration & Constantes (Inchangées) ---
CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B',
    'E major': '12B', 'F major': '7B', 'F# major': '2B', 'G major': '9B',
    'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A',
    'E minor': '9A', 'F minor': '4A', 'F# minor': '11A', 'G minor': '6A',
    'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
}
FR_TO_EN = {'Do': 'C', 'Do#': 'C#', 'Ré': 'D', 'Ré#': 'D#', 'Mi': 'E', 'Fa': 'F', 'Fa#': 'F#', 'Sol': 'G', 'Sol#': 'G#', 'La': 'A', 'La#': 'A#', 'Si': 'B'}
KEYS_FR = ['Do', 'Do#', 'Ré', 'Ré#', 'Mi', 'Fa', 'Fa#', 'Sol', 'Sol#', 'La', 'La#', 'Si']
KEYS_EN = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

PROFILES = {
    'Krumhansl': {
        'major': np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]),
        'minor': np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.48])
    },
    'Temperley': {
        'major': np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0]),
        'minor': np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0])
    },
    'Aarden': {
        'major': np.array([17.7661, 0.145624, 14.9265, 0.160186, 19.8049, 11.3587, 0.291248, 22.062, 0.145624, 8.15494, 0.232998, 4.95122]),
        'minor': np.array([18.2648, 0.737619, 14.0499, 16.8599, 0.702494, 14.4362, 0.702494, 18.6161, 4.56621, 1.93186, 7.37619, 1.75623])
    }
}

def normalize(arr):
    arr = arr - np.mean(arr)
    std = np.std(arr)
    return arr / std if std > 0 else arr

def detect_key_segment(chroma_weighted):
    votes = []
    all_scores = np.zeros((12, 2))
    for profile_name, profile in PROFILES.items():
        maj, min_ = normalize(profile['major']), normalize(profile['minor'])
        chroma_n = normalize(chroma_weighted)
        corrs = []
        for i in range(12):
            mc = np.corrcoef(chroma_n, np.roll(maj, i))[0, 1]
            nc = np.corrcoef(chroma_n, np.roll(min_, i))[0, 1]
            corrs.append((mc, nc))
            all_scores[i][0] += mc
            all_scores[i][1] += nc
        best_i = np.argmax([max(m, n) for m, n in corrs])
        votes.append((best_i, corrs[best_i][0] > corrs[best_i][1]))
    
    vote_counter = Counter(votes)
    best_vote, count = vote_counter.most_common(1)[0]
    if count == 1:
        flat_best = np.argmax(all_scores.flatten())
        best_vote = (flat_best // 2, (flat_best % 2) == 0)
    return best_vote, all_scores / len(PROFILES)

def compute_confidence(all_scores, best_idx, is_major):
    mode_idx = 0 if is_major else 1
    best_score = all_scores[best_idx][mode_idx]
    all_flat = all_scores.flatten()
    sorted_scores = np.sort(all_flat)[::-1]
    second_best = sorted_scores[1] if sorted_scores[0] == best_score else sorted_scores[0]
    gap = best_score - second_best
    confidence = int(np.clip(((best_score + 1) / 2) * 100 + min(gap * 30, 15), 0, 99))
    return confidence

# --- Nouvelle fonction de traitement par section ---
def analyser_portion(y_section, sr):
    if len(y_section) < sr * 2: return None
    
    segment_sec, hop_sec = 15, 7
    segment_samples, hop_samples = segment_sec * sr, hop_sec * sr
    
    votes, scores_list = [], []
    starts = range(0, max(1, len(y_section) - segment_samples), hop_samples)
    
    for start in starts:
        seg = y_section[start:start + segment_samples]
        if len(seg) < sr * 2: continue
        chroma_seg = librosa.feature.chroma_cqt(y=seg, sr=sr, bins_per_octave=36, n_octaves=7)
        rms = librosa.feature.rms(y=seg)[0][:chroma_seg.shape[1]]
        weights = rms / (rms.sum() + 1e-8)
        chroma_w = np.average(chroma_seg, axis=1, weights=weights)
        
        vote, scores = detect_key_segment(chroma_w)
        votes.append(vote)
        scores_list.append(scores)
    
    if not votes: return None
    
    best_vote = Counter(votes).most_common(1)[0][0]
    avg_scores = np.mean(scores_list, axis=0)
    conf = compute_confidence(avg_scores, best_vote[0], best_vote[1])
    
    # Formattage texte
    note_fr = KEYS_FR[best_vote[0]]
    mode_en = "major" if best_vote[1] else "minor"
    camelot = CAMELOT_MAP.get(f"{FR_TO_EN[note_fr]} {mode_en}", "?")
    
    return {
        "tonalite": f"{note_fr} {'majeur' if best_vote[1] else 'mineur'}",
        "camelot": camelot,
        "confiance": conf,
        "scores": avg_scores,
        "idx": best_vote[0],
        "is_maj": best_vote[1]
    }

# --- Interface & Logique de téléchargement ---
option = st.radio("Source audio", ["Fichier local", "YouTube"])
audio_path = None

if option == "Fichier local":
    uploaded = st.file_uploader("Audio", type=["mp3", "wav", "m4a", "flac"])
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
            tmp.write(uploaded.getvalue())
            audio_path = tmp.name
else:
    url = st.text_input("Lien YouTube")
    if url and st.button("Analyser"):
        with st.spinner("Téléchargement..."):
            os.system(f'yt-dlp -x --audio-format wav -o "temp.wav" "{url}"')
            audio_path = "temp.wav"

if audio_path and os.path.exists(audio_path):
    with st.spinner("Analyse des deux sections en cours..."):
        y, sr = librosa.load(audio_path, sr=22050)
        y_harmonic, _ = librosa.effects.hpss(y, margin=3.0)
        
        # Correction tuning globale
        tuning = librosa.estimate_tuning(y=y_harmonic, sr=sr)
        if abs(tuning) > 0.05:
            y_harmonic = librosa.effects.pitch_shift(y=y_harmonic, sr=sr, n_steps=-tuning)

        # --- SCISSION DES DONNÉES ---
        split_point = 40 * sr
        y_intro = y_harmonic[:split_point]
        y_reste = y_harmonic[split_point:]
        
        res1 = analyser_portion(y_intro, sr)
        res2 = analyser_portion(y_reste, sr) if len(y_reste) > sr*2 else None

        # --- AFFICHAGE ---
        st.header("🎯 Résultats de l'analyse double")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("⏱️ 40 premières secondes")
            if res1:
                st.metric("Tonalité", res1["tonalite"])
                st.metric("Code Camelot", res1["camelot"])
                st.progress(res1["confiance"]/100, text=f"Confiance: {res1['confiance']}%")
            else:
                st.warning("Audio trop court pour la section 1")

        with col_right:
            st.subheader("⏳ Reste du morceau")
            if res2:
                st.metric("Tonalité", res2["tonalite"])
                st.metric("Code Camelot", res2["camelot"])
                st.progress(res2["confiance"]/100, text=f"Confiance: {res2['confiance']}%")
            else:
                st.info("Pas assez de données pour le reste du morceau")

        # Alerte si changement de tonalité
        if res1 and res2 and res1["tonalite"] != res2["tonalite"]:
            st.warning(f"⚠️ **Modulation détectée !** Le morceau semble passer de {res1['tonalite']} à {res2['tonalite']}.")
        elif res1 and res2:
            st.success("✅ Tonalité stable sur l'ensemble du morceau.")

    # Nettoyage
    if os.path.exists(audio_path): os.remove(audio_path)
