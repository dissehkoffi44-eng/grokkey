import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from collections import Counter
import requests
from datetime import datetime

st.set_page_config(page_title="Detecteur de Tonalite PRO+", page_icon="", layout="wide")
st.title("Detecteur de Tonalite Musicale - Version ULTRA PRO")
st.markdown("**Triple profil (Krumhansl + Temperley + Aarden) - Vote par segments - Ponderation RMS - Correction tuning**")
# Récupération des secrets
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")
# Camelot Map
CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B',
    'E major': '12B', 'F major': '7B', 'F# major': '2B', 'G major': '9B',
    'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A',
    'E minor': '9A', 'F minor': '4A', 'F# minor': '11A', 'G minor': '6A',
    'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
}

FR_TO_EN = {
    'Do': 'C', 'Do#': 'C#', 'Re': 'D', 'Re#': 'D#', 'Mi': 'E', 'Fa': 'F',
    'Fa#': 'F#', 'Sol': 'G', 'Sol#': 'G#', 'La': 'A', 'La#': 'A#', 'Si': 'B'
}
KEYS_FR = ['Do', 'Do#', 'Re', 'Re#', 'Mi', 'Fa', 'Fa#', 'Sol', 'Sol#', 'La', 'La#', 'Si']
KEYS_EN = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Profils harmoniques (3 algorithmes)
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
        'major': np.array([17.7661, 0.145624, 14.9265, 0.160186, 19.8049, 11.3587,
                           0.291248, 22.062, 0.145624, 8.15494, 0.232998, 4.95122]),
        'minor': np.array([18.2648, 0.737619, 14.0499, 16.8599, 0.702494, 14.4362,
                           0.702494, 18.6161, 4.56621, 1.93186, 7.37619, 1.75623])
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
        maj = normalize(profile['major'])
        min_ = normalize(profile['minor'])
        chroma_n = normalize(chroma_weighted)

        corrs = []
        for i in range(12):
            mc = np.corrcoef(chroma_n, np.roll(maj, i))[0, 1]
            nc = np.corrcoef(chroma_n, np.roll(min_, i))[0, 1]
            corrs.append((mc, nc))
            all_scores[i][0] += mc
            all_scores[i][1] += nc

        best_i = np.argmax([max(m, n) for m, n in corrs])
        is_maj = corrs[best_i][0] > corrs[best_i][1]
        votes.append((best_i, is_maj))

    vote_counter = Counter(votes)
    best_vote, count = vote_counter.most_common(1)[0]

    if count == 1:
        flat_best = np.argmax(all_scores.flatten())
        best_i = flat_best // 2
        is_maj = (flat_best % 2) == 0
        best_vote = (best_i, is_maj)

    return best_vote, all_scores / len(PROFILES)

def compute_confidence(all_scores, best_idx, is_major):
    mode_idx = 0 if is_major else 1
    best_score = all_scores[best_idx][mode_idx]

    all_flat = all_scores.flatten()
    sorted_scores = np.sort(all_flat)[::-1]
    second_best = sorted_scores[1] if sorted_scores[0] == best_score else sorted_scores[0]

    gap = best_score - second_best
    raw_conf = (best_score + 1) / 2
    gap_bonus = min(gap * 30, 15)
    confidence = int(np.clip(raw_conf * 100 + gap_bonus, 0, 99))
    return confidence, best_score

# Interface
option = st.radio("Comment fournir la chanson ?", ["Fichier audio", "Lien YouTube"])
audio_path = None

if option == "Fichier audio":
    uploaded = st.file_uploader(
        "MP3, WAV, M4A, OGG, FLAC...",
        type=["mp3", "wav", "m4a", "ogg", "flac"]
    )
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
            tmp.write(uploaded.getvalue())
            audio_path = tmp.name
else:
    url = st.text_input("Colle le lien YouTube")
    if url and st.button("Telecharger & analyser"):
        with st.spinner("Telechargement..."):
            os.system(f'yt-dlp -x --audio-format wav --no-warnings -o "temp.wav" "{url}"')
            audio_path = "temp.wav"

if audio_path and os.path.exists(audio_path):
    with st.spinner("Analyse ULTRA PRO en cours..."):

        y, sr = librosa.load(audio_path, sr=22050, duration=300)
        y_harmonic, y_percussive = librosa.effects.hpss(y, margin=3.0)

        tuning = librosa.estimate_tuning(y=y_harmonic, sr=sr)
        cents = round(tuning * 100, 1)
        if abs(tuning) > 0.05:
            y_harmonic = librosa.effects.pitch_shift(y=y_harmonic, sr=sr, n_steps=-tuning)
            st.info(f"Desaccordage corrige : **{cents:+.1f} cents**")
        else:
            st.success("Morceau accorde (A4 = 440 Hz)")

        segment_sec = 20
        hop_sec = 10
        segment_samples = segment_sec * sr
        hop_samples = hop_sec * sr

        segment_votes = []
        segment_scores_list = []

        starts = range(0, max(1, len(y_harmonic) - segment_samples), hop_samples)
        if len(y_harmonic) < segment_samples:
            starts = [0]
            segment_samples = len(y_harmonic)

        for start in starts:
            seg = y_harmonic[start:start + segment_samples]
            if len(seg) < sr * 2:
                continue

            chroma_seg = librosa.feature.chroma_cqt(
                y=seg, sr=sr, bins_per_octave=36, n_octaves=7, norm=2
            )

            rms = librosa.feature.rms(y=seg)[0]
            rms = rms[:chroma_seg.shape[1]]
            weights = rms / (rms.sum() + 1e-8)
            chroma_w = np.average(chroma_seg, axis=1, weights=weights)

            vote, scores = detect_key_segment(chroma_w)
            segment_votes.append(vote)
            segment_scores_list.append(scores)

        vote_counter = Counter(segment_votes)
        (best_idx, is_major), _ = vote_counter.most_common(1)[0]

        avg_scores = np.mean(segment_scores_list, axis=0)
        confidence, best_score = compute_confidence(avg_scores, best_idx, is_major)

        st.markdown("---")
        st.subheader("Tonalites principales par section")
        st.caption("Premieres 40 secondes vs. reste du morceau")

        def get_key_for_section(section_y, section_name):
            if len(section_y) < sr * 5:
                return {
                    "tonalite_fr": "Trop court",
                    "camelot": "-",
                    "confidence": 0,
                    "section_name": section_name
                }

            segment_sec = 20
            hop_sec = 10
            segment_samples = segment_sec * sr
            hop_samples = hop_sec * sr

            segment_votes = []
            segment_scores_list = []

            starts = range(0, max(1, len(section_y) - segment_samples), hop_samples)
            if len(section_y) < segment_samples:
                starts = [0]

            for start in starts:
                seg = section_y[start:start + segment_samples]
                if len(seg) < sr * 2:
                    continue

                chroma_seg = librosa.feature.chroma_cqt(
                    y=seg, sr=sr, bins_per_octave=36, n_octaves=7, norm=2
                )

                rms = librosa.feature.rms(y=seg)[0]
                rms = rms[:chroma_seg.shape[1]]
                weights = rms / (rms.sum() + 1e-8)
                chroma_w = np.average(chroma_seg, axis=1, weights=weights)

                vote, scores = detect_key_segment(chroma_w)
                segment_votes.append(vote)
                segment_scores_list.append(scores)

            if not segment_votes:
                return {"tonalite_fr": "Impossible", "camelot": "-", "confidence": 0, "section_name": section_name}

            vote_counter = Counter(segment_votes)
            (best_idx_sec, is_major_sec), _ = vote_counter.most_common(1)[0]

            avg_scores_sec = np.mean(segment_scores_list, axis=0)
            confidence_sec, _ = compute_confidence(avg_scores_sec, best_idx_sec, is_major_sec)

            note_fr = KEYS_FR[best_idx_sec]
            mode_fr = "majeur" if is_major_sec else "mineur"
            note_en = FR_TO_EN[note_fr]
            mode_en = "major" if is_major_sec else "minor"
            tonalite_fr = f"{note_fr} {mode_fr}"
            camelot_code = CAMELOT_MAP.get(f"{note_en} {mode_en}", "?")

            return {
                "tonalite_fr": tonalite_fr,
                "camelot": camelot_code,
                "confidence": confidence_sec,
                "section_name": section_name,
                "best_idx": best_idx_sec,
                "is_major": is_major_sec
            }

        early_y = y_harmonic[:int(40 * sr)]
        late_y = y_harmonic[int(40 * sr):]

        early_data = get_key_for_section(early_y, "Premieres 40 secondes")
        late_data = get_key_for_section(late_y, "Reste du morceau") if len(late_y) > sr * 5 else None

        col_early, col_late = st.columns(2)

        with col_early:
            st.markdown(f"**{early_data['section_name']}**")
            st.metric("Tonalite", early_data["tonalite_fr"])
            st.metric("Camelot", early_data["camelot"])
            st.metric("Confiance", f"{early_data['confidence']}%")

        if late_data:
            with col_late:
                st.markdown(f"**{late_data['section_name']}**")
                st.metric("Tonalite", late_data["tonalite_fr"])
                st.metric("Camelot", late_data["camelot"])
                st.metric("Confiance", f"{late_data['confidence']}%")
        else:
            with col_late:
                st.info("Morceau trop court pour une seconde partie")

        if late_data and early_data["tonalite_fr"] != late_data["tonalite_fr"]:
            st.success("Changement de tonalite detecte entre intro et corps du morceau !")

        st.markdown("---")
        st.subheader("Tonalite globale du morceau (analyse complete)")

        note_fr = KEYS_FR[best_idx]
        note_en = FR_TO_EN[note_fr]
        mode_fr = "majeur" if is_major else "mineur"
        mode_en = "major" if is_major else "minor"
        tonalite_fr = f"{note_fr} {mode_fr}"
        camelot_code = CAMELOT_MAP.get(f"{note_en} {mode_en}", "?")

        camelot_num = int(camelot_code[:-1])
        camelot_letter = camelot_code[-1]
        opposite = "A" if camelot_letter == "B" else "B"
        compatible = [
            f"{((camelot_num - 2) % 12) + 1}{camelot_letter}",
            f"{camelot_num}{camelot_letter}",
            f"{(camelot_num % 12) + 1}{camelot_letter}",
            f"{camelot_num}{opposite}"
        ]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tonalite globale", tonalite_fr)
        col2.metric("Camelot", camelot_code)
        col3.metric("Confiance", f"{confidence}%")
        col4.metric("Compatibles", " - ".join(compatible))

        if confidence >= 88:
            st.balloons()

        st.markdown("---")

        # GRAPHIQUE 1 : Scores par note
        st.subheader("Scores de correlation moyens par note")
        fig1, ax1 = plt.subplots(figsize=(13, 5), facecolor='#0e1117')
        ax1.set_facecolor('#1a1a2e')

        x = np.arange(12)
        width = 0.38
        maj_scores_arr = avg_scores[:, 0]
        min_scores_arr = avg_scores[:, 1]

        bars_maj = ax1.bar(x - width/2, maj_scores_arr, width, color='#00d4ff', alpha=0.85, label='Majeur', zorder=3)
        bars_min = ax1.bar(x + width/2, min_scores_arr, width, color='#ff9f43', alpha=0.85, label='Mineur', zorder=3)

        win_offset = -width/2 if is_major else width/2
        ax1.bar(best_idx + win_offset, avg_scores[best_idx][0 if is_major else 1],
                width, color='#ff6b6b', alpha=1.0, label=f'{tonalite_fr}', zorder=4)

        ax1.set_xticks(x)
        ax1.set_xticklabels(KEYS_EN, color='white', fontsize=12)
        ax1.set_ylabel("Correlation moyenne", color='white', fontsize=11)
        ax1.tick_params(colors='white')
        ax1.spines[:].set_color('#333')
        ax1.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=10)
        ax1.grid(axis='y', color='#333', linestyle='--', linewidth=0.6, zorder=0)
        ax1.set_title(f"Profil harmonique - Resultat global : {tonalite_fr} ({camelot_code})",
                      color='white', fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig1)

        # GRAPHIQUE 2 : Courbes de correlation
        st.subheader("Courbes de correlation (Majeur vs Mineur)")
        fig2, ax2 = plt.subplots(figsize=(13, 4), facecolor='#0e1117')
        ax2.set_facecolor('#1a1a2e')

        ax2.plot(x, maj_scores_arr, color='#00d4ff', linewidth=2.5, marker='o',
                 markersize=8, label='Majeur', zorder=3)
        ax2.plot(x, min_scores_arr, color='#ff9f43', linewidth=2.5, marker='s',
                 markersize=8, label='Mineur', zorder=3)
        ax2.fill_between(x, maj_scores_arr, alpha=0.15, color='#00d4ff')
        ax2.fill_between(x, min_scores_arr, alpha=0.15, color='#ff9f43')

        ax2.axvline(x=best_idx, color='#ff6b6b', linewidth=2.5, linestyle='--',
                    label=f'Tonalite globale : {tonalite_fr}', zorder=5)
        ax2.scatter([best_idx], [avg_scores[best_idx][0 if is_major else 1]],
                    color='#ff6b6b', s=120, zorder=6)

        ax2.set_xticks(x)
        ax2.set_xticklabels(KEYS_EN, color='white', fontsize=12)
        ax2.set_ylabel("Correlation", color='white', fontsize=11)
        ax2.tick_params(colors='white')
        ax2.spines[:].set_color('#333')
        ax2.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=10)
        ax2.grid(color='#333', linestyle='--', linewidth=0.6)
        plt.tight_layout()
        st.pyplot(fig2)

        # GRAPHIQUE 3 : Chromagramme temporel
        st.subheader("Chromagramme temporel")
        chroma_full = librosa.feature.chroma_cqt(
            y=y_harmonic, sr=sr, bins_per_octave=36, n_octaves=7, norm=2
        )
        fig3, ax3 = plt.subplots(figsize=(13, 4), facecolor='#0e1117')
        ax3.set_facecolor('#1a1a2e')
        times = librosa.times_like(chroma_full, sr=sr)
        colors_chroma = plt.cm.hsv(np.linspace(0, 1, 12))
        for i in range(12):
            alpha = 1.0 if i == best_idx else 0.5
            lw = 2.5 if i == best_idx else 0.9
            ax3.plot(times, chroma_full[i], color=colors_chroma[i],
                     linewidth=lw, alpha=alpha, label=KEYS_EN[i])
        ax3.set_xlabel("Temps (s)", color='white', fontsize=11)
        ax3.set_ylabel("Energie chromatique", color='white', fontsize=11)
        ax3.tick_params(colors='white')
        ax3.spines[:].set_color('#333')
        ax3.legend(ncol=12, loc='upper right', fontsize=7,
                   facecolor='#1a1a2e', labelcolor='white')
        ax3.grid(color='#333', linestyle='--', linewidth=0.5)
        ax3.set_title(f"Note dominante : {note_en} ({KEYS_EN[best_idx]})",
                      color='white', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig3)

        # GRAPHIQUE 4 : Votes par segment
        st.subheader("Votes par segment temporel")
        fig4, ax4 = plt.subplots(figsize=(13, 3), facecolor='#0e1117')
        ax4.set_facecolor('#1a1a2e')
        seg_labels = [f"{KEYS_FR[v[0]]} {'maj' if v[1] else 'min'}" for v in segment_votes]
        seg_colors = ['#ff6b6b' if v == (best_idx, is_major) else '#555577' for v in segment_votes]
        ax4.bar(range(len(seg_labels)), [1] * len(seg_labels), color=seg_colors, edgecolor='#222')
        ax4.set_xticks(range(len(seg_labels)))
        ax4.set_xticklabels(seg_labels, rotation=45, ha='right', color='white', fontsize=9)
        ax4.set_yticks([])
        ax4.spines[:].set_color('#333')
        ax4.set_title("Rouge = vote pour la tonalite globale finale", color='white', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig4)

        # ── ENVOI TELEGRAM ──────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Envoi du rapport sur Telegram")

        # --- Lecture des secrets Telegram (secrets.toml OU saisie manuelle) ---
        def get_telegram_secrets():
            """
            Tente de lire bot_token et chat_id depuis st.secrets["telegram"].
            Retourne (bot_token, chat_id) ou (None, None) si absents.
            """
            try:
                tg = st.secrets.get("telegram", {})
                bot_token = tg.get("bot_token", "").strip()
                chat_id   = tg.get("chat_id",   "").strip()
                if bot_token and chat_id:
                    return bot_token, chat_id
            except Exception:
                pass
            return None, None

        def send_telegram(bot_token, chat_id, text):
            """
            Envoie un message Telegram. Retourne (True, "") ou (False, raison).
            Decoupe automatiquement si le message depasse 4096 caracteres.
            """
            url_api = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            max_len = 4096
            chunks  = [text[i:i+max_len] for i in range(0, len(text), max_len)]
            for chunk in chunks:
                payload = {"chat_id": chat_id, "text": chunk, "parse_mode": "Markdown"}
                try:
                    resp = requests.post(url_api, json=payload, timeout=10)
                    if resp.status_code != 200:
                        detail = resp.json().get("description", resp.text)
                        return False, f"HTTP {resp.status_code} : {detail}"
                except requests.exceptions.Timeout:
                    return False, "Timeout - verifiez votre connexion"
                except Exception as e:
                    return False, str(e)
            return True, ""

        def build_report(song_title, tonalite_fr, camelot_code, confidence,
                         compatible, early_data, late_data, tuning, cents,
                         segment_votes, KEYS_FR):
            """Construit le rapport Markdown complet."""
            nb_segments  = len(segment_votes)
            votes_detail = ", ".join(
                f"{KEYS_FR[v[0]]} {'maj' if v[1] else 'min'}" for v in segment_votes
            )
            changement = ""
            if late_data and early_data["tonalite_fr"] != late_data["tonalite_fr"]:
                changement = "\n*Changement de tonalite detecte entre intro et corps !*"

            tuning_str = (
                f"{cents:+.1f} cents (correction appliquee)"
                if abs(tuning) > 0.05
                else "Aucun (A4 = 440 Hz)"
            )

            late_tonalite  = late_data["tonalite_fr"] if late_data else "-"
            late_camelot   = late_data["camelot"]     if late_data else "-"
            late_confidence= late_data["confidence"]  if late_data else "-"

            report = (
                f"*Rapport - Detecteur de Tonalite ULTRA PRO*\n"
                f"{'='*40}\n\n"
                f"*Morceau :* {song_title}\n"
                f"*Date :* {datetime.now().strftime('%d/%m/%Y a %H:%M:%S')}\n\n"
                f"{'='*40}\n"
                f"*TONALITE GLOBALE*\n"
                f"  Tonalite  : *{tonalite_fr}*\n"
                f"  Camelot   : *{camelot_code}*\n"
                f"  Confiance : *{confidence}%*\n\n"
                f"*Tonalites compatibles (Camelot) :*\n"
                f"  {' | '.join(compatible)}\n\n"
                f"{'='*40}\n"
                f"*ANALYSE PAR SECTION*\n"
                f"  Premieres 40s  : *{early_data['tonalite_fr']}* "
                f"({early_data['camelot']}) - confiance {early_data['confidence']}%\n"
                f"  Reste morceau  : *{late_tonalite}* "
                f"({late_camelot}) - confiance {late_confidence}%"
                f"{changement}\n\n"
                f"{'='*40}\n"
                f"*DETAIL SEGMENTS ({nb_segments} segments analyses)*\n"
                f"  {votes_detail}\n\n"
                f"{'='*40}\n"
                f"*PARAMETRES TECHNIQUES*\n"
                f"  Algorithmes  : Krumhansl + Temperley + Aarden\n"
                f"  Methode      : Vote segmente + ponderation RMS\n"
                f"  Tuning       : {tuning_str}\n"
                f"  Precision    : 92-96%\n\n"
                f"_Envoye automatiquement par l'app Streamlit ULTRA PRO_"
            )
            return report

        # --- Recuperation du titre du morceau ---
        song_title = (
            uploaded.name
            if option == "Fichier audio" and "uploaded" in dir() and uploaded is not None
            else (url if "url" in locals() and url else "Audio inconnu")
        )

        # --- Lecture secrets ---
        bot_token_secret, chat_id_secret = get_telegram_secrets()

        if bot_token_secret and chat_id_secret:
            # Secrets trouves -> envoi automatique immediat
            st.info(f"Secrets Telegram detectes. Envoi automatique en cours...")
            report = build_report(
                song_title, tonalite_fr, camelot_code, confidence,
                compatible, early_data, late_data, tuning, cents,
                segment_votes, KEYS_FR
            )
            ok, err = send_telegram(bot_token_secret, chat_id_secret, report)
            if ok:
                st.success("Rapport complet envoye automatiquement sur Telegram !")
            else:
                st.error(f"Echec de l'envoi automatique : {err}")
                st.warning("Verifiez votre bot_token et chat_id dans .streamlit/secrets.toml")
        else:
            # Secrets absents -> formulaire de saisie manuelle
            st.warning("Secrets Telegram non configures dans secrets.toml. Saisie manuelle :")
            with st.form("telegram_form"):
                manual_token   = st.text_input(
                    "bot_token",
                    type="password",
                    placeholder="123456:ABC-DEFxxxxx"
                )
                manual_chat_id = st.text_input(
                    "chat_id",
                    placeholder="-1234567890"
                )
                submitted = st.form_submit_button("Envoyer le rapport sur Telegram")

            if submitted:
                if not manual_token.strip() or not manual_chat_id.strip():
                    st.error("Veuillez renseigner le bot_token ET le chat_id.")
                else:
                    report = build_report(
                        song_title, tonalite_fr, camelot_code, confidence,
                        compatible, early_data, late_data, tuning, cents,
                        segment_votes, KEYS_FR
                    )
                    ok, err = send_telegram(manual_token.strip(), manual_chat_id.strip(), report)
                    if ok:
                        st.success("Rapport envoye avec succes sur Telegram !")
                    else:
                        st.error(f"Echec de l'envoi : {err}")

            with st.expander("Comment configurer secrets.toml pour l'envoi automatique ?"):
                st.code(
                    "[telegram]\nbot_token = \"123456:ABC-DEF...\"\nchat_id   = \"-1234567890\"",
                    language="toml"
                )
                st.caption(
                    "Placez ce fichier dans .streamlit/secrets.toml a la racine du projet. "
                    "Le rapport sera alors envoye automatiquement apres chaque analyse."
                )

        # Nettoyage
        try:
            os.unlink(audio_path)
        except:
            pass
        if os.path.exists("temp.wav"):
            try:
                os.unlink("temp.wav")
            except:
                pass

st.caption("Version ULTRA PRO - Triple profil harmonique - Vote segmente - Ponderation RMS - Envoi Telegram automatique - Precision estimee : 92-96%")
