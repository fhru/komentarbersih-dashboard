import streamlit as st
import pandas as pd
import logging
import time
from typing import List, Dict
import os
import plotly.express as px

# Import utils
from utils.cleaning import preprocess_teks, preprocess_teks_batch, init_dictionary
from utils.predictor import predict_text, predict_batch, load_model, get_model_info
from utils.scraper import scrape_youtube_comments, get_comment_texts

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="KomentarBersih",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def init_model():
    """
    Inisialisasi model saat aplikasi pertama kali dijalankan.
    Menggunakan cache_resource agar model hanya dimuat sekali.
    """
    logger.info("Memulai inisialisasi model...")
    try:
        # Load model
        load_model()
        logger.info("✅ Model berhasil dimuat!")
        return True
    except Exception as e:
        logger.error(f"❌ Error saat memuat model: {e}")
        return False

@st.cache_resource
def init_dictionary_cache():
    """
    Inisialisasi dictionary saat aplikasi pertama kali dijalankan.
    Menggunakan cache_resource agar dictionary hanya dimuat sekali.
    """
    logger.info("Memulai inisialisasi dictionary...")
    try:
        # Load dictionary
        init_dictionary()
        logger.info("✅ Dictionary berhasil dimuat!")
        return True
    except Exception as e:
        logger.error(f"❌ Error saat memuat dictionary: {e}")
        return False

def styled_metric(label, value, delta=None, color=None, icon=None):
    """
    Custom styled metric box menggunakan gradasi warna pada background.
    """
    # Map warna gradasi untuk setiap kategori
    gradient_map = {
        'primary': 'linear-gradient(1deg, #2E86C1 0%, #5DADE2 100%)',
        'success': 'linear-gradient(1deg, #28B463 0%, #82E0AA 100%)',
        'danger': 'linear-gradient(1deg, #CB4335 0%, #F1948A 100%)',
        'warning': 'linear-gradient(1deg, #F1C40F 0%, #F9E79F 100%)',
        'info': 'linear-gradient(1deg, #17A589 0%, #76D7C4 100%)',
        None: 'linear-gradient(1deg, #34495E 0%, #85929E 100%)'
    }
    box_gradient = gradient_map.get(color, gradient_map[None])
    icon_html = f'<span style="font-size:1.3em;">{icon}</span> ' if icon else ''
    st.markdown(f'''
        <div style="
            background: {box_gradient};
            padding:1em 1.5em;
            border-radius:1em;
            margin-bottom:0.5em;
            box-shadow:0 2px 8px #0001;
            display:flex;
            flex-direction:column;
            align-items:center;
        ">
            <div style="font-size:1.1em;color:#fff;opacity:0.85;">{icon_html}{label}</div>
            <div style="font-size:2.2em;font-weight:bold;color:#fff;">{value}</div>
            {f'<div style="color:#fff;opacity:0.7;font-size:1em;">{delta}</div>' if delta else ''}
        </div>
    ''', unsafe_allow_html=True)

def main():
    """Fungsi utama aplikasi"""
    
    # Header
    st.title("🛡️ KomentarBersih")
    st.markdown(
        """
        <div style="font-size:1.1em; margin-bottom: 0.5em;">
            Identifikasi komentar yang mengandung unsur <b>judi</b> secara otomatis menggunakan model IndoBERT.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")
    
    # Inisialisasi model dan dictionary saat aplikasi dimulai
    col1, col2 = st.columns(2)
    
    with col1:
        with st.spinner("🔄 Memuat model IndoBERT..."):
            model_loaded = init_model()
    
    with col2:
        with st.spinner("📚 Memuat dictionary slang..."):
            dict_loaded = init_dictionary_cache()
    
    if not model_loaded:
        st.error("❌ Gagal memuat model! Aplikasi tidak dapat berjalan.")
        st.stop()
    
    if not dict_loaded:
        st.warning("⚠️ Gagal memuat dictionary! Preprocessing mungkin tidak optimal.")
    
    # Sidebar untuk navigasi
    st.sidebar.title("Menu")
    page = st.sidebar.selectbox(
        "Pilih Fitur:",
        ["Single Komentar", "File CSV", "URL YouTube"]
    )
    
    if page == "Single Komentar":
        single_comment_page()
    elif page == "File CSV":
        csv_input_page()
    elif page == "URL YouTube":
        youtube_input_page()

def single_comment_page():
    """Halaman untuk input single komentar"""
    st.header("💬 Input Komentar")
    
    # Input text
    comment = st.text_area(
        "Masukkan komentar yang akan dianalisis:",
        height=150,
        placeholder="Contoh: Promo judi online terbaik, deposit 10rb dapat bonus 100rb!"
    )
    
    if st.button("🔍 Analisis Komentar", type="primary"):
        if comment.strip():
            with st.spinner("Sedang menganalisis..."):
                try:
                    # Logging
                    logger.info("Memulai analisis single komentar")
                    
                    # Cleaning
                    cleaned_comment = preprocess_teks(comment)
                    logger.info(f"Teks asli: {comment[:100]}...")
                    logger.info(f"Teks bersih: {cleaned_comment[:100]}...")
                    
                    # Prediction
                    result = predict_text(cleaned_comment)
                    
                    # Tampilkan hasil dengan metric box custom
                    st.success("✅ Analisis selesai!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        styled_metric("Label", result['label'], color='primary')
                    with col2:
                        styled_metric("Confidence", f"{result['confidence']:.2%}", color='info')
                    with col3:
                        styled_metric("Prediction", result['prediction'], color='success' if result['prediction']==0 else 'danger')
                    
                    # Detail analisis
                    with st.expander("📋 Detail Analisis"):
                        st.write("**Teks Asli:**")
                        st.write(comment)
                        st.write("**Teks Setelah Cleaning:**")
                        st.write(cleaned_comment)
                        st.write("**Hasil Prediksi:**")
                        st.json(result)
                    
                    logger.info(f"Analisis selesai - Label: {result['label']}, Confidence: {result['confidence']:.2%}")
                    
                except Exception as e:
                    st.error(f"❌ Error saat analisis: {e}")
                    logger.error(f"Error saat analisis single komentar: {e}")
        else:
            st.warning("⚠️ Masukkan komentar terlebih dahulu!")

def csv_input_page():
    """Halaman untuk input file CSV"""
    st.header("📊 Input File CSV")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Pilih file CSV:",
        type=['csv'],
        help="File CSV harus memiliki kolom 'komentar' atau 'comment'"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File berhasil diupload! ({len(df)} baris)")
            
            # Check column
            comment_column = None
            for col in ['komentar', 'comment', 'text', 'teks']:
                if col in df.columns:
                    comment_column = col
                    break
            
            if comment_column is None:
                st.error("❌ Kolom 'komentar' tidak ditemukan dalam file CSV!")
                st.write("Kolom yang tersedia:", list(df.columns))
                return
            
            st.info(f"📋 Menggunakan kolom: '{comment_column}'")
            
            # Preview data
            with st.expander("👀 Preview Data"):
                st.dataframe(df.head())
            
            # Analysis settings
            max_comments = st.slider(
                "Jumlah komentar yang dianalisis:",
                min_value=1,
                max_value=len(df),
                value=min(100, len(df))
            )
            
            if st.button("🔍 Analisis CSV", type="primary"):
                with st.spinner("Sedang menganalisis..."):
                    try:
                        # Logging
                        logger.info(f"Memulai analisis CSV - {max_comments} komentar")
                        
                        # Get comments
                        comments = df[comment_column].head(max_comments).tolist()
                        
                        # Progress bar untuk cleaning
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Cleaning dengan batch processing
                        status_text.text("🔄 Membersihkan teks...")
                        cleaned_comments = []
                        
                        # Process dalam batch untuk efisiensi
                        batch_size = 10  # Kurangi batch size untuk progress yang lebih halus
                        total_batches = (len(comments) + batch_size - 1) // batch_size
                        
                        for i in range(0, len(comments), batch_size):
                            batch = comments[i:i+batch_size]
                            # Convert to string and handle NaN
                            batch_texts = [str(comment) if pd.notna(comment) else "" for comment in batch]
                            cleaned_batch = preprocess_teks_batch(batch_texts)
                            cleaned_comments.extend(cleaned_batch)
                            
                            # Update progress dengan perhitungan yang lebih akurat
                            current_batch = (i // batch_size) + 1
                            progress = current_batch / total_batches
                            progress_bar.progress(progress)
                            
                            # Update status text
                            status_text.text(f"🔄 Membersihkan teks... ({current_batch}/{total_batches} batch)")
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ Cleaning selesai!")
                        
                        # Progress bar untuk prediction
                        status_text.text("🤖 Melakukan prediksi...")
                        progress_bar.progress(0)
                        
                        # Prediction dengan batch processing
                        results = predict_batch(cleaned_comments)
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ Prediksi selesai!")
                        
                        # Create results dataframe
                        results_df = pd.DataFrame({
                            'Teks Asli': comments,
                            'Teks Bersih': cleaned_comments,
                            'Label': [r['label'] for r in results],
                            'Confidence': [f"{r['confidence']:.2%}" for r in results],
                            'Prediction': [r['prediction'] for r in results]
                        })
                        
                        # Display results
                        st.success("✅ Analisis selesai!")
                        
                        # Statistics
                        judi_count = sum(1 for r in results if r['prediction'] == 1)
                        normal_count = sum(1 for r in results if r['prediction'] == 0)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            styled_metric("Total Komentar", len(results), color='primary', icon='💬')
                        with col2:
                            styled_metric("Komentar Judi", judi_count, color='danger', icon='🎰')
                        with col3:
                            styled_metric("Komentar Normal", normal_count, color='success', icon='✅')
                        
                        # Chart visualisasi
                        chart_df = pd.DataFrame({
                            'Kategori': ['Judi', 'Normal'],
                            'Jumlah': [judi_count, normal_count]
                        })
                        st.markdown("### Visualisasi Hasil")
                        col_chart1, col_chart2 = st.columns(2)
                        with col_chart1:
                            fig_pie = px.pie(chart_df, names='Kategori', values='Jumlah', title='Distribusi Komentar')
                            st.plotly_chart(fig_pie, use_container_width=True)
                        with col_chart2:
                            fig_bar = px.bar(chart_df, x='Kategori', y='Jumlah', color='Kategori', title='Jumlah Komentar per Kategori', text='Jumlah')
                            st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # Results table
                        st.subheader("📋 Hasil Analisis")
                        st.dataframe(results_df)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Hasil CSV",
                            data=csv,
                            file_name="hasil_analisis_komentar.csv",
                            mime="text/csv"
                        )
                        
                        logger.info(f"Analisis CSV selesai - Judi: {judi_count}, Normal: {normal_count}")
                        
                    except Exception as e:
                        st.error(f"❌ Error saat analisis: {e}")
                        logger.error(f"Error saat analisis CSV: {e}")
        except Exception as e:
            st.error(f"❌ Error saat membaca file: {e}")
            logger.error(f"Error saat membaca file CSV: {e}")

def youtube_input_page():
    """Halaman untuk input URL YouTube"""
    st.header("🎥 Input URL YouTube")
    
    # Input URL
    youtube_url = st.text_input(
        "Masukkan URL video YouTube:",
        placeholder="https://www.youtube.com/watch?v=..."
    )
    
    # Settings
    col1, col2 = st.columns(2)
    with col1:
        max_comments = st.number_input(
            "Jumlah komentar maksimal:",
            min_value=1,
            max_value=1000,
            value=50
        )
    
    with col2:
        threshold = st.slider(
            "Threshold confidence:",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.1
        )
    
    if st.button("🔍 Analisis Komentar YouTube", type="primary"):
        if youtube_url.strip():
            with st.spinner("Sedang mengambil komentar dari YouTube..."):
                try:
                    # Logging
                    logger.info(f"Memulai scraping YouTube: {youtube_url}")
                    
                    # Scrape comments
                    result = scrape_youtube_comments(youtube_url, max_comments)
                    
                    if result['success']:
                        st.toast("✅ Berhasil mengambil komentar dari YouTube!")
                        
                        # Video info
                        video_info = result['video_info']
                        st.subheader("📺 Informasi Video")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Judul:** {video_info['title']}")
                            st.write(f"**Channel:** {video_info['channel']}")
                        with col2:
                            st.write(f"**Views:** {video_info['view_count']}")
                            st.write(f"**Komentar:** {video_info['comment_count']}")
                        
                        # Get comment texts
                        comment_texts = [comment['text'] for comment in result['comments']]
                        
                        # Progress bar untuk cleaning
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Cleaning dengan batch processing
                        status_text.text("🔄 Membersihkan teks...")
                        cleaned_comments = []
                        
                        # Process dalam batch untuk efisiensi
                        batch_size = 10  # Kurangi batch size untuk progress yang lebih halus
                        total_batches = (len(comment_texts) + batch_size - 1) // batch_size
                        
                        for i in range(0, len(comment_texts), batch_size):
                            batch = comment_texts[i:i+batch_size]
                            cleaned_batch = preprocess_teks_batch(batch)
                            cleaned_comments.extend(cleaned_batch)
                            
                            # Update progress dengan perhitungan yang lebih akurat
                            current_batch = (i // batch_size) + 1
                            progress = current_batch / total_batches
                            progress_bar.progress(progress)
                            
                            # Update status text
                            status_text.text(f"🔄 Membersihkan teks... ({current_batch}/{total_batches} batch)")
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ Cleaning selesai!")
                        
                        # Progress bar untuk prediction
                        status_text.text("🤖 Melakukan prediksi...")
                        progress_bar.progress(0)
                        
                        # Prediction dengan batch processing
                        results = predict_batch(cleaned_comments)
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ Prediksi selesai!")
                        
                        # Logging untuk debugging
                        for i, (original, cleaned, pred_result) in enumerate(zip(comment_texts, cleaned_comments, results)):
                            logger.info(f"Comment {i+1}: '{original[:30]}...' -> '{cleaned[:30]}...' -> {pred_result['prediction']} ({pred_result['confidence']:.2%})")
                        
                        # Filter by threshold
                        filtered_results = []
                        for i, result_pred in enumerate(results):
                            if result_pred['confidence'] >= threshold:
                                filtered_results.append({
                                    'index': i,
                                    'original_text': comment_texts[i],
                                    'cleaned_text': cleaned_comments[i],
                                    'label': result_pred['label'],
                                    'confidence': result_pred['confidence'],
                                    'prediction': result_pred['prediction']
                                })
                        
                        # Statistics
                        judi_count = sum(1 for r in results if r['prediction'] == 1)
                        normal_count = sum(1 for r in results if r['prediction'] == 0)
                        filtered_count = len(filtered_results)
                        
                        st.subheader("📊 Statistik Analisis")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            styled_metric("Total Komentar", len(results), color='primary')
                        with col2:
                            styled_metric("Komentar Judi", judi_count, color='danger')
                        with col3:
                            styled_metric("Komentar Normal", normal_count, color='success')
                        with col4:
                            styled_metric("Filtered (≥threshold)", filtered_count, color='info')
                        
                        # Chart visualisasi
                        chart_df = pd.DataFrame({
                            'Kategori': ['Judi', 'Normal'],
                            'Jumlah': [judi_count, normal_count]
                        })
                        st.markdown("### Visualisasi Hasil")
                        col_chart1, col_chart2 = st.columns(2)
                        with col_chart1:
                            fig_pie = px.pie(chart_df, names='Kategori', values='Jumlah', title='Distribusi Komentar')
                            st.plotly_chart(fig_pie, use_container_width=True)
                        with col_chart2:
                            fig_bar = px.bar(chart_df, x='Kategori', y='Jumlah', color='Kategori', title='Jumlah Komentar per Kategori', text='Jumlah')
                            st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # Results table
                        if filtered_results:
                            st.subheader("📋 Hasil Analisis (Filtered)")
                            results_df = pd.DataFrame(filtered_results)
                            st.dataframe(results_df)
                            
                            # Debug info
                            with st.expander("🔍 Debug Info (Semua Komentar)"):
                                debug_df = pd.DataFrame({
                                    'Original Text': comment_texts,
                                    'Cleaned Text': cleaned_comments,
                                    'Label': [r['label'] for r in results],
                                    'Confidence': [f"{r['confidence']:.2%}" for r in results],
                                    'Prediction': [r['prediction'] for r in results]
                                })
                                st.dataframe(debug_df)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Hasil CSV",
                                data=csv,
                                file_name="hasil_analisis_youtube.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("⚠️ Tidak ada komentar yang memenuhi threshold!")
                            
                            # Debug info jika tidak ada hasil
                            with st.expander("🔍 Debug Info (Semua Komentar)"):
                                debug_df = pd.DataFrame({
                                    'Original Text': comment_texts,
                                    'Cleaned Text': cleaned_comments,
                                    'Label': [r['label'] for r in results],
                                    'Confidence': [f"{r['confidence']:.2%}" for r in results],
                                    'Prediction': [r['prediction'] for r in results]
                                })
                                st.dataframe(debug_df)
                        
                        logger.info(f"Analisis YouTube selesai - Judi: {judi_count}, Normal: {normal_count}, Filtered: {filtered_count}")
                        
                    else:
                        st.error(f"❌ Error: {result['error']}")
                        logger.error(f"Error scraping YouTube: {result['error']}")
                        
                except Exception as e:
                    st.error(f"❌ Error saat analisis: {e}")
                    logger.error(f"Error saat analisis YouTube: {e}")
        else:
            st.warning("⚠️ Masukkan URL YouTube terlebih dahulu!")

if __name__ == "__main__":
    main()
