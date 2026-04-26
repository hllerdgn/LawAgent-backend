import streamlit as st
import requests
import json

# Sayfa Konfigürasyonu
st.set_page_config(
    page_title="LawAgent AI - Hukuk Asistanı",
    page_icon="⚖️",
    layout="centered"
)

# Stil Ayarları (Daha güvenli ve temiz yapı)
st.markdown("""
    <style>
    /* Ana arka plan ve genel font */
    .stApp {
        background-color: #f5f7f9;
    }
    /* Buton tasarımı */
    div.stButton > button:first-child {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background-color: #0e1117;
        color: white;
        font-weight: bold;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #262730;
        color: #00d4ff;
    }
    /* Kaynak kutucukları tasarımı */
    .source-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #0e1117;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .source-title {
        color: #0e1117;
        font-weight: bold;
        font-size: 1.1em;
        margin-bottom: 5px;
    }
    .source-text {
        color: #444444;
        font-style: italic;
        font-size: 0.95em;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("⚖️ LawAgent AI")
st.subheader("TÜBİTAK 2209/A Akıllı Hukuk Asistanı")

# Yan Panel (Ayarlar)
with st.sidebar:
    st.header("⚙️ Ayarlar")
    k_value = st.slider("Getirilecek Madde Sayısı (k)", min_value=1, max_value=15, value=7)
    st.divider()
    st.info("Bu sistem Türk Borçlar (TBK), Ticaret (TTK) ve Tüketici (TKHK) Hukuku üzerine uzmanlaşmıştır.")

# Ana Ekran - Sorgu Girişi
query = st.text_area("Hukuki sorunuzu yazınız:", 
                    placeholder="Örn: Cayma hakkı süresi kaç gündür?", 
                    height=120)

if st.button("Hukuki Analiz Yap"):
    if not query.strip():
        st.warning("⚠️ Lütfen hukuki bir soru giriniz.")
    else:
        with st.spinner("🔍 Hukuki maddeler taranıyor ve yanıt üretiliyor..."):
            try:
                # Backend API'ye istek (127.0.0.1 bazen localhost'tan daha stabil çalışır)
                response = requests.post(
                    "http://127.0.0.1:8000/ask",
                    json={"query": query, "k": k_value},
                    timeout=45
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # 1. Hukuki Değerlendirme
                    st.markdown("### 📝 Hukuki Değerlendirme")
                    st.info(data["answer"])
                    
                    # İşlem Detayları
                    st.caption(f"⏱️ İşlem Süresi: {data.get('sure_ms', 0)} ms | 🤖 Model: Llama-3.3-70b")

                    # 2. Dayanak Maddeler
                    if data.get("sources"):
                        st.markdown("### 📚 Dayanak Maddeler")
                        for src in data["sources"]:
                            # HTML formatında şık kutucuklar
                            st.markdown(f"""
                                <div class="source-box">
                                    <div class="source-title">{src['kanun']} Madde {src['madde']}</div>
                                    <div class="source-text">{src['ozet']}</div>
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    st.error(f"❌ API Hatası: {response.status_code}")
            
            except requests.exceptions.ConnectionError:
                st.error("❌ Backend API'ye bağlanılamadı! Lütfen generator.py dosyasının --api moduyla çalıştığından emin olun.")
            except Exception as e:
                st.error(f"⚠️ Bir hata oluştu: {str(e)}")

st.divider()
st.caption("© 2026 LawAgent AI - Akademik Proje Kapsamında Geliştirilmiştir.")