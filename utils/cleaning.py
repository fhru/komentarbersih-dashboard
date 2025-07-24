import re
import emoji
import unicodedata
from datasets import load_dataset
from typing import Dict, List
from functools import lru_cache
import time

# Global variable untuk caching dictionary
_slang_dict = None
_dict_loaded = False

def normalisasi_unicode(text: str) -> str:
    """
    Menormalisasi karakter unicode menjadi ASCII.
    
    Args:
        text (str): Teks input dengan karakter unicode
        
    Returns:
        str: Teks yang sudah dinormalisasi dengan karakter ASCII
    """
    if not isinstance(text, str):
        text = str(text)
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

def clean_text(text: str) -> str:
    """
    Membersihkan teks dari karakter yang tidak diinginkan
    
    Args:
        text (str): Teks input yang akan dibersihkan
        
    Returns:
        str: Teks yang sudah dibersihkan
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Konversi ke huruf kecil dan handle emoji
    text = text.lower()
    text = emoji.demojize(text)
    
    # Transformasi untuk membersihkan teks
    transformasi = [
        (r':[a-zA-Z_]+:', ' '),      # Hapus nama emoji
        (r'http\S+', ''),             # Hapus URL
        (r'@\w+|#\w+', ''),          # Hapus mention dan hashtag
        (r'[^\w\s]', ' '),           # Hapus simbol, pertahankan alfanumerik
        (r'\b\d+\b', ' '),           # Hapus angka berdiri sendiri
        (r'\s+', ' '),               # Hapus spasi berlebih
        (r'(\w)\1{2,}', r'\1'),      # Normalisasi karakter berulang (aaa -> a)
    ]
    
    # Terapkan transformasi
    for pola, pengganti in transformasi:
        text = re.sub(pola, pengganti, text)
    
    return text.strip()

def combine_character(text: str) -> str:
    """
    Menggabungkan huruf yang terpisah (contoh: p r o m o -> promo).
    
    Args:
        text (str): Teks input dengan huruf terpisah
        
    Returns:
        str: Teks dengan huruf yang sudah digabungkan
    """
    if not isinstance(text, str):
        return str(text)
    
    pola = r'\b(?:[a-zA-Z]\s){2,}[a-zA-Z]\b'
    matches = re.findall(pola, text)
    
    for match in matches:
        gabungan = match.replace(' ', '')
        text = text.replace(match, gabungan)
    
    return text

def init_dictionary():
    """
    Inisialisasi dictionary slang saat aplikasi pertama kali dijalankan.
    Dictionary hanya dimuat sekali dan disimpan dalam cache.
    """
    global _slang_dict, _dict_loaded
    
    if _dict_loaded:
        print("Dictionary sudah dimuat sebelumnya")
        return True
    
    print("Memuat dictionary slang...")
    start_time = time.time()
    
    try:
        _slang_dict = load_slang_dict()
        load_time = time.time() - start_time
        print(f"Dictionary berhasil dimuat dalam {load_time:.2f} detik")
        _dict_loaded = True
        return True
        
    except Exception as e:
        print(f"Error saat memuat dictionary: {e}")
        return False

@lru_cache(maxsize=1)
def load_slang_dict() -> Dict[str, str]:
    """
    Memuat kamus slang dari definisi manual dan dataset Hugging Face.
    Menggunakan caching untuk menghindari pemuatan dataset berulang.
    
    Returns:
        Dict[str, str]: Kamus yang memetakan kata slang ke kata formal
    """
    kamus_manual = {
        "gk": "tidak", "gak": "tidak", "g": "tidak", "tdk": "tidak", "ga": "tidak",
        "nggak": "tidak", "enggak": "tidak", "gpp": "tidak apa-apa",
        "gakpapa": "tidak apa-apa", "tp": "tapi", "tapi": "tetapi",
        "kl": "kalau", "klw": "kalau", "kalo": "kalau", "klo": "kalau",
        "krn": "karena", "karena": "sebab", "jd": "jadi", "sdh": "sudah",
        "udh": "sudah", "udah": "sudah", "dl": "dulu",
        "sm": "sama", "sama": "dengan", "dg": "dengan", "dr": "dari",
        "utk": "untuk", "yg": "yang", "jg": "juga", "d": "di",
        "dll": "dan lain-lain", "dst": "dan seterusnya", "ttp": "tetap",
        "tsb": "tersebut", "dlm": "dalam", "pdhl": "padahal",
        "mrk": "mereka", "sy": "saya", "gw": "saya", "gue": "saya",
        "gua": "saya", "w": "saya", "gwe": "saya", "km": "kamu",
        "lu": "kamu", "lo": "kamu", "q": "aku", "ak": "aku",
        "aq": "aku", "elo": "kamu", "elu": "kamu", "loe": "kamu",
        "mnrt": "menurut", "spt": "seperti", "bener": "benar", "kok": "mengapa",
        "lg": "lagi", "bgt": "banget", "banget": "sekali", "cm": "cuma",
        "cuman": "cuma", "emg": "memang", "emng": "memang",
        "bs": "bisa", "bsa": "bisa", "sabi": "bisa",
        "bikin": "membuat", "ksih": "kasih", "ksh": "kasih",
        "jgn": "jangan", "jngn": "jangan",
        "biar": "agar", "supaya": "agar",
        "anjay": "astaga", "anjir": "astaga", "anjrit": "astaga",
        "wkwk": "haha", "wkwkwk": "haha", "wk": "haha", "lol": "haha",
        "ngakak": "tertawa", "santuy": "santai", "woles": "santai", "mager": "malas",
        "gabut": "tidak ada kerjaan", "baper": "terbawa perasaan",
        "kepo": "penasaran", "julid": "iri", "gibah": "bergosip",
        "panik": "takut", "cape": "capek", "capekkk": "capek",
        "pusinggg": "pusing", "skuy": "ayo", "gas": "ayo", "gaskeun": "ayo",
        "mantul": "bagus", "uhuy": "mantap", "mantab": "mantap",
        "kocak": "lucu", "ngeri": "hebat", "goks": "hebat",
        "pecah": "seru", "smg": "semoga",
        "receh": "tidak penting", "lebay": "berlebihan", "php": "pemberi harapan palsu",
        "auto": "langsung", "halu": "berkhayal",
        "ngab": "teman", "cuy": "teman", "ngabers": "remaja pria",
        "bro": "saudara", "sis": "kakak", "tmn": "teman",
        "tmn2": "teman-teman", "bocil": "anak kecil", "org": "orang",
        "bang": "kakak", "bg": "kakak", "bng": "kakak", "kak": "kakak",
        "min": "minimal", "jp": "jackpot", "jepe": "jackpot", "jepey": "jackpot",
        "bonus": "hadiah", "depo": "deposit", "wd": "withdraw",
        "bet": "banget", "gmpng": "mudah", "gampang": "mudah",
        "win": "menang", "betting": "taruhan",
        "slot": "permainan judi", "event": "acara", "promo": "promosi",
        "gacr": "gacor", "gcr": "gacor",
        "mekswin": "maxwin", "gacir": "gacor", "y": "ya", "kn": "kan", "cs" : "dan kawan kawan",
        "dri": "dari", "msk": "masuk", "thn": "tahun", "th": "tahun", "korup": "korupsi",
        "ortu": "orang tua", "jekpot": "jackpot", "ny": "nya", "mmg": "memang", "klihatan": "terlihat",
        "keliatan": "terlihat", "demen": "suka", "kayak": "seperti", "dah": "sudah", "knp": "kenapa",
        "wtf": "astaga", "sosmed": "sosial media", "gaspol": "ayo", "maen": "main",
        "judol": "judi online", "smpe": "sampai", "sampe": "sampai", "nyampe": "sampai",
        "pinjol": "pinjaman online", "ntar": "nanti", "nnti": "nanti", "nti": "nanti",
        "gini": "seperti ini", "gni": "seperti ini", "begini": "seperti ini", "bpk": "bapak",
        "bp": "bapak", "tilep": "mengambil", "mirip": "seperti", "mrp": "seperti", "drpd": "daripada",
        "thdp": "terhadap", "jga": "juga", "mngkin": "mungkin", "ap": "apa", "bkl": "akan",
        "bakal": "akan", "mna": "dimana", "mn": "dimana", "mana": "dimana", "cilik": "kecil",
        "pny": "punya", "wong": "orang", "msh": "masih", "sj": "saja", "pk": "bapak", "dn": "dan",
        "plis": "tolong", "hati2": "hati hati", "tlpn": "telepon", "tlp": "telepon",
        "ngakak": "tertawa", "jir": "astaga"
    }
    
    try:
        # load dataset dari hugging face
        dataset = load_dataset("zeroix07/indo-slang-words", split="train")
        
        # Ekstrak pasangan slang-formal
        hf_slang = []
        for row in dataset:
            if "text" in row and ":" in row["text"]:
                teks = row["text"].strip()
                if ":" in teks:
                    hf_slang.append(teks)
        
        # Parse slang pairs with validation
        kamus_hf = {}
        for pair in hf_slang:
            if ":" in pair:
                parts = pair.split(":", 1)
                if len(parts) == 2:
                    slang, formal = parts[0].strip(), parts[1].strip()
                    if slang and formal:  # Ensure not empty
                        kamus_hf[slang] = formal
        
        # Gabungkan dengan kamus manual
        kamus_gabungan = {**kamus_hf, **kamus_manual}
        print(f"Berhasil memuat {len(kamus_gabungan)} kata slang ({len(kamus_hf)} dari HF, {len(kamus_manual)} manual)")
        
        return kamus_gabungan
    
    except Exception as e:
        print(f"Error saat memuat kamus slang dari Hugging Face: {e}")
        return kamus_manual

def replace_slang(text: str, kamus_slang: Dict[str, str] = None) -> str:
    """
    Mengganti kata slang
    
    Args:
        text (str): Teks input yang berisi kata slang
        kamus_slang (Dict[str, str]): Kamus slang. Jika None, akan memuat kamus default.
        
    Returns:
        str: Teks dengan kata slang yang sudah diganti
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Pastikan dictionary sudah dimuat
    if not _dict_loaded:
        if not init_dictionary():
            return text
    
    if kamus_slang is None:
        kamus_slang = _slang_dict
    
    # Pisahkan teks menjadi kata-kata dan ganti setiap kata
    words = text.split()
    replaced_words = []
    
    for word in words:
        # Cek kecocokan
        if word in kamus_slang:
            replaced_words.append(kamus_slang[word])
        else:
            # Cek kata dengan tanda baca
            cleaned_word = re.sub(r'[^\w]', '', word)
            if cleaned_word in kamus_slang:
                # Pertahankan tanda baca asli
                tanda_baca = word[len(cleaned_word):]
                replaced_words.append(kamus_slang[cleaned_word] + tanda_baca)
            else:
                replaced_words.append(word)
    
    return " ".join(replaced_words)

def preprocess_teks_batch(texts: List[str]) -> List[str]:
    """
    Pipeline lengkap untuk preprocessing batch teks.
    Optimasi: gunakan list comprehension dan pandas jika tersedia.
    """
    if not texts:
        return []
    
    # Pastikan dictionary sudah dimuat
    if not _dict_loaded:
        if not init_dictionary():
            return [str(text) if text else "" for text in texts]
    
    # Optimasi dengan list comprehension
    return [preprocess_teks(text) for text in texts]

def preprocess_teks(text: str) -> str:
    """
    Pipeline lengkap untuk preprocessing teks.
    
    Args:
        text (str): Teks input yang akan dipreprocess
        
    Returns:
        str: Teks yang sudah dipreprocess
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Normalisasi karakter unicode
    text = normalisasi_unicode(text)
    
    # Bersihkan teks
    text = clean_text(text)
    
    # Gabungkan huruf terpisah
    text = combine_character(text)
    
    # Ganti kata slang
    text = replace_slang(text)
    
    return text.strip()

# Contoh penggunaan dan testing
if __name__ == "__main__":
    # Muat dictionary slang di awal
    init_dictionary()
    # Test pipeline preprocessing
    teks_test = "gk tau nih 😂 p r o m o judol skrg lg gacor bgt!"
    print(f"Teks asli: {teks_test}")
    hasil = preprocess_teks(teks_test)
    print(f"Teks hasil: {hasil}")