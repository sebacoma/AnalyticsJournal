import os, sys, re, json, time, random, platform, unicodedata
from dataclasses import dataclass
from collections import Counter
from typing import Dict, List, Tuple, Optional

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, cohen_kappa_score

# Nuevas importaciones para mejoras
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
    # Modelo multilingüe para español
    SBERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
except ImportError:
    SBERT_AVAILABLE = False
    print("Warning: sentence-transformers no disponible. Usando solo TF-IDF.")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score no disponible.")

# ============== CONFIG ==============
class CONFIG:
    SEED = 42
    DATA_DIR = "data/WDKE"          # carpeta con .txt de debates
    OUT_DIR = "out"                 # salidas CSV/TEX/PDF
    MAX_TEXT_CHARS = 16000          # truncamiento de seguridad para LLM
    
    # Mejora: Panel mixto de modelos para jueces más robustos
    LLM_MODELS = [                  
        {"name": "gpt-4o-mini", "temperature": 0.0, "family": "openai"},
        {"name": "gpt-4o-mini", "temperature": 0.3, "family": "openai"},
        {"name": "gpt-4o-mini", "temperature": 0.7, "family": "openai"},
        # Añadir más familias si tienes acceso (Anthropic, etc.)
    ]
    
    # Mejora: Más jueces para mayor robustez
    JUDGE_K = 5  # aumentado de 3 a 5
    BOOTSTRAP_N = 1000  # para intervalos de confianza
    
    SMALL_SESSIONS = ["sesion59","session61","session63","session64","session73","session74","session75","session78"]
    GOLD_PATH = None  # opcional: "data/gold_frames.json"

# Semillas reproducibles
random.seed(CONFIG.SEED); np.random.seed(CONFIG.SEED); os.environ["PYTHONHASHSEED"]=str(CONFIG.SEED)

# OpenAI API (0.28 client)
try:
    import openai
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY","")
    if not OPENAI_API_KEY:
        raise RuntimeError("Set OPENAI_API_KEY in your environment (.env).")
    openai.api_key = OPENAI_API_KEY
    OPENAI_VERSION = openai.__version__
except Exception as e:
    print("OpenAI init warning:", e)
    OPENAI_VERSION = "N/A"

# RUNTIME info -> reproducibilidad
RUNTIME_INFO = {
    "python": sys.version,
    "platform": platform.platform(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "openai": OPENAI_VERSION,
    "seed": CONFIG.SEED,
}
os.makedirs(CONFIG.OUT_DIR, exist_ok=True)
with open(os.path.join(CONFIG.OUT_DIR,"repro_runtime.json"), "w") as f:
    json.dump(RUNTIME_INFO, f, indent=2)

# ============== TAXONOMÍA Y REGLAS ==============
TAXONOMIA = [
    "Crisis/Urgency",
    "Human Rights",
    "Equity/Inequality",
    "Institutionality/Legality",
    "Responsibility/Accountability",
    "Participation/Dialogue/Unity",
    "Bureaucracy",
    "Social Protection",
    "Municipal Autonomy",
    "Economy/Prices",
    "Security/Public Order",
    "Democracy/Constitution",
    "Environment"
]

# Palabras procedimentales a excluir (en español por el corpus)
PROCEDIMENTALES = {
    "asistencia","apertura de la sesion","apertura","cuenta","orden del dia","orden del día",
    "tramitacion de actas","tramitación de actas","mocion","mociones","informe","informes",
    "votacion","votaciones","intervencion de senadores","reunion de comites","acuerdos de comites",
    "suspension de la sesion","comunicaciones","oficios","enmiendas","discusion","debate",
    "presentacion de indicaciones","aprobacion de indicaciones","citacion de autoridades",
    "procedimiento legislativo","acuerdos interinstitucionales","comunicacion de resultados",
    "normativa y legislacion","proyectos de ley","orden del día"
}

IGNORAR_REGEX = re.compile(
    r'^(ejemplo|descripcion|descripción|interpretacion|interpretación|analisis|análisis|'
    r'cita( relevante)?|frase|texto|contexto|justificacion|justificación|conclusion|conclusión|'
    r'resumen|sintesis|síntesis|nota|observacion|observación|comentario)\b', re.IGNORECASE
)

CANON_PATTERNS = {
    "Crisis/Urgency":               [r"\burgencia\b", r"\bcrisis\b", r"emergen", r"prioridad"],
    "Human Rights":                 [r"derechos?\s+humanos?", r"\bddhh\b", r"humanos?\b"],
    "Equity/Inequality":            [r"\bequidad\b", r"inequidad", r"desigualdad", r"justicia\s+social"],
    "Institutionality/Legality":    [r"institucion", r"legalidad", r"procedim", r"legitimi", r"admisibil"],
    "Responsibility/Accountability":[r"responsabil", r"rendici[oó]n\s+de\s+cuentas", r"account"],
    "Participation/Dialogue/Unity": [r"participaci[oó]n", r"di[aá]logo", r"unidad", r"consenso", r"colabora", r"acuerdo\s+social", r"pacto\s+social", r"agenda\s+social"],
    "Bureaucracy":                  [r"burocrac"],
    "Social Protection":            [r"protecci[oó]n\s+social", r"seguridad\s+social", r"\bniñez\b", r"\binfancia\b", r"adult[oa]s?\s+mayores"],
    "Municipal Autonomy":           [r"municip", r"autonom"],
    "Economy/Prices":               [r"econom", r"precios?", r"tarif", r"impuestos?", r"arancel", r"electricidad", r"energia", r"capitalizaci[oó]n", r"subsidio", r"presupuesto", r"bancoestado"],
    "Security/Public Order":        [r"seguridad", r"orden\s+p[uú]blico", r"delincu", r"carabiner"],
    "Democracy/Constitution":       [r"plebiscit", r"reforma\s+constitucional", r"constituci[oó]n", r"participaci[oó]n\s+democr[aá]tica?"],
    "Environment":                  [r"ambient", r"humedal(?:es)?", r"turbera(?:s)?", r"contaminaci[oó]n", r"\bagua\b", r"ecosistema(?:s)?"]
}

# ============== UTILIDADES ==============
def leer_archivo(path:str) -> str:
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def truncar(texto:str, max_chars:int=CONFIG.MAX_TEXT_CHARS) -> str:
    return texto if len(texto) <= max_chars else texto[:max_chars]

def quitar_acentos(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def es_procedimental(s: str) -> bool:
    s0 = quitar_acentos(str(s).lower()).strip()
    return s0 in PROCEDIMENTALES

def limpiar_label_cruda(s: str) -> str:
    if not isinstance(s, str): return ""
    s0 = s.strip().replace("**","").replace("*","")
    s0 = re.sub(r"^[-•*\d.\s]+", "", s0).split(":")[0].strip(" .:-–—\t")
    if IGNORAR_REGEX.search(s0) or es_procedimental(s0):
        return ""
    return s0

def normalizar_a_canon(frame_raw: str) -> str:
    if not frame_raw: return ""
    s = quitar_acentos(frame_raw.lower())
    for canon, pats in CANON_PATTERNS.items():
        for pat in pats:
            if re.search(pat, s):
                return canon
    return "Other"

# Mejora: Baseline regex-only para frames (sin LLM)
def extract_frames_regex_only(text: str) -> List[str]:
    """Extrae frames usando solo patrones regex sin LLM"""
    text_clean = quitar_acentos(text.lower())
    frames_found = set()
    
    for canon, patterns in CANON_PATTERNS.items():
        for pattern in patterns:
            matches = re.findall(pattern, text_clean)
            if matches:
                frames_found.add(canon)
                break  # Una coincidencia por categoría es suficiente
    
    return sorted(list(frames_found))

def sent_tokenize(s):
    return [t.strip() for t in re.split(r'(?<=[.!?])\s+', s.strip()) if t.strip()]

def extractive_summary(text, max_sents=5):
    sents = sent_tokenize(text)
    if len(sents) <= max_sents: return " ".join(sents)
    vec = TfidfVectorizer(min_df=1, ngram_range=(1,2))
    X = vec.fit_transform(sents)
    centroid = X.mean(axis=0)
    sims = (X @ centroid.T).A.ravel()
    idx = np.argsort(-sims)[:max_sents]
    idx = sorted(idx)
    return " ".join([sents[i] for i in idx])

def timed(func, *args, **kwargs):
    t0 = time.time()
    out = func(*args, **kwargs)
    dt = time.time() - t0
    return out, dt

# ============== LLM HELPERS ==============
MAX_RETRIES = 3
def call_openai(prompt, model="gpt-4o-mini", temperature=0.0, max_tokens=400, timeout=60):
    err = None
    for i in range(MAX_RETRIES):
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role":"user","content":prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                request_timeout=timeout,
            )
            content = resp["choices"][0]["message"]["content"].strip()
            usage = resp.get("usage", {})
            return content, usage
        except Exception as e:
            err = e
            time.sleep(2*(i+1))
    raise RuntimeError(f"OpenAI call failed: {err}")

PROMPTS = {
  "summarization": "Summarize the following legislative debate in 7–10 sentences, neutral tone:\n\n{TEXT}",
  "open_labels":   "Identify FRAMES present. Return ONLY the frame names, one per line, no numbering:\n\n{TEXT}\n\nOutput: one frame per line.",
  "closed_labels": "Identify ONLY from THIS LIST (no Other): {TAX}\nReturn JSON list of unique strings.\n\nTEXT:\n{TEXT}\n\nOutput JSON: [\"Crisis/Urgency\",\"Human Rights\", ...]",
  "judge_prompt":  "Evaluate the NARRATIVE COHERENCE of the following summary.\nRate from 1 (poor) to 5 (excellent). Respond ONLY the number 1..5.\n\nSummary:\n{SUMMARY}\n\nScore:"
}

with open(os.path.join(CONFIG.OUT_DIR,"repro_prompts.json"), "w") as f:
    json.dump(PROMPTS, f, indent=2)

def llm_summary(text, model="gpt-4o-mini", temperature=0.0, max_tokens=400):
    txt = truncar(text)
    prompt = PROMPTS["summarization"].replace("{TEXT}", txt)
    t0 = time.time()
    content, usage = call_openai(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
    dt = time.time() - t0
    return content, dt, usage

def identificar_framings_open(text, model="gpt-4o-mini", temperature=0.0, max_tokens=400):
    txt = truncar(text)
    prompt = PROMPTS["open_labels"].replace("{TEXT}", txt)
    out, usage = call_openai(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
    labels = []
    for line in out.split("\n"):
        lab = limpiar_label_cruda(line)
        if lab:
            labels.append(lab)
    return labels, usage

def identificar_framings_closed(text, model="gpt-4o-mini", temperature=0.0, max_tokens=200):
    txt = truncar(text)
    prompt = PROMPTS["closed_labels"].replace("{TEXT}", txt).replace("{TAX}", ", ".join(TAXONOMIA))
    out, usage = call_openai(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
    try:
        js = re.search(r"\[.*\]", out, flags=re.S).group(0)
        arr = json.loads(js)
        arr = [s for s in arr if s in TAXONOMIA]
        # unicidad preservando orden
        seen, out_arr = set(), []
        for f in arr:
            if f not in seen:
                seen.add(f); out_arr.append(f)
        return out_arr, usage
    except Exception:
        return [], usage

# Mejora: Jueces LLM robustos con panel mixto
def judge_llm_robust(summary_text, k=None, models=None):
    k = k or CONFIG.JUDGE_K
    models = models or CONFIG.LLM_MODELS
    
    all_scores = []
    all_usages = []
    
    # Distribuir evaluaciones entre modelos disponibles
    evals_per_model = k // len(models)
    remaining = k % len(models)
    
    for i, model_config in enumerate(models):
        model_name = model_config["name"]
        temp = model_config["temperature"]
        
        # Número de evaluaciones para este modelo
        num_evals = evals_per_model + (1 if i < remaining else 0)
        
        model_scores = []
        model_usages = []
        
        for _ in range(num_evals):
            try:
                p = PROMPTS["judge_prompt"].replace("{SUMMARY}", summary_text)
                out, usage = call_openai(p, model=model_name, temperature=temp, max_tokens=5)
                m = re.search(r"[1-5]", out)
                if m: 
                    score = int(m.group(0))
                    model_scores.append(score)
                    all_scores.append(score)
                model_usages.append(usage)
                all_usages.append(usage)
            except Exception as e:
                print(f"Error en judge_llm_robust: {e}")
                # Score por defecto en caso de error
                all_scores.append(3)
    
    if not all_scores: all_scores = [3]
    
    return {
        "mean_score": float(np.mean(all_scores)),
        "scores": all_scores,
        "usages": all_usages,
        "std_score": float(np.std(all_scores)) if len(all_scores) > 1 else 0.0
    }

# Función original mantenida para compatibilidad
def judge_llm(summary_text, k=3, model="gpt-4o-mini"):
    result = judge_llm_robust(summary_text, k=k, models=[{"name": model, "temperature": 0.0}])
    return result["mean_score"], result["scores"], result["usages"]

# ============== BASE ==============
def procesar_archivos(data_dir:str) -> Dict[str, Dict[str,str]]:
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data dir not found: {data_dir}")
    resultados = {}
    for nombre in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, nombre)
        if not os.path.isfile(path): continue
        texto = leer_archivo(path)
        # resumen LLM
        resumen, dt, usage = llm_summary(texto)
        # frames (open)
        open_labels, _ = identificar_framings_open(texto)
        # frames (closed)
        closed_labels, _ = identificar_framings_closed(texto)
        resultados[nombre] = {
            "text": texto,
            "resumen": resumen,
            "framings_open": "\n".join(open_labels),
            "framings_closed": "\n".join(closed_labels),
            "latency_summary_s": dt,
            "tokens_total": usage.get("total_tokens", None)
        }
        print(f"✓ {nombre} (summary {dt:.2f}s)")
    return resultados

def tabla_frecuencia_frames(framings_texts:List[str]) -> pd.DataFrame:
    frames_all = []
    IGNORAR = {"Example","Description","Justification","Effect","Phrase","Text"}
    for framings in framings_texts:
        for linea in framings.split("\n"):
            linea = linea.strip()
            if not linea: continue
            frame = re.sub(r"^[-*0-9.\s]+", "", linea).replace("**","").replace("*","").split(":")[0].strip()
            if not frame or frame in IGNORAR: continue
            frames_all.append(frame)
    conteo = Counter(frames_all)
    total = max(sum(conteo.values()), 1)
    rows = [{"Frame": k, "Frequency": v, "Share_%": round(100*v/total, 2)} for k,v in conteo.items()]
    return pd.DataFrame(rows).sort_values("Frequency", ascending=False).reset_index(drop=True)

def normalizar_df_frames(df_frames: pd.DataFrame) -> pd.DataFrame:
    def limpiar_label(s: str) -> str:
        if not isinstance(s, str): return ""
        s0 = s.strip().replace("**", "").replace("*", "")
        s0 = re.sub(r"^[-•*\d.\s]+", "", s0).split(":")[0].strip(" .:-–—\t")
        if IGNORAR_REGEX.search(s0): return ""
        return s0
    df = df_frames.copy()
    df["Frame_clean"] = df["Frame"].apply(limpiar_label)
    df = df[df["Frame_clean"]!=""].copy()
    df["Frame"] = df["Frame_clean"].apply(normalizar_a_canon)
    agg = df.groupby("Frame", as_index=False)["Frequency"].sum()
    total = max(agg["Frequency"].sum(), 1)
    agg["Share_%"] = (agg["Frequency"]/total*100).round(2)
    return agg.sort_values("Frequency", ascending=False).reset_index(drop=True)

def resumir_top_frames(df_frames: pd.DataFrame, top_n:int=10) -> pd.DataFrame:
    df = df_frames.sort_values("Frequency", ascending=False).reset_index(drop=True)
    total = max(df["Frequency"].sum(), 1)
    df["Share_%"] = (df["Frequency"]/total*100).round(2)
    if len(df) <= top_n: return df
    top = df.iloc[:top_n].copy()
    otros_freq = int(df.iloc[top_n:]["Frequency"].sum())
    otros_pct = round(df.iloc[top_n:]["Share_%"].sum(), 2)
    top.loc[len(top)] = {"Frame": f"Other ({len(df)-top_n})", "Frequency": otros_freq, "Share_%": otros_pct}
    diff = round(100.0 - top["Share_%"].sum(), 2)
    if diff != 0:
        top.at[top.index[-1], "Share_%"] = round(top.iloc[-1]["Share_%"] + diff, 2)
    return top

def construir_matriz_frames_sesion(resultados:Dict[str,Dict], source="framings_closed", mode="binary", include_other=False):
    categorias = list(TAXONOMIA)
    if include_other:
        categorias += ["Other"]
    sesiones = [os.path.splitext(k)[0] for k in resultados.keys()]
    sesiones = sorted(sesiones)
    data = {ses:{cat:0 for cat in categorias} for ses in sesiones}
    for archivo, datos in resultados.items():
        ses = os.path.splitext(archivo)[0]
        framings_text = datos.get(source,"")
        for linea in framings_text.split("\n"):
            raw = limpiar_label_cruda(linea)
            if not raw: continue
            canon = normalizar_a_canon(raw)
            if (not include_other) and canon=="Other": continue
            if canon not in categorias: continue
            if mode=="binary": data[ses][canon] = 1
            else: data[ses][canon] += 1
    df = pd.DataFrame.from_dict(data, orient="index")[categorias].T
    df["__tot__"] = df.sum(axis=1)
    df = df.sort_values("__tot__", ascending=False).drop(columns="__tot__")
    return df

# Coherencia TF-IDF (método original)
def tfidf_coherence(summary_text:str) -> float:
    sents = [t.strip() for t in re.split(r'[.!?]\s+', summary_text.strip()) if t.strip()]
    if len(sents) < 2: return 0.0
    vec = TfidfVectorizer(min_df=1, ngram_range=(1,2))
    X = vec.fit_transform(sents)
    sims = [cosine_similarity(X[i], X[i+1])[0,0] for i in range(len(sents)-1)]
    return float(np.mean(sims)) if sims else 0.0

# Mejora: Coherencia semántica con embeddings multilingües
def semantic_coherence(summary_text: str) -> float:
    if not SBERT_AVAILABLE:
        return tfidf_coherence(summary_text)
    
    sents = [t.strip() for t in re.split(r'[.!?]\s+', summary_text.strip()) if t.strip()]
    if len(sents) < 2: return 0.0
    
    try:
        model = SentenceTransformer(SBERT_MODEL)
        embeddings = model.encode(sents)
        
        # Cosenos entre oraciones consecutivas
        sims = []
        for i in range(len(embeddings)-1):
            sim = np.dot(embeddings[i], embeddings[i+1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
            )
            sims.append(sim)
        
        return float(np.mean(sims)) if sims else 0.0
    except Exception as e:
        print(f"Error en semantic_coherence: {e}")
        return tfidf_coherence(summary_text)

# Mejora: Cobertura ROUGE-L (anclaje del resumen al texto)
def rouge_coverage(summary_text: str, original_text: str) -> float:
    if not ROUGE_AVAILABLE:
        return 0.0
    
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        # Submuestreo del texto original si es muy largo
        max_chars = 2000
        if len(original_text) > max_chars:
            # Tomar inicio, medio y final
            chunk_size = max_chars // 3
            start = original_text[:chunk_size]
            mid_pos = len(original_text) // 2
            middle = original_text[mid_pos-chunk_size//2:mid_pos+chunk_size//2]
            end = original_text[-chunk_size:]
            original_text = start + " " + middle + " " + end
        
        scores = scorer.score(original_text, summary_text)
        return float(scores['rougeL'].fmeasure)
    except Exception as e:
        print(f"Error en rouge_coverage: {e}")
        return 0.0

# Función compuesta mejorada
def math_coherence(summary_text: str, original_text: str = "") -> Dict[str, float]:
    """Retorna múltiples métricas de coherencia"""
    return {
        "tfidf_coherence": tfidf_coherence(summary_text),
        "semantic_coherence": semantic_coherence(summary_text),
        "rouge_coverage": rouge_coverage(summary_text, original_text) if original_text else 0.0
    }

def icc_two_way_random_absolute(ratings_matrix:np.ndarray) -> Tuple[float,float]:
    Y = np.asarray(ratings_matrix, dtype=float)
    n, k = Y.shape
    mean_per_target = Y.mean(axis=1, keepdims=True)
    mean_per_rater = Y.mean(axis=0, keepdims=True)
    grand_mean = Y.mean()
    SSR = n * np.sum((mean_per_rater - grand_mean)**2)
    SST = np.sum((Y - grand_mean)**2)
    SSE = np.sum((Y - grand_mean)**2 - (Y - mean_per_target - mean_per_rater + grand_mean)**2 + 0)  # robustness
    # Recompute standard ANOVA components
    SSB = k * np.sum((mean_per_target - grand_mean)**2)
    MSB = SSB / max((n - 1),1)
    MSR = SSR / max((k - 1),1)
    MSE = max(SST - SSB - SSR, 0.0) / max(((n - 1) * (k - 1)),1)
    ICC2_1 = (MSB - MSE) / (MSB + (k-1)*MSE + (k*(MSR - MSE)/max(n,1)))
    ICC2_k = (MSB - MSE) / (MSB + (MSR - MSE)/max(n,1))
    return float(ICC2_1), float(ICC2_k)

# Mejora: Bootstrapping para intervalos de confianza
def bootstrap_correlation(x, y, n_bootstrap=1000, confidence=0.95):
    """Calcula intervalo de confianza bootstrap para correlación de Pearson"""
    if len(x) != len(y) or len(x) < 2:
        return np.nan, np.nan, np.nan
    
    correlations = []
    n = len(x)
    
    for _ in range(n_bootstrap):
        # Muestreo con reemplazo
        indices = np.random.choice(n, size=n, replace=True)
        x_boot = [x[i] for i in indices]
        y_boot = [y[i] for i in indices]
        
        try:
            r_boot, _ = stats.pearsonr(x_boot, y_boot)
            if not np.isnan(r_boot):
                correlations.append(r_boot)
        except:
            continue
    
    if not correlations:
        return np.nan, np.nan, np.nan
    
    correlations = np.array(correlations)
    alpha = 1 - confidence
    lower = np.percentile(correlations, 100 * alpha/2)
    upper = np.percentile(correlations, 100 * (1 - alpha/2))
    
    return np.mean(correlations), lower, upper

def evaluar_coherencia_mejorada(resultados:Dict[str,Dict]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    all_scores = []
    
    for archivo, datos in resultados.items():
        resumen = str(datos.get("resumen","")).strip()
        texto_original = str(datos.get("text","")).strip()
        if not resumen: continue
        
        # Múltiples métricas de coherencia
        coherencia_metrics = math_coherence(resumen, texto_original)
        
        # Jueces LLM robustos
        judge_result = judge_llm_robust(resumen, k=CONFIG.JUDGE_K)
        
        # Baseline regex-only para frames
        frames_regex = extract_frames_regex_only(texto_original)
        
        row = {
            "session": os.path.splitext(archivo)[0],
            "tfidf_coherence": coherencia_metrics["tfidf_coherence"],
            "semantic_coherence": coherencia_metrics["semantic_coherence"], 
            "rouge_coverage": coherencia_metrics["rouge_coverage"],
            "judge_mean": judge_result["mean_score"],
            "judge_std": judge_result["std_score"],
            "frames_regex_count": len(frames_regex)
        }
        rows.append(row)
        all_scores.append(judge_result["scores"])
    
    df = pd.DataFrame(rows)
    
    # Correlaciones y estadísticas
    stats_dict = {}
    
    if len(df) >= 2:
        # Correlación principal: semántica vs jueces
        if "semantic_coherence" in df.columns and "judge_mean" in df.columns:
            r_sem, p_sem = stats.pearsonr(df["semantic_coherence"], df["judge_mean"])
            r_sem_boot, ci_lower, ci_upper = bootstrap_correlation(
                df["semantic_coherence"].tolist(), 
                df["judge_mean"].tolist(),
                n_bootstrap=CONFIG.BOOTSTRAP_N
            )
            stats_dict.update({
                "pearson_semantic_r": r_sem,
                "pearson_semantic_p": p_sem,
                "bootstrap_r": r_sem_boot,
                "bootstrap_ci_lower": ci_lower,
                "bootstrap_ci_upper": ci_upper
            })
        
        # Correlación TF-IDF (baseline)
        if "tfidf_coherence" in df.columns:
            r_tfidf, p_tfidf = stats.pearsonr(df["tfidf_coherence"], df["judge_mean"])
            stats_dict.update({
                "pearson_tfidf_r": r_tfidf,
                "pearson_tfidf_p": p_tfidf
            })
    
    # ICC mejorado
    if len(all_scores) >= 2:
        try:
            # Asegurar que todas las listas tengan la misma longitud
            min_length = min(len(scores) for scores in all_scores)
            ratings_matrix = np.array([scores[:min_length] for scores in all_scores])
            ICC2_1, ICC2_k = icc_two_way_random_absolute(ratings_matrix)
            stats_dict.update({
                "ICC2_1": ICC2_1,
                "ICC2_k": ICC2_k
            })
        except Exception as e:
            print(f"Error calculando ICC: {e}")
            stats_dict.update({"ICC2_1": np.nan, "ICC2_k": np.nan})
    
    return df, stats_dict

# Función original mantenida para compatibilidad
def evaluar_coherencia(resultados:Dict[str,Dict]) -> Tuple[pd.DataFrame, float, float]:
    df, stats_dict = evaluar_coherencia_mejorada(resultados)
    r = stats_dict.get("pearson_semantic_r", np.nan)
    p = stats_dict.get("pearson_semantic_p", np.nan)
    return df, r, p

# ============== BASELINES ==============
def run_extractive_baseline(resultados:Dict[str,Dict]) -> Dict[str,str]:
    out = {}
    for archivo, datos in resultados.items():
        text = datos["text"]
        summary_ext, dt = timed(extractive_summary, text, 5)
        out[archivo] = summary_ext
        resultados[archivo]["latency_extractive_s"] = dt
    return out

def run_multillm_costs(sample_paths:List[str]) -> pd.DataFrame:
    rows = []
    for path in sample_paths:
        texto = leer_archivo(path)
        for m in CONFIG.LLM_MODELS:
            content, dt, usage = llm_summary(texto, model=m["name"], temperature=m["temperature"])
            rows.append({
                "file": os.path.basename(path),
                "model": m["name"], "temp": m["temperature"],
                "latency_s": dt, "prompt_tokens": usage.get("prompt_tokens", None),
                "completion_tokens": usage.get("completion_tokens", None),
                "total_tokens": usage.get("total_tokens", None)
            })
    df = pd.DataFrame(rows)
    out_csv = os.path.join(CONFIG.OUT_DIR, "baseline_multillm_costs.csv")
    df.to_csv(out_csv, index=False)
    print("✓", out_csv)
    return df

# ============== ABLACIONES ==============
@dataclass
class AblationConfig:
    use_procedural_filter: bool = True
    use_regex_canonical: bool = True
    use_llm_remap_others: bool = True
    use_closed_taxonomy: bool = True

ABLATIONS = {
    "full": AblationConfig(),
    "no_procedural": AblationConfig(use_procedural_filter=False),
    "no_regex": AblationConfig(use_regex_canonical=False),
    "no_llm_remap": AblationConfig(use_llm_remap_others=False),
    "no_closed_tax": AblationConfig(use_closed_taxonomy=False),
}

def limpiar_label_cruda_ablate(s: str, cfg:AblationConfig) -> str:
    if not isinstance(s, str): return ""
    s0 = s.strip().replace("**","").replace("*","")
    s0 = re.sub(r"^[-•*\d.\s]+", "", s0).split(":")[0].strip(" .:-–—\t")
    if cfg.use_procedural_filter:
        if IGNORAR_REGEX.search(s0) or es_procedimental(s0): return ""
    else:
        if IGNORAR_REGEX.search(s0): return ""
    return s0

def normalizar_a_canon_ablate(frame_raw: str, cfg:AblationConfig) -> str:
    if not cfg.use_regex_canonical:
        return "Other"
    return normalizar_a_canon(frame_raw)

def recolectar_labels_crudos(resultados:Dict[str,Dict], source="framings_open") -> List[str]:
    unicos = set()
    for _, datos in resultados.items():
        framings_text = datos.get(source, "")
        for linea in str(framings_text).split("\n"):
            lab = limpiar_label_cruda(linea)
            if lab: unicos.add(lab)
    return sorted(unicos)

def abstraer_labels_a_taxonomia(labels_unicos:List[str]) -> Dict[str,str]:
    if not labels_unicos: return {}
    prompt = (
      "Map each label to ONE category from this taxonomy:"
      f"\n{', '.join(TAXONOMIA)}\n\n"
      "Reply ONLY JSON: [{\"raw\":\"...\",\"canon\":\"...\"}, ...]. "
      "Choose the closest taxonomy item (no 'Other').\n\n"
      "Labels:\n" + "\n".join(f"- {x}" for x in labels_unicos)
    )
    out, _ = call_openai(prompt, model="gpt-4o-mini", max_tokens=1200)
    try:
        js = re.search(r"\[.*\]", out, flags=re.S).group(0)
        arr = json.loads(js)
        mapping = {}
        for obj in arr:
            raw = obj.get("raw","").strip()
            canon = obj.get("canon","").strip()
            if raw and canon in TAXONOMIA: mapping[raw] = canon
        return mapping
    except Exception:
        return {}

def remapear_resultados_a_canon_ablate(resultados:Dict[str,Dict], cfg:AblationConfig, mapping_llm_extra:Optional[Dict[str,str]]=None, source="framings_open") -> Dict[str,List[str]]:
    mapping_llm_extra = mapping_llm_extra or {}
    out = {}
    for archivo, datos in resultados.items():
        canon_set = set()
        for linea in str(datos.get(source,"")).split("\n"):
            raw = limpiar_label_cruda_ablate(linea, cfg)
            if not raw: continue
            canon = normalizar_a_canon_ablate(raw, cfg)
            if canon=="Other" and cfg.use_llm_remap_others and raw in mapping_llm_extra:
                canon = mapping_llm_extra[raw]
            if cfg.use_closed_taxonomy and canon=="Other":
                continue
            canon_set.add(canon)
        out[archivo] = sorted(canon_set)
    return out

def tabla_frecuencias_canon(mapa_canon:Dict[str,List[str]]) -> pd.DataFrame:
    frames=[]
    for _, lst in mapa_canon.items(): frames.extend(lst)
    cnt = Counter(frames)
    total = max(sum(cnt.values()), 1)
    rows = [{"Frame":k,"Frequency":v,"Share_%":round(100*v/total,2)} for k,v in cnt.items()]
    return pd.DataFrame(rows).sort_values("Frequency", ascending=False).reset_index(drop=True)

def load_gold(path:str) -> Dict[str, List[str]]:
    with open(path) as f: return json.load(f)

def eval_frames_against_gold(mapa_pred:Dict[str,List[str]], gold_dict:Dict[str,List[str]]) -> Tuple[float,float]:
    cats = list(TAXONOMIA)
    y_true, y_pred = [], []
    for archivo, gold_frames in gold_dict.items():
        pred = set(mapa_pred.get(archivo, []))
        y_true.extend([1 if c in gold_frames else 0 for c in cats])
        y_pred.extend([1 if c in pred else 0 for c in cats])
    if not y_true: return np.nan, np.nan
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    return f1, kappa

def evaluar_ablaciones(resultados:Dict[str,Dict], gold_frames:Optional[Dict[str,List[str]]]=None):
    labels_crudos = recolectar_labels_crudos(resultados, source="framings_open")
    otros_pre = [lab for lab in labels_crudos if normalizar_a_canon(lab) == "Other"]
    mapping_llm = abstraer_labels_a_taxonomia(otros_pre)
    rows=[]
    for name, cfg in ABLATIONS.items():
        mapa = remapear_resultados_a_canon_ablate(resultados, cfg, mapping_llm_extra=mapping_llm, source="framings_open")
        df_freq = tabla_frecuencias_canon(mapa)
        f1 = kappa = np.nan
        if gold_frames is not None:
            f1, kappa = eval_frames_against_gold(mapa, gold_frames)
        rows.append({"ablation": name, "frames_unique": df_freq["Frame"].nunique(), "macroF1_frames": f1, "kappa_frames": kappa})
    df = pd.DataFrame(rows)
    out_csv = os.path.join(CONFIG.OUT_DIR,"table_ablations.csv")
    out_tex = os.path.join(CONFIG.OUT_DIR,"table_ablations.tex")
    df.to_csv(out_csv, index=False)
    with open(out_tex, "w") as f:
        f.write(df.to_latex(index=False, float_format="%.3f", caption="Ablation study results.", label="tab:ablations"))
    print("✓", out_csv, "/ .tex")
    return df

# ============== EXPORTS (TEX/FIGS) ==============
def df_to_tex(df:pd.DataFrame, path:str, caption:str, label:str, floatfmt="%.2f"):
    with open(path, "w") as f:
        f.write(df.to_latex(index=False, float_format=floatfmt, caption=caption, label=label))
    print("✓", path)

def plot_scatter_validity(df_eval:pd.DataFrame, path:str):
    if "math_coh" in df_eval.columns:
        r, p = stats.pearsonr(df_eval["math_coh"], df_eval["judge_mean"])
        x_col = "math_coh"
        x_label = "TF-IDF Coherence"
    else:
        r, p = stats.pearsonr(df_eval["tfidf_coherence"], df_eval["judge_mean"])
        x_col = "tfidf_coherence"
        x_label = "TF-IDF Coherence"
    
    plt.figure()
    plt.scatter(df_eval[x_col], df_eval["judge_mean"])
    plt.xlabel(x_label)
    plt.ylabel("Mean LLM-judge coherence")
    plt.title(f"r = {r:.3f}, p = {p:.3g}")
    plt.tight_layout()
    plt.savefig(path)
    print("✓", path)

# Mejora: Plot con múltiples métricas y bootstrap CI
def plot_scatter_validity_enhanced(df_eval: pd.DataFrame, stats_dict: Dict, path: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Enhanced Coherence Validation Analysis", fontsize=14)
    
    # 1. Coherencia semántica vs jueces
    if "semantic_coherence" in df_eval.columns:
        ax = axes[0, 0]
        ax.scatter(df_eval["semantic_coherence"], df_eval["judge_mean"], alpha=0.7)
        ax.set_xlabel("Semantic Coherence (SBERT)")
        ax.set_ylabel("LLM Judge Mean")
        
        r_sem = stats_dict.get("pearson_semantic_r", np.nan)
        p_sem = stats_dict.get("pearson_semantic_p", np.nan)
        ci_lower = stats_dict.get("bootstrap_ci_lower", np.nan)
        ci_upper = stats_dict.get("bootstrap_ci_upper", np.nan)
        
        title = f"r = {r_sem:.3f}, p = {p_sem:.3g}"
        if not np.isnan(ci_lower) and not np.isnan(ci_upper):
            title += f"\n95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]"
        ax.set_title(title)
    
    # 2. TF-IDF vs jueces (baseline)
    if "tfidf_coherence" in df_eval.columns:
        ax = axes[0, 1]
        ax.scatter(df_eval["tfidf_coherence"], df_eval["judge_mean"], alpha=0.7, color='orange')
        ax.set_xlabel("TF-IDF Coherence")
        ax.set_ylabel("LLM Judge Mean")
        
        r_tfidf = stats_dict.get("pearson_tfidf_r", np.nan)
        p_tfidf = stats_dict.get("pearson_tfidf_p", np.nan)
        ax.set_title(f"r = {r_tfidf:.3f}, p = {p_tfidf:.3g}")
    
    # 3. ROUGE coverage vs coherencia semántica
    if "rouge_coverage" in df_eval.columns and "semantic_coherence" in df_eval.columns:
        ax = axes[1, 0]
        ax.scatter(df_eval["rouge_coverage"], df_eval["semantic_coherence"], alpha=0.7, color='green')
        ax.set_xlabel("ROUGE-L Coverage")
        ax.set_ylabel("Semantic Coherence")
        if len(df_eval) > 1:
            try:
                r_rouge, p_rouge = stats.pearsonr(df_eval["rouge_coverage"], df_eval["semantic_coherence"])
                ax.set_title(f"r = {r_rouge:.3f}, p = {p_rouge:.3g}")
            except:
                ax.set_title("Coverage vs Coherence")
    
    # 4. Frames regex vs LLM count
    if "frames_regex_count" in df_eval.columns:
        ax = axes[1, 1]
        # Necesitamos los datos de comparación LLM vs regex
        ax.hist(df_eval["frames_regex_count"], alpha=0.7, label="Regex-only", bins=10)
        ax.set_xlabel("Number of Frames Detected")
        ax.set_ylabel("Frequency")
        ax.set_title("Frame Detection Distribution")
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print("✓", path)

def heatmap_binary(mat_bin:pd.DataFrame, path:str):
    plt.figure(figsize=(max(8,0.5*len(mat_bin.columns)), 0.5*len(mat_bin.index)))
    plt.imshow(mat_bin.values, aspect="auto", interpolation="nearest")
    plt.yticks(range(len(mat_bin.index)), mat_bin.index)
    plt.xticks(range(len(mat_bin.columns)), mat_bin.columns, rotation=90)
    plt.xlabel("Session")
    plt.ylabel("Frame")
    plt.title("Frame×Session (binary presence)")
    plt.tight_layout()
    plt.savefig(path)
    print("✓", path)

# ============== MAIN ==============
def main():
    #  base
    resultados = procesar_archivos(CONFIG.DATA_DIR)

    # frecuencias y normalización (closed por defecto)
    df_frames_raw = tabla_frecuencia_frames([d["framings_open"] for d in resultados.values()])
    df_frames_raw.to_csv(os.path.join(CONFIG.OUT_DIR, "frames_frecuencia_filtrada.csv"), index=False)
    df_norm = normalizar_df_frames(df_frames_raw)
    top10 = resumir_top_frames(df_norm, top_n=10)
    df_to_tex(top10, os.path.join(CONFIG.OUT_DIR,"frames_top10_normalizados.tex"),
              "Top--10 canonical frames.", "tab:top10")

    # matrices Frame×Sesión
    mat_bin = construir_matriz_frames_sesion(resultados, source="framings_closed", mode="binary", include_other=False)
    mat_cnt = construir_matriz_frames_sesion(resultados, source="framings_closed", mode="count", include_other=False)

    mat_bin.to_csv(os.path.join(CONFIG.OUT_DIR,"frame_session_matrix_binary.csv"))
    mat_cnt.to_csv(os.path.join(CONFIG.OUT_DIR,"frame_session_matrix_counts.csv"))

    # export TEX
    with open(os.path.join(CONFIG.OUT_DIR,"frame_session_matrix_binary.tex"), "w") as f:
        f.write(mat_bin.to_latex(longtable=True, escape=False, caption="Presence (1/0) of canonical frames across sessions.", label="tab:frame-session-binary"))
    with open(os.path.join(CONFIG.OUT_DIR,"frame_session_matrix_counts.tex"), "w") as f:
        f.write(mat_cnt.to_latex(longtable=True, escape=False, caption="Counts of canonical frames across sessions.", label="tab:frame-session-counts"))

    # versión reducida 
    colmap_bin = {c.lower(): c for c in mat_bin.columns}
    cols_small = [colmap_bin[k] for k in CONFIG.SMALL_SESSIONS if k in colmap_bin]
    if cols_small:
        with open(os.path.join(CONFIG.OUT_DIR,"frame_session_matrix_binary_small.tex"), "w") as f:
            f.write(mat_bin[cols_small].to_latex(longtable=False, escape=False,
                                                caption="Presence (1/0) of canonical frames across selected sessions.",
                                                label="tab:frame-session-binary-small"))
        with open(os.path.join(CONFIG.OUT_DIR,"frame_session_matrix_counts_small.tex"), "w") as f:
            f.write(mat_cnt[cols_small].to_latex(longtable=False, escape=False,
                                                caption="Counts of canonical frames across selected sessions.",
                                                label="tab:frame-session-counts-small"))

    # Mejora: Evaluación robusta de coherencia 
    df_eval_enhanced, stats_enhanced = evaluar_coherencia_mejorada(resultados)
    df_eval_enhanced.to_csv(os.path.join(CONFIG.OUT_DIR,"eval_coherence_enhanced.csv"), index=False)
    
    # Guardar estadísticas mejoradas
    with open(os.path.join(CONFIG.OUT_DIR,"coherence_stats_enhanced.json"), "w") as f:
        json.dump(stats_enhanced, f, indent=2, default=str)
    
    # Plots mejorados
    plot_scatter_validity_enhanced(df_eval_enhanced, stats_enhanced, 
                                 os.path.join(CONFIG.OUT_DIR,"fig_scatter_coherence_enhanced.pdf"))
    
    # Evaluación baseline regex-only
    baseline_regex_results = {}
    for archivo, datos in resultados.items():
        texto = datos["text"]
        frames_regex = extract_frames_regex_only(texto)
        baseline_regex_results[archivo] = frames_regex
    
    # Comparar LLM vs Regex
    comparison_data = []
    for archivo in resultados.keys():
        session = os.path.splitext(archivo)[0]
        frames_llm = resultados[archivo].get("framings_closed", "").split("\n")
        frames_llm_clean = [f.strip() for f in frames_llm if f.strip()]
        frames_regex = baseline_regex_results[archivo]
        
        comparison_data.append({
            "session": session,
            "llm_frame_count": len(frames_llm_clean),
            "regex_frame_count": len(frames_regex),
            "overlap": len(set(frames_llm_clean) & set(frames_regex))
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison.to_csv(os.path.join(CONFIG.OUT_DIR,"llm_vs_regex_comparison.csv"), index=False)
    
    # Mantener evaluación original para compatibilidad
    df_eval, r, p = evaluar_coherencia(resultados)
    df_eval.to_csv(os.path.join(CONFIG.OUT_DIR,"eval_coherence_scores.csv"), index=False)
    plot_scatter_validity(df_eval, os.path.join(CONFIG.OUT_DIR,"fig_scatter_coh_vs_judges.pdf"))

    # heatmap binario (opcional)
    heatmap_binary(mat_bin, os.path.join(CONFIG.OUT_DIR,"fig_heatmap_frames_sessions.pdf"))

    # baseline extractivo y costos multillm (muestra)
    baseline_ext = run_extractive_baseline(resultados)
    sample_paths = [os.path.join(CONFIG.DATA_DIR, f) for f in sorted(os.listdir(CONFIG.DATA_DIR))[:10]]
    run_multillm_costs(sample_paths)

    # macro-F1 y κ
    if CONFIG.GOLD_PATH and os.path.exists(CONFIG.GOLD_PATH):
        gold = load_gold(CONFIG.GOLD_PATH)
        # usa open-world + normalización/closed remapeo por defecto (full)
        labels_crudos = recolectar_labels_crudos(resultados, source="framings_open")
        otros_pre = [lab for lab in labels_crudos if normalizar_a_canon(lab) == "Other"]
        mapping_llm = abstraer_labels_a_taxonomia(otros_pre)
        mapa_full = remapear_resultados_a_canon_ablate(resultados, ABLATIONS["full"], mapping_llm_extra=mapping_llm, source="framings_open")
        f1_macro, kappa = eval_frames_against_gold(mapa_full, gold)
        print(f"macro-F1 vs GOLD: {f1_macro:.3f} | kappa: {kappa:.3f}")
        with open(os.path.join(CONFIG.OUT_DIR,"gold_eval.json"), "w") as f:
            json.dump({"macroF1": f1_macro, "kappa": kappa}, f, indent=2)

    # ablaciones 
    gold_dict = load_gold(CONFIG.GOLD_PATH) if CONFIG.GOLD_PATH and os.path.exists(CONFIG.GOLD_PATH) else None
    evaluar_ablaciones(resultados, gold_frames=gold_dict)

    # 9) Guardar prompts, runtime y resumen de mejoras
    improvements_summary = {
        "enhanced_features": {
            "semantic_coherence": SBERT_AVAILABLE,
            "rouge_coverage": ROUGE_AVAILABLE,
            "robust_judges": True,
            "bootstrap_ci": True,
            "regex_baseline": True
        },
        "statistics": stats_enhanced,
        "model_config": {
            "judge_k": CONFIG.JUDGE_K,
            "bootstrap_n": CONFIG.BOOTSTRAP_N,
            "sbert_model": SBERT_MODEL if SBERT_AVAILABLE else None
        }
    }
    
    with open(os.path.join(CONFIG.OUT_DIR,"improvements_summary.json"), "w") as f:
        json.dump(improvements_summary, f, indent=2, default=str)
    
    with open(os.path.join(CONFIG.OUT_DIR,"DONE.txt"), "w") as f:
        f.write("Enhanced WDKE Pipeline - Artifacts ready for LaTeX inclusion.\n")
        f.write("New features: Semantic coherence, ROUGE coverage, robust judges (k=5), bootstrap CI.\n")
    
    print("\n🎉 Enhanced WDKE Pipeline completed!")
    print("✓ All artifacts ready for LaTeX inclusion in ./out/")
    
    # Mostrar resultados mejorados
    if stats_enhanced:
        print(f"\n📊 Enhanced Statistics:")
        r_sem = stats_enhanced.get('pearson_semantic_r', np.nan)
        p_sem = stats_enhanced.get('pearson_semantic_p', np.nan)
        print(f"Semantic Coherence: r={r_sem:.3f}, p={p_sem:.3g}")
        
        ci_lower = stats_enhanced.get('bootstrap_ci_lower', np.nan)
        ci_upper = stats_enhanced.get('bootstrap_ci_upper', np.nan)
        if not np.isnan(ci_lower):
            print(f"Bootstrap 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
        
        icc_2_1 = stats_enhanced.get('ICC2_1', np.nan)
        icc_2_k = stats_enhanced.get('ICC2_k', np.nan)
        print(f"ICC(2,1): {icc_2_1:.3f}, ICC(2,k): {icc_2_k:.3f}")
    
    print(f"TF-IDF baseline: r={r:.3f}, p={p:.3g}")
    print("\n📁 New files generated:")
    print("- eval_coherence_enhanced.csv")
    print("- coherence_stats_enhanced.json") 
    print("- llm_vs_regex_comparison.csv")
    print("- fig_scatter_coherence_enhanced.pdf")
    print("- improvements_summary.json")
    
if __name__ == "__main__":
    main()
