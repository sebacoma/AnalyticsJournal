# -*- coding: utf-8 -*-
"""
WDKE Pipeline end-to-end (Conda + VS Code):
- Resúmenes LLM y extractivos
- Extracción de frames (open vs closed)
- Normalización a taxonomía
- Baselines y Ablaciones
- Fiabilidad (ICC LLM-judges), κ y macro-F1 vs GOLD (opcional)
- Validez (Pearson coherencia-LLM judges)
- Costos/latencia/prompts/semillas/versiones
- Export CSV, TEX, PDF (para LaTeX \input y \includegraphics)

Requisitos:
  - environment.yml (Conda)
  - OPENAI_API_KEY en el entorno (no hardcodear)

Ajusta rutas en CONFIG abajo.
"""
import os, sys, re, json, time, random, platform, unicodedata
from dataclasses import dataclass
from collections import Counter
from typing import Dict, List, Tuple, Optional

# Cargar variables de entorno desde .env
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

# ============== CONFIG ==============
class CONFIG:
    SEED = 42
    DATA_DIR = "data/WDKE"          # carpeta con .txt de debates
    OUT_DIR = "out"                 # salidas CSV/TEX/PDF
    MAX_TEXT_CHARS = 16000          # truncamiento de seguridad para LLM
    LLM_MODELS = [                  # >=3 modelos si tienes acceso
        {"name": "gpt-4o-mini", "temperature": 0.0},
        {"name": "gpt-4o-mini", "temperature": 0.7},
        # {"name": "gpt-4.1-mini", "temperature": 0.0},
        # {"name": "gpt-4o", "temperature": 0.0},
    ]
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
    # Intentar diferentes encodings para manejar archivos con formatos diversos
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    # Si ninguno funciona, usar 'utf-8' con errores ignorados
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

def judge_llm(summary_text, k=3, model="gpt-4o-mini"):
    scores = []
    usages = []
    for _ in range(k):
        p = PROMPTS["judge_prompt"].replace("{SUMMARY}", summary_text)
        out, usage = call_openai(p, model=model, temperature=0.0, max_tokens=5)
        m = re.search(r"[1-5]", out)
        if m: scores.append(int(m.group(0)))
        usages.append(usage)
    if not scores: scores = [3]
    return float(np.mean(scores)), scores, usages

# ============== PIPELINE BASE ==============
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

def math_coherence(summary_text:str) -> float:
    sents = [t.strip() for t in re.split(r'[.!?]\s+', summary_text.strip()) if t.strip()]
    if len(sents) < 2: return 0.0
    vec = TfidfVectorizer(min_df=1, ngram_range=(1,2))
    X = vec.fit_transform(sents)
    sims = [cosine_similarity(X[i], X[i+1])[0,0] for i in range(len(sents)-1)]
    return float(np.mean(sims)) if sims else 0.0

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

def evaluar_coherencia(resultados:Dict[str,Dict]) -> Tuple[pd.DataFrame, float, float]:
    rows = []
    all_scores = []
    for archivo, datos in resultados.items():
        resumen = str(datos.get("resumen","")).strip()
        if not resumen: continue
        mcoh = math_coherence(resumen)
        judge_mean, judge_scores, _ = judge_llm(resumen, k=3)
        rows.append({"session": os.path.splitext(archivo)[0], "math_coh": mcoh, "judge_mean": judge_mean})
        all_scores.append(judge_scores)
    df = pd.DataFrame(rows)
    r, p = (np.nan, np.nan)
    if len(df) >= 2:
        r, p = stats.pearsonr(df["math_coh"], df["judge_mean"])
    ICC2_1 = ICC2_k = np.nan
    if len(all_scores) >= 2:
        ratings = np.asarray(all_scores)  # N x K
        ICC2_1, ICC2_k = icc_two_way_random_absolute(ratings)
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
    r, p = stats.pearsonr(df_eval["math_coh"], df_eval["judge_mean"])
    plt.figure()
    plt.scatter(df_eval["math_coh"], df_eval["judge_mean"])
    plt.xlabel("Automatic coherence")
    plt.ylabel("Mean LLM-judge coherence")
    plt.title(f"r = {r:.3f}, p = {p:.3g}")
    plt.tight_layout()
    plt.savefig(path)
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
    # 1) pipeline base
    resultados = procesar_archivos(CONFIG.DATA_DIR)

    # 2) frecuencias y normalización (closed por defecto)
    df_frames_raw = tabla_frecuencia_frames([d["framings_open"] for d in resultados.values()])
    df_frames_raw.to_csv(os.path.join(CONFIG.OUT_DIR, "frames_frecuencia_filtrada.csv"), index=False)
    df_norm = normalizar_df_frames(df_frames_raw)
    top10 = resumir_top_frames(df_norm, top_n=10)
    df_to_tex(top10, os.path.join(CONFIG.OUT_DIR,"frames_top10_normalizados.tex"),
              "Top--10 canonical frames.", "tab:top10")

    # 3) matrices Frame×Sesión
    mat_bin = construir_matriz_frames_sesion(resultados, source="framings_closed", mode="binary", include_other=False)
    mat_cnt = construir_matriz_frames_sesion(resultados, source="framings_closed", mode="count", include_other=False)

    mat_bin.to_csv(os.path.join(CONFIG.OUT_DIR,"frame_session_matrix_binary.csv"))
    mat_cnt.to_csv(os.path.join(CONFIG.OUT_DIR,"frame_session_matrix_counts.csv"))

    # export TEX
    with open(os.path.join(CONFIG.OUT_DIR,"frame_session_matrix_binary.tex"), "w") as f:
        f.write(mat_bin.to_latex(longtable=True, escape=False, caption="Presence (1/0) of canonical frames across sessions.", label="tab:frame-session-binary"))
    with open(os.path.join(CONFIG.OUT_DIR,"frame_session_matrix_counts.tex"), "w") as f:
        f.write(mat_cnt.to_latex(longtable=True, escape=False, caption="Counts of canonical frames across sessions.", label="tab:frame-session-counts"))

    # versión reducida para el cuerpo del paper
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

    # 4) evaluación de coherencia + jueces LLM + ICC + Pearson
    df_eval, r, p = evaluar_coherencia(resultados)
    df_eval.to_csv(os.path.join(CONFIG.OUT_DIR,"eval_coherence_scores.csv"), index=False)
    plot_scatter_validity(df_eval, os.path.join(CONFIG.OUT_DIR,"fig_scatter_coh_vs_judges.pdf"))

    # 5) heatmap binario (opcional)
    heatmap_binary(mat_bin, os.path.join(CONFIG.OUT_DIR,"fig_heatmap_frames_sessions.pdf"))

    # 6) baseline extractivo y costos multillm (muestra)
    baseline_ext = run_extractive_baseline(resultados)
    sample_paths = [os.path.join(CONFIG.DATA_DIR, f) for f in sorted(os.listdir(CONFIG.DATA_DIR))[:10]]
    run_multillm_costs(sample_paths)

    # 7) GOLD opcional -> macro-F1 y κ
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

    # 8) ablaciones (con o sin GOLD)
    gold_dict = load_gold(CONFIG.GOLD_PATH) if CONFIG.GOLD_PATH and os.path.exists(CONFIG.GOLD_PATH) else None
    evaluar_ablaciones(resultados, gold_frames=gold_dict)

    # 9) guardar prompts y runtime (ya guardado)
    with open(os.path.join(CONFIG.OUT_DIR,"DONE.txt"), "w") as f:
        f.write("Artifacts ready for LaTeX inclusion.\n")
    print("\n✓ All artifacts ready for LaTeX inclusion in ./out/")
    print(f"Pearson r={r:.3f}, p={p:.3g}")
    print("Insert in LaTeX: \\input{frame_session_matrix_binary.tex}, etc.")
    
if __name__ == "__main__":
    main()
