# Metra – Framework de Evaluación Multidimensional para LLMs

**Metra** es un marco integral y extensible diseñado para la evaluación de modelos de lenguaje de gran tamaño (LLMs), abordando no solo la **precisión**, sino también dimensiones críticas como la **factualidad**, la **ética** y la **equidad**. Este sistema permite realizar evaluaciones automáticas, transparentes y reproducibles mediante una interfaz de línea de comandos (CLI).

---

## Objetivo

El objetivo principal del proyecto es proporcionar una herramienta que supere las limitaciones de las métricas tradicionales, incorporando criterios éticos y sociales relevantes. A través de una arquitectura modular y datasets diseñados manualmente, Metra ofrece una visión más rica del comportamiento de los LLMs.

---

## Tecnologías principales

### Python
Desarrollado íntegramente en **Python**, utilizando entornos virtuales y gestionado desde **Visual Studio Code**. Esta elección permitió una integración fluida de librerías especializadas en NLP.

### Frameworks

- **RAGAS**: utilizado como base para extender funcionalidades y construir métricas personalizadas.
- **LangChain**: facilita la interacción con modelos locales de forma modular y escalable.

### Modelos LLMs
Los modelos evaluados (como **Mistral** o **LLaMA 3**) se ejecutan localmente mediante **Ollama**, lo que garantiza independencia de servicios comerciales y un coste nulo en inferencia.

### Jupyter Notebook
Utilizado para el análisis exploratorio y validación de las métricas, permitiendo iteración rápida y visualización de resultados.

### Gestión de entorno
Las dependencias están recogidas en `requirements.txt`. Se recomienda usar entorno virtual. El archivo `.gitignore` evita incluir carpetas como `venv/`, `__pycache__/` o archivos `.pyc`.

---

## Estructura del proyecto

```
metra/
├── data/                      # Datasets organizados por dimensión
│   ├── accuracy_datasets/
│   ├── factuality_datasets/
│   ├── ethics_datasets/
│   ├── fairness_datasets/
│   └── templates_datasets/
├── doc/                       # Análisis y documentación generada
├── notebooks/                # Jupyter notebooks para el análisis de los resultados
├── evaluator/                 # Núcleo del sistema de evaluación
│   ├── main.py                # CLI principal (comando `metra`)
│   └── custom_metrics/        # Métricas personalizadas
├── results/                   # Resultados generados automáticamente
├── requirements.txt           # Dependencias del sistema
├── pyproject.toml             # Configuración del CLI
└── README.md                  # Documentación del repositorio
```

---

## Instalación

```bash
git clone https://github.com/tuusuario/metra.git
cd metra
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -e .
```

---

## Ejecución

La evaluación se realiza mediante la CLI con el comando `metra`, definido en `pyproject.toml`.

### Argumentos obligatorios:
- `--metric`: métrica a evaluar (`accuracy`, `factuality`, `ethics`, `fairness`)
- `--model`: modelo a evaluar (`mistral`, `llama3`, etc.)

### Argumentos opcionales:
- `--output`: ruta personalizada para guardar los resultados

### Ejemplo:

```bash
metra --metric ethics --model mistral
```

Esto buscará:

- **Dataset**: `data/ethics_datasets/ethics_test_mistral.json`
- **Salida**: `results/ethics_results_mistral.json`

Durante la ejecución se muestran por terminal las puntuaciones individuales y medias finales.

---

## Métricas disponibles

| Métrica                      | Categoría     | Descripción                                                                                   |
|------------------------------|---------------|-----------------------------------------------------------------------------------------------|
| `hybrid_accuracy`            | Precisión     | Combina BERTScore y similitud de embeddings para medir coincidencia semántica.               |
| `composite_factuality`       | Factualidad   | Combina extracción de hechos (SVO) con verificación por LLM con justificación textual.        |
| `ethical_alignment`          | Ética         | Evalúa similitud con un benchmark ético y penaliza respuestas tóxicas con Toxic-BERT.         |
| `counterfactual_fairness`    | Equidad       | Evalúa justicia en pares contrafactuales usando puntuación de LLM + análisis de sentimiento.  |

---

## Derechos

Proyecto desarrollado como parte del Trabajo de Fin de Grado en Ingeniería Informática (Universidad de Málaga, 2025).
No se garantiza mantenimiento activo.

---

## Autoría

Desarrollado por **Tessa Luaces De Fazio**  
Tutor: **Francisco Emilio López Valverde**  
Universidad de Málaga – Trabajo de Fin de Grado, curso 2024/2025

---
