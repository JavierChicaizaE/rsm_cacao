# RSM · Tostado de Cacao Nacional 🍫

Aplicativo interactivo de **Metodología de Superficie de Respuesta (RSM)** para la
optimización del proceso de tostado de cacao Nacional (Ecuador), desarrollado en
Python + Streamlit.

Permite a un usuario no especialista: generar diseños experimentales (CCD / Box-Behnken),
ajustar modelos de 1er y 2do orden, revisar diagnósticos de residuos, optimizar el proceso
(análisis canónico, ascenso más pronunciado, análisis de cresta y deseabilidad de
Derringer-Suich para múltiples respuestas) y visualizar los resultados (contornos,
superficies 3D, Pareto y perturbación).

> ⚠️ **Nota de transparencia:** este proyecto usa un dataset **sintético**, generado con
> tendencias basadas en literatura científica real sobre tostado de cacao (ver
> `data_generator.py` y la declaración en el reporte técnico). No sustituye datos de
> laboratorio.

## Estructura del repositorio

```
rsm_cacao/
├── app.py                 # App Streamlit (interfaz de usuario, 6 pestañas)
├── rsm_core.py             # Motor estadístico RSM (diseños, modelos, ANOVA, optimización)
├── data_generator.py       # Generador del dataset sintético de tostado de cacao
├── requirements.txt        # Dependencias
├── .streamlit/config.toml  # Tema visual de la app
├── data/
│   └── cacao_ccd_sintetico.csv   # Dataset de prueba (20 corridas, CCD)
└── README.md
```

## Instalación y ejecución local

```bash
# 1. Clonar el repositorio
git clone <URL_DE_TU_REPO>
cd rsm_cacao

# 2. Crear entorno virtual (recomendado)
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la app
streamlit run app.py
```

La app se abre automáticamente en `http://localhost:8501`.

## Despliegue gratuito (para obtener el link de acceso público)

**Opción recomendada: Streamlit Community Cloud**

1. Sube este proyecto a un repositorio de GitHub (público).
2. Entra a https://share.streamlit.io con tu cuenta de GitHub.
3. Clic en **"New app"** → selecciona tu repositorio, la rama (`main`) y el archivo
   principal (`app.py`).
4. Clic en **"Deploy"**. En 2-3 minutos tendrás una URL pública tipo
   `https://tu-app.streamlit.app` — ese es el link de acceso que debes entregar.

**Alternativas:** Hugging Face Spaces (SDK: Streamlit) o Render.com (Web Service,
comando de arranque `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`).

## Uso rápido

1. **① Datos y Diseño** — genera el CCD sintético (o Box-Behnken), o sube tu propio CSV.
2. **② Ajuste del Modelo** — selecciona las respuestas y ajusta modelos de 1er/2do orden;
   revisa R², R² ajustado, ANOVA y la prueba de falta de ajuste.
3. **③ Diagnóstico** — inspecciona residuos (vs. predichos, Q-Q normal, vs. orden de corrida).
4. **④ Optimización** — análisis canónico (máximo/mínimo/silla), ascenso más pronunciado,
   análisis de cresta, y optimización multi-respuesta con deseabilidad de Derringer-Suich.
5. **⑤ Visualización** — contornos, superficies 3D, Pareto de efectos, perturbación.
6. **⑥ Exportar** — descarga un reporte consolidado en Excel (datos, coeficientes, ANOVA,
   óptimo de deseabilidad).

## Metodología y respuestas del caso

| Factor | Rango |
|---|---|
| Temperatura de tostado | 120 – 160 °C |
| Tiempo de tostado | 10 – 30 min |
| Humedad inicial del grano | 5 – 8 % |

| Respuesta | Objetivo típico |
|---|---|
| Polifenoles totales (mg GAE/g) | Maximizar |
| Actividad antioxidante DPPH (%) | Maximizar |
| Índice de pardeamiento (color) | Valor objetivo (target) |
| Puntaje sensorial (1-9) | Maximizar |

## Autoría y declaración de IA

Ver la sección "Declaración de uso de IA" en el reporte técnico (PDF) entregado junto
con este repositorio.
