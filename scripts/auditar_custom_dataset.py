#!/usr/bin/env python3
"""
DS Guardian - Auditoría e Integración de Datasets Personalizados
================================================================
Permite al usuario seleccionar o ingresar la ruta de cualquier dataset
(CSV, Excel) de la carpeta Descargas o cualquier otra ruta local,
especificar la variable objetivo (target) y ejecutar el pipeline
completo de gobierno de datos y auditoría de DS Guardian.
"""

import os
import sys
import glob
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.utils.multiclass import type_of_target

# Ignorar advertencias menores de numpy en consola
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Asegurar importación de ds_guardian
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ds_guardian import eda, limpieza, modelos, auditoria, visualizacion

# Códigos de colores ANSI para consola
C_GREEN = '\033[92m'
C_RED = '\033[91m'
C_YELLOW = '\033[93m'
C_BLUE = '\033[94m'
C_BOLD = '\033[1m'
C_CYAN = '\033[96m'
C_RESET = '\033[0m'

def buscar_datasets():
    """Busca archivos CSV y Excel en Descargas, data/, archive/ y directorio actual."""
    rutas_busqueda = [
        Path.home() / "Descargas",
        Path.home() / "Downloads",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "archive",
        PROJECT_ROOT
    ]
    archivos = []
    extensiones = ('*.csv', '*.xlsx', '*.xls')
    
    for folder in rutas_busqueda:
        if folder.exists() and folder.is_dir():
            for ext in extensiones:
                archivos.extend(list(folder.glob(ext)))
                
    # Eliminar duplicados y ordenar por fecha de modificación (más reciente primero)
    archivos_unicos = sorted(list(set(archivos)), key=lambda x: x.stat().st_mtime, reverse=True)
    return archivos_unicos

def cargar_dataframe(path):
    """Carga un archivo CSV o Excel manejando diferentes codificaciones."""
    path = str(path)
    if path.endswith(('.xlsx', '.xls')):
        return pd.read_excel(path)
    elif path.endswith('.csv'):
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding='latin1')
    else:
        raise ValueError(f"Formato no soportado para el archivo: {path}")

def ejecutar_auditoria_custom():
    print(f"{C_BOLD}{C_BLUE}============================================================")
    print("      🛡️ DS GUARDIAN - CARGA Y AUDITORÍA DE DATASETS")
    print(f"============================================================{C_RESET}\n")

    archivos_encontrados = buscar_datasets()
    
    print(f"{C_BOLD}Seleccione el dataset a auditar:{C_RESET}")
    for idx, f in enumerate(archivos_encontrados[:10], start=1):
        rel_path = f.relative_to(Path.home()) if Path.home() in f.parents else f
        print(f" [{idx}] ~/{rel_path}")
        
    num_opcion_manual = min(11, len(archivos_encontrados) + 1)
    print(f" [{num_opcion_manual}] 📁 Ingresar ruta de archivo manualmente")
    
    print("-" * 60)
    eleccion = input(f"Ingrese una opción [1-{num_opcion_manual}]: ").strip()

    filepath = None
    if eleccion.isdigit():
        idx_sel = int(eleccion)
        if 1 <= idx_sel <= len(archivos_encontrados[:10]):
            filepath = archivos_encontrados[idx_sel - 1]
        elif idx_sel == num_opcion_manual:
            ruta_input = input("\nIngrese la ruta completa del archivo (.csv o .xlsx): ").strip()
            filepath = Path(os.path.expanduser(ruta_input))
            
    if not filepath or not filepath.exists():
        print(f"\n{C_RED}❌ Archivo no encontrado o ruta inválida: {filepath}{C_RESET}")
        return

    print(f"\n{C_CYAN}📂 Cargando dataset desde '{filepath}'...{C_RESET}")
    try:
        df = cargar_dataframe(filepath)
    except Exception as e:
        print(f"{C_RED}❌ Error al leer el archivo: {e}{C_RESET}")
        return

    nombre_dataset = filepath.stem
    print(f"{C_GREEN}✅ Dataset cargado correctamente.{C_RESET}")
    print(f"   -> Filas: {df.shape[0]} | Columnas: {df.shape[1]}\n")

    # Mostrar columnas disponibles
    columnas = list(df.columns)
    print(f"{C_BOLD}Columnas encontradas en el dataset:{C_RESET}")
    for idx, col in enumerate(columnas, start=1):
        nulos_cnt = df[col].isnull().sum()
        dtype_str = str(df[col].dtype)
        print(f"  {idx:2d}. {col:30s} (Tipo: {dtype_str:10s} | Nulos: {nulos_cnt})")

    print("-" * 60)
    target_sel = input("\n🎯 Seleccione el número o nombre de la columna OBJETIVO (Target): ").strip()
    
    columna_target = None
    if target_sel.isdigit():
        idx_t = int(target_sel)
        if 1 <= idx_t <= len(columnas):
            columna_target = columnas[idx_t - 1]
    elif target_sel in columnas:
        columna_target = target_sel

    if not columna_target:
        print(f"{C_RED}❌ Columna objetivo no válida.{C_RESET}")
        return

    print(f"{C_GREEN}🎯 Columna seleccionada como Target: '{columna_target}'{C_RESET}")

    # Detectar o preguntar por columnas a ignorar (como IDs)
    cols_id_potenciales = [c for c in columnas if 'id' in c.lower() or c.lower() in ['index', 'nro', 'codigo']]
    if cols_id_potenciales and columna_target not in cols_id_potenciales:
        print(f"\n{C_YELLOW}💡 Se detectaron posibles columnas identificadoras (IDs): {cols_id_potenciales}{C_RESET}")
        drop_ids = input("¿Desea eliminarlas automáticamente para prevenir fugas/memorización? (S/n): ").strip().lower()
        if drop_ids != 'n':
            df = df.drop(columns=[c for c in cols_id_potenciales if c in df.columns])
            print(f"{C_GREEN}  -> Columnas ID eliminadas.{C_RESET}")

    # Detección inteligente del tipo de tarea (Clasificación vs Regresión)
    y_raw = df[columna_target].dropna()
    tipo_target_sklearn = type_of_target(y_raw)
    
    if tipo_target_sklearn in ['continuous', 'continuous-multioutput'] or (pd.api.types.is_float_dtype(y_raw) and y_raw.nunique() > 10):
        sugerencia_clasif = False
        nombre_tarea = "Regresión (Valores Numéricos Continuos)"
    else:
        sugerencia_clasif = True
        nombre_tarea = "Clasificación (Etiquetas Discretas)"

    print(f"\n{C_CYAN}🧠 Detección de Target: La variable '{columna_target}' se identificó como: {nombre_tarea}{C_RESET}")
    print(f"   Valores únicos en target: {y_raw.nunique()} | Tipo de dato: {y_raw.dtype}")
    
    cambiar = input("¿Desea mantener esta sugerencia? ([S]/n): ").strip().lower()
    es_clasificacion = sugerencia_clasif
    if cambiar == 'n':
        opc_t = input("Elija tipo de tarea ([1] Clasificación, [2] Regresión): ").strip()
        es_clasificacion = (opc_t == '1')

    # PIPELINE DE DS GUARDIAN
    print("\n" + "=" * 60)
    print("🚀 EJECUTANDO PIPELINE DE GOBERNANZA DS GUARDIAN")
    print("=" * 60)

    # 1. EDA y Optimización de Memoria
    print(f"\n{C_CYAN}📊 1. MÓDULO EDA (Optimización de Memoria & Outliers){C_RESET}")
    df = eda.optimizar_memoria(df)
    
    # Generar Mapa de Calor de Correlaciones (Mejora 3)
    corr_plot_path = f"plots/{nombre_dataset}_correlation.png"
    visualizacion.plot_correlacion(df, save_path=corr_plot_path)

    # Outlier Capping en columnas numéricas de características
    cols_num = df.select_dtypes(include=[np.number]).columns.drop(columna_target, errors='ignore').tolist()
    if cols_num:
        df = eda.acotar_outliers_iqr(df, columnas=cols_num)

    # 2. Separación de Variables y Train/Test Split
    print(f"\n{C_CYAN}✂️ 2. SEPARACIÓN TRAIN / TEST (Previene Data Leakage){C_RESET}")
    X = df.drop(columns=[columna_target])
    y = df[columna_target]

    # Eliminar filas con nulos en el target si existen
    if y.isnull().any():
        idx_valid = y.dropna().index
        X = X.loc[idx_valid]
        y = y.loc[idx_valid]

    stratify_arg = y if es_clasificacion and y.nunique() <= 10 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_arg
    )
    print(f"   -> Train set: {X_train.shape[0]} muestras")
    print(f"   -> Test set:  {X_test.shape[0]} muestras")

    # 3. Limpieza e Imputación segura
    print(f"\n{C_CYAN}🧹 3. MÓDULO LIMPIEZA (Imputación & Codificación aislada){C_RESET}")
    X_train, X_test = limpieza.imputar_nulos(X_train, X_test)
    X_train, X_test = limpieza.codificar_variables(X_train, X_test)

    # 4. Auditoría QA
    print(f"\n{C_CYAN}🕵️ 4. MÓDULO AUDITORÍA (Control de Calidad QA){C_RESET}")
    detalle_auditoria = auditoria.revisar_datos_finales(X_train, y=y_train, retornar_detalle=True)
    es_valido = detalle_auditoria.get('aprobado', False)

    if not es_valido:
        print(f"\n{C_RED}❌ La auditoría detectó errores críticos. Entrenamiento cancelado.{C_RESET}")
        return

    # Comprobar opción de Tuning de Hiperparámetros (Mejora 4)
    hacer_tuning = input(f"\n{C_YELLOW}⚙️ ¿Desea aplicar Optimización Automática de Hiperparámetros (Random Search)? (s/[N]): {C_RESET}").strip().lower() == 's'

    # 5. Entrenamiento y Evaluación
    print(f"\n{C_CYAN}🤖 5. MÓDULO MODELOS (Entrenamiento, Explicabilidad & Exportación){C_RESET}")
    plot_path = f"plots/{nombre_dataset}_eval.png"
    metricas = {}
    modelo_final = None

    if es_clasificacion:
        try:
            # Opción de Balanceo de Clases (Mejora 2)
            class_weight = 'balanced' if detalle_auditoria.get('clases_balanceadas') is False else None
            if class_weight:
                print(f"{C_GREEN}⚖️ Aplicando ajuste automático de pesos ('class_weight=balanced') por desbalance detectado.{C_RESET}")

            clf_base = RandomForestClassifier(random_state=42, class_weight=class_weight)
            
            if hacer_tuning:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
                clf = modelos.optimizar_hiperparametros(clf_base, param_grid, X_train, y_train, cv=3, n_iter=5)
            else:
                clf = clf_base
                clf.fit(X_train, y_train)

            modelo_final = clf
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None
            
            modelos.evaluar_clasificacion(y_test, y_pred, y_prob=y_prob, save_path=plot_path)
            
            # Graficar Curva ROC (Mejora 3)
            if y_prob is not None:
                roc_path = f"plots/{nombre_dataset}_roc_curve.png"
                visualizacion.plot_curva_roc(y_test, y_prob, save_path=roc_path)

            acc = accuracy_score(y_test, y_pred)
            metricas['Accuracy'] = acc
            try:
                f1 = f1_score(y_test, y_pred, average='weighted')
                metricas['F1-Score (Weighted)'] = f1
            except Exception:
                pass

            top_feat = modelos.graficar_importancia_caracteristicas(
                clf, list(X_train.columns), top_n=10, save_path=f"plots/{nombre_dataset}_importance.png"
            )
            
            auditoria.registrar_y_comparar_modelo(
                nombre_proyecto=nombre_dataset,
                metrica_principal_nombre="Accuracy",
                valor_metrica=acc
            )
        except ValueError as err:
            if "continuous" in str(err).lower():
                print(f"\n{C_YELLOW}⚠️ El target contiene valores numéricos continuos. Cambiando automáticamente a Regresión...{C_RESET}")
                es_clasificacion = False

    if not es_clasificacion:
        reg_base = RandomForestRegressor(random_state=42)
        if hacer_tuning:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
            reg = modelos.optimizar_hiperparametros(reg_base, param_grid, X_train, y_train, cv=3, n_iter=5, scoring='r2')
        else:
            reg = reg_base
            reg.fit(X_train, y_train)

        modelo_final = reg
        y_pred = reg.predict(X_test)
        modelos.evaluar_regresion(y_test, y_pred)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        metricas['R2 Score'] = r2
        metricas['RMSE'] = rmse
        
        top_feat = modelos.graficar_importancia_caracteristicas(
            reg, list(X_train.columns), top_n=10, save_path=f"plots/{nombre_dataset}_importance.png"
        )
        
        auditoria.registrar_y_comparar_modelo(
            nombre_proyecto=nombre_dataset,
            metrica_principal_nombre="R2 Score",
            valor_metrica=r2
        )

    # 6. Exportación de Modelo Binario y Metadatos JSON (Mejora 1)
    rutas_modelo = modelos.guardar_modelo_entrenado(
        modelo=modelo_final,
        feature_names=list(X_train.columns),
        nombre_proyecto=nombre_dataset,
        metricas=metricas
    )

    # 7. Generación de Reportes Markdown y HTML Interactivo (Mejora 5)
    df_info = {
        'filas': len(df),
        'columnas': len(df.columns),
        'nulos': X_train.isnull().sum().sum(),
        'memoria_mb': f"{df.memory_usage().sum() / (1024 * 1024):.2f}"
    }
    
    reporte_md_path = f"reportes/reporte_auditoria_{nombre_dataset}.md"
    auditoria.generar_reporte_auditoria_markdown(
        nombre_proyecto=nombre_dataset,
        df_info=df_info,
        resultado_auditoria=es_valido,
        metricas_modelo=metricas,
        top_features=top_feat,
        output_path=reporte_md_path,
        detalle_auditoria=detalle_auditoria
    )

    reporte_html_path = f"reportes/reporte_auditoria_{nombre_dataset}.html"
    auditoria.generar_reporte_auditoria_html(
        nombre_proyecto=nombre_dataset,
        df_info=df_info,
        resultado_auditoria=es_valido,
        metricas_modelo=metricas,
        top_features=top_feat,
        output_path=reporte_html_path
    )

    print("\n" + "=" * 60)
    print(f"{C_GREEN}{C_BOLD}🎉 PIPELINE ENTERPRISE DE DS GUARDIAN COMPLETADO CON ÉXITO PARA '{nombre_dataset}'{C_RESET}")
    print(f"📦 Modelo guardado en: {rutas_modelo['model_path']}")
    print(f"📊 Gráficos guardados en carpeta: plots/")
    print(f"📄 Reporte Markdown en: {reporte_md_path}")
    print(f"🌐 Reporte HTML Interactivo en: {reporte_html_path}")
    print("=" * 60)

if __name__ == '__main__':
    ejecutar_auditoria_custom()
