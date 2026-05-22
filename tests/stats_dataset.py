import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from skimage import feature
from skimage.filters.rank import entropy
from skimage.morphology import disk
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def calculate_stats_full_image(img_array, mask_array=None):
    """Calcola le statistiche sull'immagine intera."""
    if img_array is None: return None
    results = {}
    
    # Parametri di Base
    results['mean_brightness'] = np.mean(img_array)
    results['std_contrast'] = np.std(img_array)
    
    # Complessità della Texture (Entropia globale)
    # Nota: su immagini grandi l'entropia locale può essere lenta.
    # Se il processo è troppo lungo, usa: results['average_entropy'] = shannon_entropy(img_array)
    results['average_entropy'] = np.mean(entropy(img_array, disk(5)))

    # Omogeneità GLCM
    glcm = feature.graycomatrix(img_array, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    results['glcm_homogeneity'] = feature.graycoprops(glcm, 'homogeneity')[0, 0]

    # Analisi Anomalie (se presenti)
    if mask_array is not None and np.any(mask_array > 0):
        mask_bin = (mask_array > 127).astype(np.uint8)
        defect_pixels = np.sum(mask_bin)
        results['defect_area_ratio'] = (defect_pixels / img_array.size) * 100
        
        # Contrasto relativo dell'anomalia rispetto al resto dell'immagine
        mean_defect = np.mean(img_array[mask_bin > 0])
        mean_backgr = np.mean(img_array[mask_bin == 0]) if np.any(mask_bin == 0) else mean_defect
        results['anomaly_contrast_ratio'] = abs(mean_defect - mean_backgr) / (mean_backgr + 1e-6)
    else:
        results['defect_area_ratio'] = 0
        results['anomaly_contrast_ratio'] = 0
        
    return results

def analyze_mvtec_full(base_path):
    base_path = Path(base_path)
    all_data = []
    
    # MVTec structure: category/[train|test]/sub_type/file.png
    # Estensioni comuni in MVTec sono .png
    print(f"Scanning MVTec dataset at: {base_path}")
    train_files = list(base_path.glob("**/train/**/*.png"))
    test_files = list(base_path.glob("**/test/**/*.png"))
    image_files = train_files + test_files

    print(f"Found {len(image_files)} images in MVTec dataset.")
    
    for img_path in tqdm(image_files, desc="Processing MVTec"):
        # Split (train/test), Categoria (bottle, cable, etc.), Tipo (good, broken, etc.)
        split = img_path.parts[-3] 
        category = img_path.parts[-4]
        sub_type = img_path.parent.name
        
        # Percorso Maschera: ground_truth/category/sub_type/filename_mask.png
        mask_filename = img_path.stem + "_mask.png"
        mask_path = base_path / category / "ground_truth" / sub_type / mask_filename

        # Caricamento
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        # Gestione Maschera: MVTec non ha maschere per lo split 'train' o per 'test/good'
        mask = None
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Calcolo statistiche sull'immagine intera
        stats = calculate_stats_full_image(img, mask)
        
        if stats:
            stats['category'] = category
            stats['split'] = split
            stats['sub_type'] = sub_type
            stats['file_name'] = img_path.name
            all_data.append(stats)
                
    return pd.DataFrame(all_data)

def plot_and_save_stats(df, output_filename="report_statistico_mvtec.pdf"):
    if df.empty:
        print("Il DataFrame è vuoto. Impossibile generare il file.")
        return

    sns.set_theme(style="whitegrid")
    # Creiamo l'oggetto PDF
    with PdfPages(output_filename) as pdf:
        
        # --- PAGINA 1: Statistiche di Base ---
        fig1, axes1 = plt.subplots(1, 2, figsize=(18, 10))
        sns.boxplot(ax=axes1[0], data=df, x='category', y='mean_brightness', palette='Set2')
        axes1[0].set_title('Luminosità Media per Categoria e Split', fontsize=16)
        axes1[0].tick_params(axis='x', rotation=45)

        sns.boxplot(ax=axes1[1], data=df, x='category', y='std_contrast', palette='Set2')
        axes1[1].set_title('Variazione del Contrasto (Std Dev)', fontsize=16)
        axes1[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        pdf.savefig(fig1)  # Salva la prima pagina
        plt.close(fig1)

        # --- PAGINA 2: Complessità Texture ---
        fig2, axes2 = plt.subplots(1, 2, figsize=(18, 10))
        sns.violinplot(ax=axes2[0], data=df, x='category', y='average_entropy', palette='Pastel1')
        axes2[0].set_title('Complessità della Texture (Entropia)', fontsize=16)
        axes2[0].tick_params(axis='x', rotation=45)

        sns.boxplot(ax=axes2[1], data=df, x='category', y='glcm_homogeneity', palette='Pastel1')
        axes2[1].set_title('Omogeneità della Superficie (GLCM)', fontsize=16)
        axes2[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        pdf.savefig(fig2)  # Salva la seconda pagina
        plt.close(fig2)

        # --- PAGINA 3: Analisi Anomalie ---
        df_abnormal = df[df['anomaly_contrast_ratio'] > 0].copy()
        if not df_abnormal.empty:
            fig3, axes3 = plt.subplots(1, 2, figsize=(18, 10))
            sns.stripplot(ax=axes3[0], data=df_abnormal, x='category', y='defect_area_ratio', 
                          hue='category', jitter=True, alpha=0.6, palette='bright', legend=False)
            axes3[0].set_title('Dimensione dei Difetti (% Area)', fontsize=16)
            axes3[0].tick_params(axis='x', rotation=45)

            sns.boxplot(ax=axes3[1], data=df_abnormal, x='category', y='anomaly_contrast_ratio', palette='bright',showfliers=False)
            axes3[1].set_title('Rapporto di Contrasto dell\'Anomalia', fontsize=16)
            axes3[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            pdf.savefig(fig3)  # Salva la terza pagina
            plt.close(fig3)

    print(f"Grafici salvati correttamente in: {output_filename}")


def get_patches(img, mask, patch_size=224):
    """
    Divide l'immagine a metà (sinistra e destra) e ridimensiona 
    ogni metà a patch_size x patch_size.
    """
    h, w = img.shape[:2] # Originale: 270, 480
    mid = w // 2         # Punto di divisione: 240

    # 1. Divisione in due metà
    # Metà Sinistra (270x240)
    left_img_full = img[:, 0:mid]
    left_mask_full = mask[:, 0:mid]

    # Metà Destra (270x240)
    right_img_full = img[:, mid:w]
    right_mask_full = mask[:, mid:w]

    # 2. Resize a 224x224
    # Per l'immagine usiamo INTER_AREA (ottimo per downsampling leggero)
    patch1_img = cv2.resize(left_img_full, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
    patch2_img = cv2.resize(right_img_full, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

    # Per la maschera usiamo INTER_NEAREST per mantenere la label binaria pulita
    patch1_mask = cv2.resize(left_mask_full, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)
    patch2_mask = cv2.resize(right_mask_full, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)

    return [(patch1_img, patch1_mask), (patch2_img, patch2_mask)]


def analyze_industrial_dataset(base_path, dataset_name="Dataset", patch = False):
    all_stats = []
    base_path = Path(base_path)

    # Cerchiamo tutte le immagini nelle sottocartelle 'images/train'
    # La struttura attesa è: category/images/train/file.jpg
    image_files = list(base_path.glob("**/images/train/*.*"))

    print(f"Analisi {dataset_name}: trovate {len(image_files)} immagini.")

    for img_path in tqdm(image_files):
        
        if patch:
            category = "CPS-AD2D_Patch"
        else:
            category = "CPS-AD2D"

        # Gestione percorsi con eccezione per cps_6
        if category == "cps_6":
            mask_path = img_path.parents[2] / "annotations" / "defect" / f"{img_path.stem}.png"
        else:
            mask_path = img_path.parents[2] / "annotations" / "train" / f"{img_path.stem}.png"

        # Caricamento
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None

        if img is not None:
            # Se non c'è maschera (es. immagini normali), creiamo una maschera vuota (nera)
            if mask is None:
                mask = np.zeros_like(img)

            # Generazione delle 2 Patch
            if patch:
                patches = get_patches(img, mask, patch_size=224)
            else:
                patches = [(img, mask)]

            for i, (p_img, p_mask) in enumerate(patches):
                # Calcolo statistiche sulla singola patch
                # Usiamo una versione leggermente modificata di calculate_image_stats
                # che accetta l'array numpy invece del path
                stats = calculate_stats_full_image(p_img, p_mask)

                if stats:
                    stats['category'] = category
                    stats['patch_id'] = f"{img_path.stem}_patch_{i}"
                    all_stats.append(stats)

    return pd.DataFrame(all_stats)




# Esecuzione
df_cpsad2d = analyze_industrial_dataset("/Users/nicolaberti/Documents/Datasets/CPS-AD2D", dataset_name="CPS-AD2D", patch=False)
df_cpsad2d_patch = analyze_industrial_dataset("/Users/nicolaberti/Documents/Datasets/CPS-AD2D", dataset_name="CPS-AD2D_Patch", patch=True)
df_mvtec = analyze_mvtec_full("/Users/nicolaberti/Documents/Datasets/mvtec_anomaly_detection")

# Uniamo i DataFrame per un'analisi comparativa
df1 = pd.concat([df_mvtec, df_cpsad2d, df_cpsad2d_patch], ignore_index=True)

#Salva in tre file CSV per eventuali analisi future
df_mvtec.to_csv("stats_mvtec.csv", index=False)
df_cpsad2d.to_csv("stats_cpsad2d.csv", index=False)
df_cpsad2d_patch.to_csv("stats_cpsad2d_patch.csv", index=False)

# Esecuzione
plot_and_save_stats(df1, "Analisi_Statistica_MVTec.pdf")