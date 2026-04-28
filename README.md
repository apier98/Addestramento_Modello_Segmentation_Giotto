README - Visualizzazione e validazione annotazioni COCO

Scopo
------
Script di debug per validare visivamente le annotazioni COCO (bbox e segmentation) sovrapponendole alle immagini. Utile per verificare la qualità delle annotazioni prima dell'addestramento del modello di segmentazione.

File principale
--------------
- visualize_coco_debug.py: script Python per renderizzare annotazioni su immagini e salvarle in una cartella scelta.

Requisiti
---------
- Python 3.8+
- opencv-python
- pillow (opzionale)
- pycocotools (opzionale, raccomandato per RLE masks)

Installazione delle dipendenze (esempio):

pip install opencv-python pillow pycocotools

Uso
---
Esempio di esecuzione:

python visualize_coco_debug.py --coco annotations.json --data_dir Data --outdir annotated_out --num 20

Argomenti principali:
- --coco: percorso al file JSON in formato COCO (default: annotations.json)
- --data_dir: cartella contenente le immagini (default: Data)
- --outdir: cartella di destinazione per le immagini annotate (obbligatorio)
- --num: numero di immagini da processare
- --seed: seme per la scelta dei colori
- --shuffle: mescola le immagini prima della selezione

Come funziona
-------------
- Legge il COCO JSON e costruisce una mappa immagine->annotazioni.
- Per ogni immagine selezionata disegna bounding box e segmentation (poligoni o RLE se pycocotools è disponibile).
- Applica un'alpha blend per rendere le sovrapposizioni trasparenti e salva l'immagine in outdir.

Note su Windows
---------------
- Lo script usa metodi compatibili con percorsi Unicode su Windows (cv2.imencode + tofile) per evitare problemi con caratteri non ASCII.

Problemi comuni
---------------
- "COCO file not found": verifica il percorso passato a --coco
- "Image not found": controlla che file_name nelle immagini del JSON corrisponda ai file in Data (può essere relativo o assoluto)
- Se le maschere RLE non si visualizzano, installare pycocotools

Prossimi passi (da aggiungere in seguito)
-----------------------------------------
- Interfaccia grafica semplice per scegliere immagini e categorie
- Salvataggio di report (es. immagine + elenco annotazioni mancanti o con bbox fuori immagine)
- Tool per correggere annotazioni direttamente dalle immagini

Conversione COCO -> formato YOLO (segmentazione)
-------------------------------------------------
È incluso uno script di utilità: coco_to_yolo_seg.py
Questo script converte le annotazioni COCO (poligoni e, se pycocotools è installato, RLE) nel formato richiesto da Ultralytics per training di segmentazione.
Configurazione consigliata:
- Ratio train/val: 80/20 (impostazione predefinita nello script)

Esempio di esecuzione:

python coco_to_yolo_seg.py --coco annotations.json --src_dir Data\\frame_tesi --out_dir dataset_yolo --train_ratio 0.8 --seed 42 --copy_images

Lo script crea la struttura:

dataset_yolo/
  images/train
  images/val
  labels/train
  labels/val
  data.yaml

Requisiti (opzionali ma raccomandati):
- pycocotools (per decodificare le maschere RLE e convertirle in poligoni)
- opencv-python (usato per contorni/decodifica e operazioni sulle immagini)

Dopo la conversione, puoi avviare l'addestramento Ultralytics (esempio):

python train_yolo26n_seg.py --data dataset_yolo/data.yaml --model yolo26n-seg.pt --epochs 100 --imgsz 640 --batch 16 --device 0 --project runs --name yolo26n-seg-run

Lo script train_yolo26n_seg.py usa l'API Python di Ultralytics per avviare il training della variante nano (yolo26n-seg). Assicurati di avere installato:

pip install ultralytics>=2.6

Comando alternativo (CLI yolo):

yolo train task=segment model=yolo26n-seg.pt data=dataset_yolo/data.yaml epochs=100 imgsz=640

Dopo l'addestramento i risultati saranno nella cartella specificata (--project/--name).

Organizzazione cartelle locali (stato del repository locale)
----------------------------------------------------------
Questa è la struttura attuale del progetto così com'è sul sistema di sviluppo. Condividerla con i colleghi aiuta a mantenere coerenza.

Root del progetto:
- Data/                      -> cartella contenente i dataset, immagini e sottocartelle (es. Data\\frame_tesi). (ESEGUITO: contiene i file di training; normalmente esclusa da Git)
- Data\\frame_tesi\\        -> immagini sorgente usate per le annotazioni e la conversione (118 items)
- visualize_coco_debug.py     -> script per visualizzare e validare annotazioni COCO
- coco_to_yolo_seg.py        -> script per convertire COCO -> formato YOLO (segmentazione)
- train_yolo26n_seg.py       -> script per avviare l'addestramento YOLO 2.6 nano (seg)
- yolo26n-seg.pt             -> peso modello nano (se presente localmente; file binario grande)
- debug/                     -> directory per strumenti di debug (logs, immagini di esempio)
- README.md                  -> questo file

Elementi tipicamente esclusi da Git (consigliato aggiungerli a .gitignore):
- Data/ (dataset e immagini) -- non committare dataset di grandi dimensioni
- runs/ (cartelle di training, checkpoints, logs)
- *.pt, *.pth (pesi grandi) -- se scaricabili dal model zoo, non salvarli nel repo
- .env, .vscode/, .idea/, __pycache__/, *.pyc
- dataset_yolo/ (se generato localmente e grande)

Esempio di file .gitignore consigliato (da adattare):

# Dataset and outputs
Data/
runs/
dataset_yolo/

# Model weights
*.pt
*.pth

# Python
__pycache__/
*.pyc

# IDE
.vscode/
.idea/

# OS
Thumbs.db
.DS_Store

Note:
- Se alcuni pesi (.pt) devono essere condivisi, caricarli su uno storage centrale (Google Drive, S3, o Git LFS per file grandi) e documentare il link nel README.

Contatti
--------
Progetto: Addestramento_Modello_Segmentation_Giotto

Inference script: detect_video_seg.py
------------------------------------
A convenience script to run segmentation inference on a video file or webcam and save annotated output (masks, boxes, labels). It uses ultralytics.YOLO and OpenCV.

Requirements:
- pip install ultralytics>=2.6 opencv-python pyyaml

Usage examples:
python detect_video_seg.py --source input.mp4 --weights runs\\yolo26n-seg-run\\weights\\best.pt --out out.mp4
python detect_video_seg.py --source 0 --weights runs\\yolo26n-seg-run\\weights\\best.pt --out out.mp4

Notes:
- If --data is not provided, the script attempts to locate Data\\dataset_yolo\\data.yaml referenced in runs/*/args.yaml to load class names.
- Default weights path is runs\\yolo26n-seg-run\\weights\\best.pt (override with --weights).
- Use --show to display a live window while processing.
