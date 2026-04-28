import argparse
import csv
import os
import torch 
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Analisi Quantitativa YOLO con Filtro Bordi - Tesi")
    parser.add_argument("--video", type=str, required=True, help="Percorso del video")
    parser.add_argument("--model", type=str, default="runs/yolo26n-seg-run/weights/best.pt", help="Percorso del file .pt")
    parser.add_argument("--out_csv", type=str, default="risultati_analisi_filtrati.csv", help="Nome del file CSV")
    parser.add_argument("--conf", type=float, default=0.25, help="Soglia di confidenza minima")
    args = parser.parse_args()

    # 1. Caricamento modello
    print(f"Caricamento del modello da: {args.model}")
    model = YOLO(args.model)

    # 2. Definizione del margine (20 pixel dai bordi)
    MARGINE_BORDO = 20 

    # 3. Preparazione file CSV
    with open(args.out_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'Oggetto_ID', 'Classe', 'Area_Pixel', 'Stato_Bordo'])

        # Esecuzione inferenza
        results = model.predict(source=args.video, conf=args.conf, device="cpu", stream=True, show=True, save=False)
        
        print(f"Inizio elaborazione video: {args.video}")
        
        for frame_idx, r in enumerate(results):
            # Recuperiamo le dimensioni del frame video corrente
            frame_height, frame_width = r.orig_shape

            if r.masks is not None:
                masks_data = r.masks.data
                boxes_data = r.boxes.xyxy  # Coordinate dei Bounding Box [x1, y1, x2, y2]
                classes = r.boxes.cls
                
                for i in range(len(masks_data)):
                    # Estraiamo le coordinate del rettangolo di delimitazione
                    x1, y1, x2, y2 = boxes_data[i].tolist()
                    
                    # --- LOGICA DEL MARGINE DI BORDO ---
                    # Verifichiamo se l'oggetto tocca la "zona proibita"
                    tocca_bordo = (x1 <= MARGINE_BORDO or 
                                   y1 <= MARGINE_BORDO or 
                                   x2 >= (frame_width - MARGINE_BORDO) or 
                                   y2 >= (frame_height - MARGINE_BORDO))
                    
                    if tocca_bordo:
                        # Se tocca il bordo, lo saltiamo e non scriviamo nel CSV
                        # In alternativa, puoi scriverlo con una nota "PARZIALE"
                        continue 
                    # -----------------------------------

                    # Se arriviamo qui, l'oggetto è interamente nell'inquadratura
                    area_pixel = masks_data[i].sum().item()
                    nome_classe = model.names[int(classes[i])]
                    
                    # Scrittura dati validi
                    writer.writerow([frame_idx, i, nome_classe, area_pixel, "INTERO"])
            
            if frame_idx % 50 == 0:
                print(f"Processati {frame_idx} frame...")

    print(f"Analisi completata. Dati salvati in: {args.out_csv}")

if __name__ == "__main__":
    main()